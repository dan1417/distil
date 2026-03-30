#!/usr/bin/env python3
"""
This evaluates KL(teacher || student) using full-distribution GPU forward passes.

Key features:
  - Full-distribution KL on 248K vocab (not top-k approximation)
  - Teacher continuation: generates 512 tokens, scores on continuation positions
  - Teacher continuations pre-generated ONCE per epoch
  - seeded prompt selection (unpredictable, reproducible)
  - Same-tokenizer enforcement (exact encoding match)
  - Model sanity check (forward pass verification after load)
  - Student load timeout (300s default)
  - VRAM monitoring at key points
  - MoE-aware param counting
"""
import os
import sys
import time
import json
import gc
import signal
import logging
import traceback
import threading
from pathlib import Path

import click
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.evalidator")

# ── Constants ──────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
DEFAULT_MAX_PARAM_RATIO = 0.15  # Students ≤ 15% of teacher ≈ 5.25B total
MAX_KL_THRESHOLD = 2.0  # Quality floor — reject if KL above this (good distill ~0.1-0.5)
EMA_ALPHA = 0.3
MAX_EVAL_PER_EPOCH = 5  # Max new models to evaluate per epoch
MAX_NEW_TOKENS = 512  # Teacher continuation length
MAX_PROMPT_TOKENS = 1024
STATE_DIR = Path("state")
STUDENT_LOAD_TIMEOUT = 300  # 5 minutes max for student model download/load


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def log_vram(label: str = ""):
    """Log current VRAM usage."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            prefix = f"VRAM [{label}]" if label else "VRAM"
            logger.info(f"{prefix}: {used:.1f}/{total:.1f}GB")
    except ImportError:
        pass


def model_sanity_check(model, tokenizer, device):
    """
    Quick sanity check: run a forward pass on a short test input and verify
    output logits are reasonable (not NaN, not all zeros, std > 0.1).

    Catches broken uploads, corrupted weights, quantized models that slipped
    past config check.
    """
    import torch
    test_ids = tokenizer("def hello():\n    return", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(test_ids).logits
    if torch.isnan(logits).any():
        return False, "broken_logits: NaN values detected in output"
    if torch.isinf(logits).any():
        return False, "broken_logits: Inf values detected in output"
    if logits.std() < 0.1:
        return False, f"broken_logits: std={logits.std().item():.4f} < 0.1 (near-constant output)"
    return True, "ok"


def load_model_with_timeout(model_repo, revision, device, dtype, timeout_seconds=STUDENT_LOAD_TIMEOUT):
    """
    Load a HuggingFace model with a timeout. Uses threading to avoid
    signal.alarm issues with non-main threads.

    Returns (model, None) on success, (None, error_message) on failure.
    """
    import torch
    from transformers import AutoModelForCausalLM

    result = [None]
    error = [None]

    def _load():
        try:
            result[0] = AutoModelForCausalLM.from_pretrained(
                model_repo,
                revision=revision,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=_load)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread still running — timeout hit
        return None, f"Model load timed out after {timeout_seconds}s"

    if error[0] is not None:
        return None, f"Model load failed: {error[0]}"

    return result[0], None


@click.command()
@click.option("--teacher-model", default=TEACHER_MODEL)
@click.option("--max-param-ratio", type=float, default=DEFAULT_MAX_PARAM_RATIO)
@click.option("--dataset-path", default="./dataset")
@click.option("--samples-per-epoch", type=int, default=80)
@click.option("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
@click.option("--max-eval-per-epoch", type=int, default=MAX_EVAL_PER_EPOCH)
@click.option("--tempo", type=int, default=360, help="Seconds between evaluation epochs")
@click.option("--state-dir", type=click.Path(), default=str(STATE_DIR))
@click.option("--student-load-timeout", type=int, default=STUDENT_LOAD_TIMEOUT,
              help="Timeout in seconds for student model download/load")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
def main(
    model_repo, revision, teacher_model, max_param_ratio,
    dataset_path, samples_per_epoch, max_new_tokens, max_eval_per_epoch,
    tempo, state_dir, student_load_timeout, log_level,
):
    
    logging.getLogger().setLevel(getattr(logging, log_level))
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.kl_divergence import generate_teacher_continuations, evaluate_student_kl
    from eval.dataset import sample_prompts_from_dataset, format_prompt
    from eval.model_checker import (
        check_model_architecture, compute_model_hash,
        check_duplicate_hash, register_model_hash,
        verify_tokenizer,
    )
    

    max_student_params_b = TEACHER_TOTAL_PARAMS_B * max_param_ratio
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # ── Load dataset ───────────────────────────────────────────────────
    logger.info("Prompts sampled fresh from full dataset each epoch")

    # ── Load teacher model (kept resident) ─────────────────────────────
    logger.info(f"Loading teacher model: {teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    teacher.eval()
    log_vram("after teacher load")
    logger.info("Teacher model loaded and resident in GPU memory")

    # ── Main loop ──────────────────────────────────────────────────────
    while True:
        try:
            # ── seeded prompt selection ──────────────────────────
            import random

            # Generate a random integer (this will be a large number)
            seed = random.randint(1, 100000)

            epoch_prompts = sample_prompts_from_dataset(samples_per_epoch, seed)
            prompt_texts = [format_prompt(p) for p in epoch_prompts]
            logger.info(f"Selected {len(prompt_texts)} prompts (seed: {seed})")

            # ── Tokenize prompts ──────────────────────────────────────
            input_ids_list = []
            for text in prompt_texts:
                ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS,
                ).input_ids.to(device)
                input_ids_list.append(ids)

            # ── Pre-generate teacher continuations ONCE for this epoch ─
            logger.info("Generating teacher continuations (once for all students)...")
            teacher_cache = generate_teacher_continuations(
                teacher, input_ids_list,
                max_new_tokens=max_new_tokens,
                block_seed=seed,
                device=device,
            )
            logger.info(f"Cached {len(teacher_cache)} teacher continuations")
            log_vram("after teacher continuation generation")

            # ── Evaluate each student model ───────────────────────────
            try:
                # 1. Architecture check
                check = check_model_architecture(model_repo, revision, max_student_params_b)
                if not check["pass"]:
                    reason = check["reason"]
                    # Standardized error messages
                    if "too_large" in reason:
                        params_b = check.get("params_b", 0)
                        logger.warning(
                            f"Model too large: "
                            f"{params_b:.2f}B > {max_student_params_b:.1f}B max"
                        )
                    elif "vocab_mismatch" in reason:
                        vocab = check.get("vocab_size", "?")
                        logger.warning(
                            f"Model REJECTED: Vocab size mismatch: "
                            f"{vocab} ≠ {248044} (teacher)"
                        )
                    elif "quantized" in reason:
                        logger.warning(
                            f"Model REJECTED: Quantized model detected — "
                            f"Requires bf16/fp16 architecture distillation"
                        )
                    else:
                        logger.warning(f"Model REJECTED: {reason}")
                    continue

                # 2. Tokenizer verification
                tok_ok, tok_reason = verify_tokenizer(teacher_model, model_repo)
                if not tok_ok:
                    logger.warning(
                        f"Model REJECTED: Tokenizer mismatch: {tok_reason}"
                    )
                    continue

                # 3. Load student (with timeout)
                logger.info(
                    f"Evaluating Model: {model_repo}@"
                    f"{revision[:12] if revision else 'main'} "
                    f"({check.get('params_b', 0):.2f}B total)"
                )
                log_vram("before student load")

                student, load_err = load_model_with_timeout(
                    model_repo, revision, device,
                    dtype=torch.bfloat16,
                    timeout_seconds=student_load_timeout,
                )
                if load_err:
                    logger.warning(f"Model REJECTED: {load_err}")
                    continue

                student.eval()
                log_vram("after student load")

                # 4. Sanity check — verify model produces valid logits
                sane, sane_reason = model_sanity_check(student, tokenizer, device)
                if not sane:
                    logger.warning(
                        f"Model REJECTED: Sanity check failed: {sane_reason}"
                    )
                    continue

                # 5. KL evaluation using cached teacher continuations
                kl_results = []
                for i, cache_entry in enumerate(teacher_cache):
                    result = evaluate_student_kl(student, cache_entry, device)
                    kl_results.append(result)
                    logger.debug(
                        f"  Prompt {i}: KL={result['kl_mean']:.4f} "
                        f"(gen_len={result['gen_len']}, positions={result['n_positions']})"
                    )

                # Weighted average by number of positions
                total_positions = sum(r["n_positions"] for r in kl_results)
                if total_positions == 0:
                    logger.warning(f"No positions evaluated")
                    continue

                avg_kl = sum(
                    r["kl_mean"] * r["n_positions"] for r in kl_results
                ) / total_positions

                logger.info(
                    f"Avg KL={avg_kl:.6f} "
                    f"({total_positions} total positions across {len(kl_results)} prompts)"
                )

            except Exception as e:
                logger.error(f"Model evaluation failed: {e}")
                traceback.print_exc()

            finally:
                if student is not None:
                    del student
                free_gpu()
                log_vram("after student cleanup")

           
            logger.info(f"Epoch complete, sleeping {tempo}s")
            time.sleep(tempo)

        except KeyboardInterrupt:
            logger.info("Shutting down — persisting state")
            break

        except Exception as e:
            logger.error(f"Epoch error: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
