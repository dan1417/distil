#!/usr/bin/env python3
import os
import sys
import json
import logging

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distillation.model_check")

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15  # ~5.25B max


@click.command()
@click.option("--model-repo", required=True, help="HuggingFace repo e.g. 'user/distilled-qwen'")
@click.option("--revision", default=None, help="HF commit SHA (pinned at latest if omitted)")
@click.option("--force", is_flag=True, help="Skip the existing-commitment check (DANGEROUS)")
def main(model_repo, revision, force):
    
    from huggingface_hub import repo_info
    from eval.model_checker import check_model_architecture

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO

    # ── Resolve revision (pin to specific SHA) ─────────────────────────
    if not revision:
        info = repo_info(model_repo, repo_type="model")
        revision = info.sha
        logger.info(f"Pinning to latest revision: {revision[:12]}...")
    else:
        logger.info(f"Using specified revision: {revision[:12]}...")

    # ── Pre-flight architecture check ──────────────────────────────────
    logger.info(f"Checking model: {model_repo}@{revision[:12]}...")
    check = check_model_architecture(model_repo, revision, max_params_b)
    if not check["pass"]:
        logger.error(f"Model check FAILED: {check['reason']}")
        logger.error("Your model does not meet the requirements. Fix and retry.")
        sys.exit(1)

    logger.info(f"✓ Model check passed: {check.get('params_b', 0):.2f}B params, "
                f"vocab_size={check.get('vocab_size', '?')}")

    
if __name__ == "__main__":
    main()
