1. How It Works
distill the teacher into a smaller model (≤5.25B total params), upload to HuggingFace.

2.How to evaluate.
evaluate by computing full-distribution KL-divergence on GPU. Lower KL = better distillation = higher rewards.

1) King-of-the-Hill Evaluation

The evaluation system uses a **king-of-the-hill** architecture for efficient, high-confidence scoring:

1. **Pre-checks (no GPU)**
   - Architecture compliance (≤5.25B params, vocab_size=248,320, no quantization)

2. **King identification** — The model with the lowest KL score from state is the "king"

3. **Head-to-head GPU eval** — **same 40 FineWeb prompts**. Both models see identical teacher continuations, making the comparison fair. The king is only put on GPU when there's a challenger — no wasted compute on idle re-evaluation.

5. **Epsilon threshold (1%)** — A model must achieve KL divergence **more than 1% lower** than the king's to dethrone it. For example, if the king has KL=0.097, a challenger needs KL < 0.096 (= 0.097 × 0.99). This prevents noisy near-ties from flipping the winner every epoch and rewards meaningful improvements.

### Disqualification

- **INVALID** — Fails architecture checks (too large, wrong tokenizer, quantized, etc.)

### Anti-Gaming

- **MoE-aware param counting**: Total params from safetensors metadata (not config estimates)

### Model Requirements

The model must:
- Use **same tokenizer** as Qwen3.5-35B-A3B (vocab_size=248,320)
- Have ≤ **5.25B total parameters** (15% of teacher's 35B)
- Be in **safetensors** format (bf16/fp16)
- Be loadable via `AutoModelForCausalLM.from_pretrained()`
- **No quantized models** (GPTQ/AWQ/GGUF rejected)

### KL Ranges (baseline, no distillation training)

| Model | Params | KL (nats) | Notes |
|-------|--------|-----------|-------|
| Qwen3.5-4B | 4.66B | ~0.10–0.15 | Strong baseline |
| Qwen3.5-2B | 2.27B | ~0.12–0.16 | Competitive |
| Qwen3.5-0.8B | 0.87B | ~0.17–0.21 | Moderate |

These are *untrained baselines* — purpose-built distillations should do significantly better. Models with KL > 2.0 are disqualified.

## Evaluation system

### Requirements

- **GPU**: 1x with ≥80GB VRAM (A100 80GB, H100, or similar)

### What Evaluator Does

1. Loads the teacher model (Qwen3.5-35B-A3B) — ~70GB VRAM
2. Loads 500 prompts from FineWeb (cached after first run)
3. Polls for new challengers every epoch (~10 min)
4. Head-to-head KL evaluation: king vs challengers on identical prompts
