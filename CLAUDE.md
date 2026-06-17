# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a PoliTO Master's course exercise on Document Understanding with Vision Language Models. The pipeline is fully consolidated into a single script (`pipeline.py`) with three parts:

1. **Part 1** — Build a dataset of unanswerable DocVQA questions by corrupting originals with four strategies: NLP entity swap, document-element swap, spatial/layout swap, and numerical swap (plus a combined mode that chains two strategies). Qwen2.5-VL-7B-Instruct acts as judge to verify each corrupted question is genuinely unanswerable. Output: `data/corrupted_dataset.json` + `data/images/`.

2. **Part 2** — Benchmark four Vision LLMs (LLaVA-1.6-Mistral-7B, Gemma-3-4b-it, Gemma-3-12b-it, Qwen2.5-VL-3B-Instruct) on the resulting dataset as a binary classification task (ANSWERABLE vs UNANSWERABLE). Each model is loaded, evaluated, then **unloaded** before the next to manage GPU memory. Output: `data/benchmark_results.json` + `data/figures/`.

3. **Part 3** — Mitigation: re-evaluate the same four models with five prompting strategies (Chain-of-Thought, Few-Shot+CoT, Knowledge Injection, Role-Based, Self-Refine) and compare against the Part 2 baseline. Output: `data/mitigation_results.json` + additional figures.

## Running the pipeline

```bash
# Install dependencies
uv sync

# Run all three parts in sequence
uv run python pipeline.py

# Run a specific part
uv run python pipeline.py --part 1          # corruption + judge
uv run python pipeline.py --part 2          # benchmark
uv run python pipeline.py --part 3          # mitigation

# Multi-page windowing (concatenate N pages per inference)
uv run python pipeline.py --part 1 --window-size 3

# Select a subset of benchmark models by alias
uv run python pipeline.py --part 2 --models llava,gemma4b

# Inspect saved results (no GPU needed)
uv run python inspect_outputs.py --distributions
uv run python inspect_outputs.py --distributions --file data/mitigation_results.json

# Spot-check raw model outputs (needs GPU)
uv run python inspect_outputs.py --model llava --strategy self_refine --n 5

# Test parse_label on any string
uv run python inspect_outputs.py --parse "VERDICT: not sure"
```

Model aliases for `--models` / `inspect_outputs.py`: `llava`, `gemma4b`, `gemma12b`, `qwen3b`.

## Environment

This project uses `uv` for dependency management. CUDA-capable GPUs are required. `pipeline.py` uses `device_map="auto"` to spread models across all available GPUs. Typical memory requirements:
- Qwen2.5-VL-7B-Instruct (judge): heaviest, ~4 GPUs
- LLaVA-1.6-Mistral-7B / Gemma-3-12b-it / Phi-4-multimodal: heavy
- Gemma-3-4b-it / Qwen2.5-VL-3B-Instruct: lighter

## Data and model flow

```
HuggingFace (VLR-CVC/DocVQA-2026)
    ↓ Part 1
data/corrupted_dataset.json     ← verified-unanswerable samples
data/images/*.jpg               ← document page images (gitignored)
    ↓ Part 2
data/benchmark_results.json     ← per-model predictions
data/figures/overall_metrics.png
data/figures/per_type_recall.png
data/figures/confusion_matrices.png
    ↓ Part 3
data/mitigation_results.json    ← baseline + mitigation predictions
data/figures/mitigation_accuracy.png
data/figures/mitigation_f1.png
data/figures/mitigation_recall.png
data/figures/mitigation_per_type_recall.png
```

`data/` is gitignored; regenerate by running the parts in order.

## Key design decisions

- **Judge model must differ from benchmarked models** to avoid circular evaluation. `Qwen2.5-VL-7B-Instruct` is the judge; it is not among the four benchmarked models.
- Corruption generates all viable types per question (not first-match fallback). A `combined` type chains exactly two corruption strategies on the same question, controlled by `combined_ratio` in `CONFIG`.
- `Image.MAX_IMAGE_PIXELS = None` is set globally to handle DocVQA's high-resolution scans without PIL decompression-bomb errors.
- Each model is explicitly unloaded (`del model; gc.collect(); torch.cuda.empty_cache()`) between runs; never load two large VLMs simultaneously.
- `infer()` in `pipeline.py` uses `apply_chat_template` when available, falling back to `processor.tokenizer.apply_chat_template`, to handle the different APIs across model families.
- Part 3 checkpoints `mitigation_results.json` after each model so a crash does not lose completed work.
- `inspect_outputs.py` imports `_load_vlm`, `_unload_vlm`, and `infer` directly from `pipeline.py` for spot-checks; it does not duplicate that logic.
