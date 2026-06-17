# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a PoliTO Master's course exercise on Document Understanding with Vision Language Models. The core task is:

1. **Part 1** (`pypart1.py`): Build a dataset of unanswerable DocVQA questions by corrupting originals (three strategies: NLP entity swap, document-element swap, spatial/layout swap), then use Qwen2.5-VL-7B-Instruct as a judge to verify each corrupted question is genuinely unanswerable. Output: `data/corrupted_dataset.json` + `data/images/`.

2. **Part 2** (`pypart2.py`): Benchmark multiple Vision LLMs (Gemma-3-4b-it, Gemma-3-12b-it, Phi-4-multimodal) on the resulting dataset as a binary classification task (ANSWERABLE vs UNANSWERABLE). Each model is loaded, evaluated, then **unloaded** before the next to manage GPU memory. Output: `data/benchmark_results.json` + `data/figures/`.

3. **Part 3**: Mitigation strategies — not yet written.

## Environment

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run scripts in order
uv run python pypart1.py
uv run python pypart2.py
```

The project requires CUDA-capable GPUs. Scripts use `device_map="auto"` to spread models across all available GPUs. Typical memory requirements:
- Qwen2.5-VL-7B-Instruct (judge): ~4 GPUs with the observed device map
- Gemma-3-4b-it: lighter; Gemma-3-12b-it / Phi-4-multimodal: heavier

## Data and model flow

```
HuggingFace (VLR-CVC/DocVQA-2026)
    ↓ Part 1
data/corrupted_dataset.json   ← 116 verified-unanswerable samples
data/images/*.jpg             ← document page images (gitignored)
    ↓ Part 2
data/benchmark_results.json   ← per-model predictions
data/figures/                 ← PNG plots (overall_metrics, per_type_recall, confusion_matrices)
```

`data/` is gitignored; regenerate by running the scripts in order.

## Key design decisions

- **Judge model must differ from benchmarked models** to avoid circular evaluation. The judge (`Qwen2.5-VL-7B-Instruct`) is not among the three benchmarked models.
- Corruption tries the preferred type first and falls back to the other two if no keyword is found in the question (`apply_corruption` in Part 1).
- `Image.MAX_IMAGE_PIXELS = None` is set globally in Part 1 to handle DocVQA's high-resolution scans without PIL decompression-bomb errors.
- Each benchmarked model is explicitly unloaded (`gc.collect()` + `torch.cuda.empty_cache()`) between runs; do not load two large VLMs simultaneously.
