# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a PoliTO Master's course exercise on Document Understanding with Vision Language Models. The core task is:

1. **Part 1** (`pypart1.py`): Build a dataset of unanswerable DocVQA questions by corrupting originals (four strategies: NLP entity swap, document-element swap, spatial/layout swap, and combined), then use Qwen2.5-VL-7B-Instruct as a judge to verify each corrupted question is genuinely unanswerable. Output: `data/corrupted_dataset.json` + `data/images/`.

2. **Part 2** (`pypart2.py`): Benchmark multiple Vision LLMs (Gemma-3-4b-it, Gemma-3-12b-it, Qwen2-VL-2B-Instruct) on the resulting dataset as a binary classification task (ANSWERABLE vs UNANSWERABLE). Each model is loaded, evaluated, then **unloaded** before the next to manage GPU memory. Output: `data/benchmark_results.json` + `data/figures/`.

3. **Part 3**: Mitigation strategies — not yet written.

## Environment

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Download data (run once before pypart1.py)
uv run python download_data.py

# Run scripts in order
uv run python pypart1.py
uv run python pypart2.py
```

`pypart1.py` also accepts `--corruption-only` to stop before loading the judge model (useful for inspecting corrupted candidates quickly).

The project requires CUDA-capable GPUs. Scripts use `device_map="auto"` to spread models across all available GPUs. Typical memory requirements:
- Qwen2.5-VL-7B-Instruct (judge): ~4 GPUs with the observed device map
- Gemma-3-4b-it: lighter; Gemma-3-12b-it / Qwen2-VL-2B-Instruct: heavier

## Data and model flow

```
HuggingFace (lmms-lab/DocVQA)
    ↓ download_data.py
data/docvqa_val.json          ← QA annotations
data/images/*.jpg             ← document page images (gitignored)
    ↓ pypart1.py
data/corrupted_candidates.json  ← raw corruptions (cached)
data/corrupted_dataset.json     ← judge-verified unanswerable samples
    ↓ pypart2.py
data/benchmark_results.json   ← per-model predictions
data/figures/                 ← JPG plots (overall_metrics, per_type_recall, confusion_matrices)
```

`data/` is gitignored; regenerate by running the scripts in order.

## Key design decisions

- **Judge model must differ from benchmarked models** to avoid circular evaluation. The judge (`Qwen2.5-VL-7B-Instruct`) is not among the three benchmarked models.
- Corruption tries the preferred type first and falls back to the other types if no keyword is found in the question (`apply_corruption` in Part 1). The `combined` type chains two single-type corruptions on the same question.
- `Image.MAX_IMAGE_PIXELS = None` is set globally in Part 1 to handle DocVQA's high-resolution scans without PIL decompression-bomb errors.
- Each benchmarked model is explicitly unloaded (`gc.collect()` + `torch.cuda.empty_cache()`) between runs; do not load two large VLMs simultaneously.
- Part 2 resumes from `data/benchmark_results.json` if it already exists, so a crash mid-run doesn't lose finished models.
