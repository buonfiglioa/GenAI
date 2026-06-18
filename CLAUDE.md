# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a PoliTO Master's course exercise on Document Understanding with Vision Language Models. The core task is:

1. **Part 1** (`pypart1.py`): Build a dataset of unanswerable DocVQA questions by corrupting originals (four strategies: NLP entity swap, document-element swap, spatial/layout swap, and combined), then use Qwen2.5-VL-7B-Instruct as a judge to verify each corrupted question is genuinely unanswerable. Output: `data/corrupted_dataset.json` + `data/images/`.

2. **Part 2** (`pypart2.py`): Benchmark multiple Vision LLMs (Gemma-3-4b-it, Gemma-3-12b-it, SmolVLM-Instruct, Qwen2.5-VL-3B-Instruct, InternVL3-4B) on the resulting dataset as a binary classification task (ANSWERABLE vs UNANSWERABLE). Each model is loaded, evaluated, then **unloaded** before the next to manage GPU memory. Output: `data/benchmark_results.json` + `data/figures/`.

3. **Part 3** (`pypart3.py`): Prompt-engineering mitigations applied to all benchmarked models to improve UNANSWERABLE detection. Five strategies: basic two-shot (`few_shot`), two-shot with corruption explanation (`explained_few_shot`), layout-targeted two-shot (`layout_few_shot`), two-shot with two distinct UNANSWERABLE examples of different corruption types (`multi_few_shot`), and explicit ANSWERABLE-bias warning (`skeptical`). Includes qualitative error analysis and a cross-model delta comparison chart. Output: `data/mitigation_results.json` + `data/figures/mitigation_*.jpg`.

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
uv run python pypart3.py                                      # all models
uv run python pypart3.py --model google/gemma-3-12b-it        # single model
uv run python pypart3.py --model HuggingFaceTB/SmolVLM-Instruct --debug  # debug bias
```

`pypart1.py` also accepts `--corruption-only` to stop before loading the judge model (useful for inspecting corrupted candidates quickly).

`pypart3.py` resumes from `data/mitigation_results.json` if it already exists (per-model, per-mitigation). `--debug` logs raw model responses to `data/debug_responses_{model}.json` and prints a bias analysis per mitigation (fraction of immediate UNANSWERABLE responses, response lengths, sample outputs).

The project requires CUDA-capable GPUs. Scripts use `device_map="auto"` to spread models across all available GPUs. Typical memory requirements:
- Qwen2.5-VL-7B-Instruct (judge): ~4 GPUs with the observed device map
- Gemma-3-4b-it, SmolVLM-Instruct, Qwen2.5-VL-3B-Instruct: lighter; Gemma-3-12b-it, InternVL3-4B: heavier

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
    ↓ pypart3.py
data/mitigation_results.json  ← per-model, per-mitigation predictions
data/debug_responses_*.json   ← raw model responses (only with --debug flag)
data/figures/                 ← mitigation_recall_{model}.jpg, mitigation_metrics_{model}.jpg,
                                 mitigation_by_type_{model}.jpg, mitigation_delta_crossmodel.jpg
    ↓ plot_results.py
data/figures/final/           ← consolidated final plots across all models
```

`data/` is gitignored; regenerate by running the scripts in order.

## Key design decisions

- **Judge model must differ from benchmarked models** to avoid circular evaluation. The judge (`Qwen2.5-VL-7B-Instruct`) is not among the five benchmarked models.
- Corruption tries the preferred type first and falls back to the other types if no keyword is found in the question (`apply_corruption` in Part 1). The `combined` type chains two single-type corruptions on the same question.
- `Image.MAX_IMAGE_PIXELS = None` is set globally in Part 1 to handle DocVQA's high-resolution scans without PIL decompression-bomb errors.
- Each benchmarked model is explicitly unloaded (`gc.collect()` + `torch.cuda.empty_cache()`) between runs; do not load two large VLMs simultaneously.
- Part 2 resumes from `data/benchmark_results.json` if it already exists, so a crash mid-run doesn't lose finished models.
- Part 3 few-shot examples are drawn from the evaluation set (same examples every run, fixed by seed). Contamination is negligible (≤3 of 500+) and noted as a limitation.
- Part 3 `layout_few_shot` specifically uses an UNANSWERABLE example with `corruption_type == "layout"` to target the layout corruption weakness (worst-performing type in Part 2).
- Part 3 `multi_few_shot` uses two UNANSWERABLE examples of different corruption types (one generic, one NLP-entity) so the model sees the diversity of unanswerability.
- Part 3 `--debug` mode detects immediate UNANSWERABLE responses (verdict within first 30 chars) as a signal of reflexive bias vs. genuine reasoning.
- `plot_results.py` is a standalone script that regenerates all final figures from the JSON results without loading any model. Outputs to `data/figures/final/`; automatically skips models not yet benchmarked.
