# GenAI — Document Understanding with Vision Language Models

PoliTO Master's course exercise. Builds and benchmarks a dataset of unanswerable DocVQA questions using Vision LLMs.

## Structure

| File | Purpose |
|---|---|
| `download_data.py` | Download QA annotations + images from HuggingFace → `data/` |
| `pypart1.py` | Corrupt DocVQA questions → judge with Qwen2.5-VL-7B → `data/corrupted_dataset.json` |
| `pypart2.py` | Benchmark Gemma-3-4b, Gemma-3-12b, SmolVLM, Qwen2.5-VL-3B, InternVL3-4B → `data/benchmark_results.json` + plots |
| `pypart3.py` | Prompt-engineering mitigations on all benchmarked models → `data/mitigation_results.json` + plots |
| `plot_results.py` | Standalone: regenerate final figures from JSON results, no model loading → `data/figures/final/` |

## Usage

```bash
uv sync
uv run python download_data.py        # one-time data download
uv run python pypart1.py              # requires data/docvqa_val.json + data/images/
uv run python pypart2.py              # requires data/corrupted_dataset.json (output of pypart1)
uv run python pypart3.py              # all models
uv run python pypart3.py --model google/gemma-3-12b-it   # single model
uv run python pypart3.py --debug      # log raw responses for bias analysis
uv run python plot_results.py         # regenerate final figures (no GPU needed)
```

`pypart1.py --corruption-only` stops before loading the judge model (useful for inspecting candidates).

`pypart3.py` resumes from `data/mitigation_results.json` if it already exists (per-model, per-mitigation).

## Mitigations (Part 3)

| Key | Strategy |
|---|---|
| `few_shot` | Two-shot with one ANSWERABLE + one UNANSWERABLE example; explicit scan instruction |
| `explained_few_shot` | Same but the UNANSWERABLE example includes a hint explaining the corruption type |
| `layout_few_shot` | Two-shot where the UNANSWERABLE example is a `layout` corruption, targeting the hardest type |
| `multi_few_shot` | Three-shot with one ANSWERABLE + two UNANSWERABLE examples of different corruption types |
| `skeptical` | Explicit warning against ANSWERABLE bias — no examples |

## Data flow

```
HuggingFace (lmms-lab/DocVQA)
    ↓ download_data.py
data/docvqa_val.json + data/images/
    ↓ pypart1.py
data/corrupted_candidates.json   ← raw corruptions (cached)
data/corrupted_dataset.json      ← judge-verified unanswerable samples
    ↓ pypart2.py
data/benchmark_results.json
data/figures/{overall_metrics,per_type_recall,confusion_matrices}.jpg
    ↓ pypart3.py
data/mitigation_results.json
data/figures/mitigation_{recall,metrics,by_type}_{model}.jpg
data/figures/mitigation_delta_crossmodel.jpg
    ↓ plot_results.py
data/figures/final/              ← consolidated final plots (all models, all mitigations)
```

`data/` is gitignored. GPU required for pypart1–3; `plot_results.py` needs no GPU. Models are loaded one at a time via `device_map="auto"`.
