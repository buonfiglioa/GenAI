# GenAI — Document Understanding with Vision Language Models

PoliTO Master's course exercise. Builds and benchmarks a dataset of unanswerable DocVQA questions using Vision LLMs.

## Structure

| File | Purpose |
|---|---|
| `pypart1.py` | Corrupt DUDE questions → judge with Qwen2.5-VL → `data/corrupted_dataset.json` |
| `pypart2.py` | Benchmark Gemma-3-4b, Gemma-3-12b, Phi-4-multimodal → `data/benchmark_results.json` + plots |
| `pypart3.py` | _(not yet written)_ Mitigation strategies |

## Usage

```bash
uv sync
uv run python pypart1.py   # requires data/val.json + data/images/
uv run python pypart2.py   # requires data/corrupted_dataset.json (output of pypart1)
```

## Data flow

```
data/val.json + data/images/
    ↓ pypart1.py
data/corrupted_candidates.json   ← raw corruptions (cached)
data/corrupted_dataset.json      ← judge-verified unanswerable samples
    ↓ pypart2.py
data/benchmark_results.json
data/figures/{overall_metrics,per_type_recall,confusion_matrices}.jpg
```

`data/` is gitignored. GPU required; models are loaded one at a time via `device_map="auto"`.
