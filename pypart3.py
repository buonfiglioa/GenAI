import argparse
import gc
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import queue
import re
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import torch
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_path":            "data/corrupted_dataset.json",
    "images_dir":              "data/images",
    "baseline_results_path":   "data/benchmark_results.json",
    "data_dir":                "data",
    "figures_dir":             "data/figures",
    "mitigation_results_path": "data/mitigation_results.json",
    # All models benchmarked in Part 2 — used when --model is not specified
    "models": [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "HuggingFaceTB/SmolVLM-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "google/gemma-4-E2B-it",
    ],
    "seed": 42,
}

# ── Prompts ───────────────────────────────────────────────────────────────────
def _corruption_hint(original_q: str, ctype: str) -> str:
    if ctype == "layout":
        return (
            f"The original question asked: \"{original_q}\"\n"
            "This version changes a spatial reference (e.g. top/bottom, left/right, first/last). "
            "The new position does not exist in the document layout."
        )
    elif ctype == "nlp_entity":
        return (
            f"The original question asked: \"{original_q}\"\n"
            "This version substitutes a key entity (name, number, date, or value) with a different one "
            "that is not present in the document."
        )
    elif ctype == "element":
        return (
            f"The original question asked: \"{original_q}\"\n"
            "This version swaps a document element reference (section, field, or label) "
            "to one not present in this document."
        )
    else:  # combined types: element+layout, nlp_entity+element, nlp_entity+layout
        return (
            f"The original question asked: \"{original_q}\"\n"
            "This version has been modified in multiple ways — "
            "one or more referenced entities or positions are not present in the document."
        )


def build_few_shot_prompts(
    ex_a_q: str, ex_u_q: str, ex_u_original_q: str, ex_u_ctype: str,
    ex_nlp_u_q: str,
) -> "tuple[str, str, str]":
    """Build few-shot prompt templates. Returns (few_shot, explained_few_shot, multi_few_shot)."""
    header = (
        "You are evaluating a document question-answering system.\n"
        "Look at the document image carefully. "
        "For each entity, value, or position mentioned in the question, "
        "verify it is actually visible in the document before deciding.\n\n"
    )
    suffix = (
        "Now evaluate:\n"
        "Can the following question be answered SOLELY from the content visible in this document?\n\n"
        "Question: {question}\n\n"
        "Reply with ONE word only: ANSWERABLE or UNANSWERABLE"
    )
    ex1 = (
        "[Example 1 — ANSWERABLE]\n"
        f"Question: \"{ex_a_q}\"\n"
        "After examining the document, the required information is present in the visible content.\n"
        "Answer: ANSWERABLE\n\n"
    )
    ex2 = (
        "[Example 2 — UNANSWERABLE]\n"
        f"Question: \"{ex_u_q}\"\n"
        "After examining the document, the specific entity, value, or position referenced "
        "in this question is not present anywhere in the visible content.\n"
        "Answer: UNANSWERABLE\n\n"
    )
    few_shot = header + ex1 + ex2 + suffix

    hint = _corruption_hint(ex_u_original_q, ex_u_ctype)
    ex2_explained = (
        "[Example 2 — UNANSWERABLE]\n"
        f"Question: \"{ex_u_q}\"\n"
        f"{hint}\n"
        "Answer: UNANSWERABLE\n\n"
    )
    explained_few_shot = header + ex1 + ex2_explained + suffix

    ex3 = (
        "[Example 3 — UNANSWERABLE]\n"
        f"Question: \"{ex_nlp_u_q}\"\n"
        "After examining the document, the entity or value referenced "
        "in this question is not present anywhere in the visible content.\n"
        "Answer: UNANSWERABLE\n\n"
    )
    multi_few_shot = header + ex1 + ex2 + ex3 + suffix

    return few_shot, explained_few_shot, multi_few_shot


# ── Reproducibility + GPU check ───────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default=None,
    help=(
        "HuggingFace model ID to apply mitigations to (must exist in benchmark_results.json). "
        "If omitted, all models in CONFIG['models'] are evaluated in sequence."
    ),
)
parser.add_argument(
    "--debug", action="store_true",
    help=(
        "Log raw model responses during inference and save them to "
        "data/debug_responses_{model}.json. Useful to diagnose systematic biases "
        "(e.g. Qwen2-VL-2B always predicting UNANSWERABLE)."
    ),
)
parser.add_argument(
    "--reset", nargs="*", metavar="MITIGATION",
    help=(
        "Delete cached results for the given mitigation name(s) so they are re-run. "
        "Applies to the models selected by --model (or all models if --model is omitted). "
        "Pass no names to reset ALL mitigations for those models. "
        "Example: --reset few_shot skeptical"
    ),
)
args = parser.parse_args()
target_models = [args.model] if args.model else CONFIG["models"]

set_seed(CONFIG["seed"])
Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["figures_dir"]).mkdir(parents=True, exist_ok=True)

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")

print(f"\nTarget models  : {target_models}")

# ── Load dataset (mirrors Part 2 exactly) ────────────────────────────────────
print(f"\nLoading dataset from {CONFIG['dataset_path']} …")
if not Path(CONFIG["dataset_path"]).exists():
    raise FileNotFoundError(
        f"{CONFIG['dataset_path']} not found — run pypart1.py first."
    )
with open(CONFIG["dataset_path"]) as f:
    verified = json.load(f)

images_dir = Path(CONFIG["images_dir"])
samples = []
skipped = 0
for item in verified:
    img_path = images_dir / f"{item['answer_page']}.jpg"
    if not img_path.exists():
        skipped += 1
        continue
    samples.append({
        "questionId":       item["questionId"],
        "doc_id":           item["doc_id"],
        "answer_page":      item["answer_page"],
        "image_path":       str(img_path),
        "question":         item["original_question"],
        "corruption_type":  item["corruption_type"],
        "label":            "ANSWERABLE",
        "label_int":        1,
    })
    samples.append({
        "questionId":       item["questionId"],
        "doc_id":           item["doc_id"],
        "answer_page":      item["answer_page"],
        "image_path":       str(img_path),
        "question":         item["corrupted_question"],
        "corruption_type":  item["corruption_type"],
        "label":            "UNANSWERABLE",
        "label_int":        0,
    })

print(f"  Verified samples   : {len(verified)}")
print(f"  Skipped (no image) : {skipped}")
print(f"  Benchmark rows     : {len(samples)}  ({len(samples)//2} pairs, perfectly balanced)")
print(f"  Corruption types   : {dict(Counter(s['corruption_type'] for s in samples if s['label_int']==0))}")

ctypes = sorted(set(s["corruption_type"] for s in samples if s["label"] == "UNANSWERABLE"))

# ── Build few-shot prompts from real dataset examples ─────────────────────────
# Examples drawn from the evaluation pool (fixed seed); contamination is negligible.
_rng_fs = random.Random(CONFIG["seed"])
_ex_a = _rng_fs.choice([s for s in samples if s["label"] == "ANSWERABLE"])
_ex_u = _rng_fs.choice([s for s in samples if s["label"] == "UNANSWERABLE"])
_ex_u_orig = next(s for s in samples if s["questionId"] == _ex_u["questionId"] and s["label"] == "ANSWERABLE")

# NLP-entity example for multi_few_shot (different type from _ex_u)
_nlp_pool = [s for s in samples if s["label"] == "UNANSWERABLE" and "nlp" in s["corruption_type"]
             and s["questionId"] != _ex_u["questionId"]]
_ex_nlp_u = _rng_fs.choice(_nlp_pool) if _nlp_pool else _ex_u

PROMPT_FEW_SHOT, PROMPT_EXPLAINED_FEW_SHOT, PROMPT_MULTI_FEW_SHOT = build_few_shot_prompts(
    _ex_a["question"], _ex_u["question"], _ex_u_orig["question"], _ex_u["corruption_type"],
    _ex_nlp_u["question"],
)

# Layout few-shot: pick an UNANSWERABLE example specifically from the layout corruption type
_layout_pool = [s for s in samples if s["label"] == "UNANSWERABLE" and s["corruption_type"] == "layout"]
_ex_layout_u = _rng_fs.choice(_layout_pool) if _layout_pool else _ex_u
PROMPT_LAYOUT_FEW_SHOT = (
    "You are evaluating a document question-answering system.\n"
    "Look at the document image carefully. "
    "For each entity, value, or position mentioned in the question, "
    "verify it is actually visible in the document before deciding.\n\n"
    "[Example 1 — ANSWERABLE]\n"
    f"Question: \"{_ex_a['question']}\"\n"
    "After examining the document, the required information is present in the visible content.\n"
    "Answer: ANSWERABLE\n\n"
    "[Example 2 — UNANSWERABLE]\n"
    f"Question: \"{_ex_layout_u['question']}\"\n"
    "After examining the document layout, the spatial position or structural element referenced "
    "(e.g. top, bottom, left, right, first, last, header, footer) "
    "is not present in this document.\n"
    "Answer: UNANSWERABLE\n\n"
    "Now evaluate:\n"
    "Can the following question be answered SOLELY from the content visible in this document?\n\n"
    "Question: {question}\n\n"
    "Reply with ONE word only: ANSWERABLE or UNANSWERABLE"
)

# Skeptical: explicit anti-ANSWERABLE-bias calibration
PROMPT_SKEPTICAL = (
    "You are evaluating a document question-answering system.\n"
    "Look at the document image carefully.\n\n"
    "IMPORTANT: Questions in this evaluation may have been deliberately modified to reference "
    "entities, values, or positions that are NOT present in the document. "
    "Do not assume a question is answerable just because it sounds plausible. "
    "If you cannot find the exact referenced element in the document, mark it UNANSWERABLE.\n\n"
    "Question: {question}\n\n"
    "Reply with ONE word only: ANSWERABLE or UNANSWERABLE"
)

print(f"\nFew-shot examples selected (seed={CONFIG['seed']}):")
print(f"  ANSWERABLE           : {_ex_a['question']!r}")
print(f"  UNANSWERABLE         : {_ex_u['question']!r}  [{_ex_u['corruption_type']}]")
print(f"  UNANSWERABLE (nlp)   : {_ex_nlp_u['question']!r}  [{_ex_nlp_u['corruption_type']}]")
print(f"  UNANSWERABLE (layout): {_ex_layout_u['question']!r}  [{_ex_layout_u['corruption_type']}]")

# ── Mitigation registry ───────────────────────────────────────────────────────
MITIGATIONS = {
    "few_shot":          {"prompt": PROMPT_FEW_SHOT,          "max_new_tokens": 16},
    "explained_few_shot": {"prompt": PROMPT_EXPLAINED_FEW_SHOT, "max_new_tokens": 16},
    "layout_few_shot":   {"prompt": PROMPT_LAYOUT_FEW_SHOT,   "max_new_tokens": 16},
    "multi_few_shot":    {"prompt": PROMPT_MULTI_FEW_SHOT,    "max_new_tokens": 16},
    "skeptical":         {"prompt": PROMPT_SKEPTICAL,          "max_new_tokens": 16},
}

# ── Load Part 2 baseline results (all models, once) ───────────────────────────
print(f"\nLoading Part 2 baseline from {CONFIG['baseline_results_path']} …")
if not Path(CONFIG["baseline_results_path"]).exists():
    raise FileNotFoundError(
        f"{CONFIG['baseline_results_path']} not found — run pypart2.py first."
    )
with open(CONFIG["baseline_results_path"]) as f:
    baseline_all = json.load(f)

for tm in target_models:
    if tm not in baseline_all:
        raise KeyError(
            f"'{tm}' not found in benchmark_results.json. "
            f"Available: {list(baseline_all.keys())}"
        )

# ── Load existing mitigation results ─────────────────────────────────────────
mit_results_path = Path(CONFIG["mitigation_results_path"])
mit_results: dict = {}
if mit_results_path.exists():
    with open(mit_results_path) as f:
        mit_results = json.load(f)

if args.reset is not None:
    to_reset = args.reset if args.reset else list(MITIGATIONS.keys())
    unknown  = [m for m in to_reset if m not in MITIGATIONS]
    if unknown:
        raise ValueError(f"Unknown mitigation(s): {unknown}. Valid: {list(MITIGATIONS.keys())}")
    for tm in target_models:
        dropped = [m for m in to_reset if m in mit_results.get(tm, {})]
        for m in dropped:
            del mit_results[tm][m]
        if dropped:
            print(f"  Reset {dropped} for {tm.split('/')[-1]}")
    tmp = mit_results_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(mit_results, f, indent=2)
    os.replace(tmp, mit_results_path)

# ── Inference helpers ─────────────────────────────────────────────────────────
_MAX_SIDE = 2048
BATCH_SIZE = 32

def _prepare_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_SIDE:
        scale = _MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _parse_verdict(response: str) -> str:
    """Return the last ANSWERABLE/UNANSWERABLE token; word-boundary match prevents
    ANSWERABLE matching inside UNANSWERABLE."""
    matches = list(re.finditer(r'\bUNANSWERABLE\b|\bANSWERABLE\b', response.upper()))
    if not matches:
        return "ANSWERABLE"  # default if model output is garbled
    return matches[-1].group()

def _apply_chat_template(processor, messages: list) -> str:
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    except (ValueError, AttributeError):
        return processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

def _build_inputs(processor, model_name: str, pil_images: list, prompt_texts: list, input_device):
    messages_batch = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pt}]}]
        for pt in prompt_texts
    ]
    texts = [_apply_chat_template(processor, m) for m in messages_batch]
    inputs = processor(text=texts, images=[[img] for img in pil_images], return_tensors="pt", padding=True).to(input_device)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(dtype=torch.bfloat16)
    return inputs

def predict_batch(processor, model, input_device,
                  pil_images: list, questions: list,
                  prompt_template: str, max_new_tokens: int,
                  model_name: str = "", return_raw: bool = False) -> list:
    pil_images   = [_prepare_image(img) for img in pil_images]
    prompt_texts = [prompt_template.format(question=q) for q in questions]

    inputs = _build_inputs(processor, model_name, pil_images, prompt_texts, input_device)

    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        #print(f"[dtype-debug] pixel_values shape={pv.shape} dtype={pv.dtype}", flush=True)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(pil_images)):
        response = processor.decode(out[i][input_len:], skip_special_tokens=True).strip()
        verdict  = _parse_verdict(response)
        results.append((verdict, response) if return_raw else verdict)
    return results

# ── Debug analysis helper ─────────────────────────────────────────────────────
def _print_debug_analysis(short: str, mit_name: str, records: list) -> None:
    """Print a concise analysis of raw responses to surface systematic biases."""
    total     = len(records)
    garbled   = sum(1 for r in records if not re.search(r'\bUNANSWERABLE\b|\bANSWERABLE\b',
                                                         r["raw_response"].upper()))
    lengths   = [len(r["raw_response"]) for r in records]
    avg_len   = sum(lengths) / total if total else 0

    # Separate by predicted verdict
    pred_answ   = [r for r in records if r["verdict"] == "ANSWERABLE"]
    pred_unansw = [r for r in records if r["verdict"] == "UNANSWERABLE"]

    def _is_immediate(raw: str) -> bool:
        """True if UNANSWERABLE appears immediately (reflexive bias signal)."""
        m = re.search(r'\bUNANSWERABLE\b|\bANSWERABLE\b', raw.upper())
        return m is not None and m.group() == "UNANSWERABLE" and m.start() < 30

    immediate = sum(1 for r in pred_unansw if _is_immediate(r["raw_response"]))

    print(f"\n  [DEBUG] {short} / {mit_name}")
    print(f"    Total samples         : {total}")
    print(f"    Predicted ANSWERABLE  : {len(pred_answ)}  ({100*len(pred_answ)/total:.1f}%)")
    print(f"    Predicted UNANSWERABLE: {len(pred_unansw)}  ({100*len(pred_unansw)/total:.1f}%)")
    print(f"    Garbled (no verdict)  : {garbled}")
    print(f"    Avg response length   : {avg_len:.0f} chars")
    print(f"    UNANSWERABLE immediate (bias signal): {immediate}/{len(pred_unansw)}"
          f"  ({100*immediate/len(pred_unansw):.1f}%)" if pred_unansw else "")

    if pred_answ:
        print(f"    Sample ANSWERABLE responses:")
        for r in pred_answ[:2]:
            snippet = r["raw_response"][:120].replace("\n", " ")
            print(f"      [{r['label']}] Q: {r['question']!r}")
            print(f"               Raw: {snippet!r}")

    if pred_unansw:
        print(f"    Sample UNANSWERABLE responses:")
        for r in pred_unansw[:2]:
            snippet = r["raw_response"][:120].replace("\n", " ")
            print(f"      [{r['label']}] Q: {r['question']!r}")
            print(f"               Raw: {snippet!r}")


# ── GPU-parallel worker ───────────────────────────────────────────────────────
# Models that cannot fit on a single GPU — must stay on device_map="auto".
_LARGE_MODELS = {"google/gemma-3-12b-it"}

def _run_mitigation_worker(
    model_name: str,
    samples: list,
    mit_name: str,
    cfg: dict,
    pil_cache: dict,
    debug_flag: bool,
    gpu_pool: "queue.Queue",
) -> "tuple[str, list, list]":
    """Acquire a free GPU, load the model, run one mitigation, unload, release the GPU."""
    gpu_id = gpu_pool.get()
    device = f"cuda:{gpu_id}"
    mdl = proc = None
    try:
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        proc.tokenizer.padding_side = "left"
        mdl = ModelClass.from_pretrained(
            model_name,
            device_map={"": gpu_id},
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        mdl.eval()
        input_dev = torch.device(device)

        per_sample: list = []
        raw_records: list = []
        for chunk in tqdm(list(_chunked(samples, BATCH_SIZE)), desc=f"GPU{gpu_id}/{mit_name}", leave=True):
            imgs    = [pil_cache[s["image_path"]] for s in chunk]
            results = predict_batch(
                proc, mdl, input_dev, imgs,
                [s["question"] for s in chunk],
                cfg["prompt"], cfg["max_new_tokens"],
                model_name=model_name, return_raw=debug_flag,
            )
            for s, result in zip(chunk, results):
                if debug_flag:
                    pred, raw_response = result
                    raw_records.append({
                        "questionId":      s["questionId"],
                        "corruption_type": s["corruption_type"],
                        "label":           s["label"],
                        "question":        s["question"],
                        "raw_response":    raw_response,
                        "verdict":         pred,
                    })
                else:
                    pred = result
                per_sample.append({
                    "questionId":      s["questionId"],
                    "corruption_type": s["corruption_type"],
                    "label":           s["label"],
                    "prediction":      pred,
                    "correct":         pred == s["label"],
                })
        return mit_name, per_sample, raw_records
    finally:
        del mdl, proc
        gc.collect()
        torch.cuda.empty_cache()
        gpu_pool.put(gpu_id)


# ── Per-model evaluation loop ─────────────────────────────────────────────────
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
    ModelClass = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForCausalLM
    ModelClass = AutoModelForCausalLM

fig_dir = Path(CONFIG["figures_dir"])
question_idx = {(s["questionId"], s["label"]): s["question"] for s in samples}

for target_model in target_models:
    short = target_model.split("/")[-1]

    print(f"\n{'#'*60}")
    print(f"# Model: {short}")
    print(f"{'#'*60}")

    baseline     = baseline_all[target_model]
    baseline_urec = baseline["unanswerable_recall"]

    print(f"  Baseline — Accuracy: {baseline['accuracy']:.3f}  "
          f"Macro F1: {baseline['macro_f1']:.3f}  "
          f"Unansw-Rec: {baseline_urec:.3f}")

    # ── Inference for remaining mitigations ───────────────────────────────────
    if target_model not in mit_results:
        mit_results[target_model] = {}

    remaining = [m for m in MITIGATIONS if m not in mit_results[target_model]]
    done      = [m for m in MITIGATIONS if m     in mit_results[target_model]]
    if done:
        print(f"  Resuming: {done} already done.")

    if remaining:
        pil_cache  = {s["image_path"]: _prepare_image(Image.open(s["image_path"])) for s in samples}
        debug_log: dict = {}

        n_gpus       = torch.cuda.device_count()
        run_parallel = n_gpus > 1 and target_model not in _LARGE_MODELS

        def _record_results(mit_name: str, per_sample: list, raw_records: list) -> None:
            """Compute metrics, update mit_results, persist to disk. Runs in main thread."""
            cfg = MITIGATIONS[mit_name]
            print(f"\n  {'='*56}")
            print(f"  Mitigation : {mit_name}  (max_new_tokens={cfg['max_new_tokens']})")
            print(f"  {'='*56}")

            if args.debug:
                debug_log[mit_name] = raw_records
                _print_debug_analysis(short, mit_name, raw_records)

            labels = [s["label_int"] for s in samples]
            preds  = [1 if p["prediction"] == "ANSWERABLE" else 0 for p in per_sample]

            acc = accuracy_score(labels, preds)
            prec,   rec,   f1,   _ = precision_recall_fscore_support(
                labels, preds, average="binary", pos_label=1, zero_division=0)
            prec_u, rec_u, f1_u, _ = precision_recall_fscore_support(
                labels, preds, average="binary", pos_label=0, zero_division=0)

            mit_results[target_model][mit_name] = {
                "accuracy":               round(acc,    4),
                "answerable_precision":   round(prec,   4),
                "answerable_recall":      round(rec,    4),
                "answerable_f1":          round(f1,     4),
                "unanswerable_precision": round(prec_u, 4),
                "unanswerable_recall":    round(rec_u,  4),
                "unanswerable_f1":        round(f1_u,   4),
                "macro_f1":               round((f1 + f1_u) / 2, 4),
                "per_sample":             per_sample,
            }
            print(f"  Accuracy : {acc:.3f}  Macro F1 : {mit_results[target_model][mit_name]['macro_f1']:.3f}  "
                  f"Unansw-F1 : {f1_u:.3f}  ΔUnansw-Rec : {rec_u - baseline_urec:+.3f}")
            tmp = mit_results_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(mit_results, f, indent=2)
            os.replace(tmp, mit_results_path)

        if run_parallel:
            print(f"\n  Parallel inference: {len(remaining)} mitigation(s) distributed across "
                  f"{n_gpus} GPUs (one {short} copy per GPU).")
            gpu_pool: queue.Queue = queue.Queue()
            for i in range(n_gpus):
                gpu_pool.put(i)

            with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
                future_to_mit = {
                    executor.submit(
                        _run_mitigation_worker,
                        target_model, samples, mit_name, MITIGATIONS[mit_name],
                        pil_cache, args.debug, gpu_pool,
                    ): mit_name
                    for mit_name in remaining
                }
                for future in as_completed(future_to_mit):
                    try:
                        mit_name, per_sample, raw_records = future.result()
                        _record_results(mit_name, per_sample, raw_records)
                    except Exception as exc:
                        failed_mit = future_to_mit[future]
                        print(f"\n  Worker for mitigation '{failed_mit}' failed: {exc}")

        else:
            # Sequential: load once with device_map="auto" (required for large models).
            print(f"\n  Loading {target_model} …")
            processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
            processor.tokenizer.padding_side = "left"
            model = ModelClass.from_pretrained(
                target_model,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model.eval()
            input_device = next(model.parameters()).device

            for mit_name in remaining:
                cfg = MITIGATIONS[mit_name]
                per_sample  = []
                raw_records = []
                for chunk in tqdm(list(_chunked(samples, BATCH_SIZE)), desc=f"{short}/{mit_name}"):
                    imgs    = [pil_cache[s["image_path"]] for s in chunk]
                    results = predict_batch(
                        processor, model, input_device, imgs,
                        [s["question"] for s in chunk],
                        cfg["prompt"], cfg["max_new_tokens"],
                        model_name=target_model, return_raw=args.debug,
                    )
                    for s, result in zip(chunk, results):
                        if args.debug:
                            pred, raw_response = result
                            raw_records.append({
                                "questionId":      s["questionId"],
                                "corruption_type": s["corruption_type"],
                                "label":           s["label"],
                                "question":        s["question"],
                                "raw_response":    raw_response,
                                "verdict":         pred,
                            })
                        else:
                            pred = result
                        per_sample.append({
                            "questionId":      s["questionId"],
                            "corruption_type": s["corruption_type"],
                            "label":           s["label"],
                            "prediction":      pred,
                            "correct":         pred == s["label"],
                        })
                _record_results(mit_name, per_sample, raw_records)

            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n  {short} unloaded.")

        if args.debug and debug_log:
            debug_path = Path(CONFIG["data_dir"]) / f"debug_responses_{short}.json"
            with open(debug_path, "w") as f:
                json.dump(debug_log, f, indent=2, ensure_ascii=False)
            print(f"\n  Debug responses saved → {debug_path}")
    else:
        print("  All mitigations already computed — skipping inference.")
        if args.debug:
            debug_path = Path(CONFIG["data_dir"]) / f"debug_responses_{short}.json"
            if debug_path.exists():
                print(f"  Debug responses already saved at {debug_path}")
            else:
                print("  Warning: no debug responses on disk — delete cached mitigation results "
                      "and rerun with --debug to collect raw responses.")

    # ── Summary table ─────────────────────────────────────────────────────────
    model_mit    = mit_results[target_model]
    all_variants = {"baseline": baseline, **model_mit}

    print(f"\n=== MITIGATION SUMMARY  ({short}) ===")
    hdr = f"{'Variant':<20} {'Acc':>6} {'MacroF1':>8} {'Unansw-F1':>10} {'Unansw-Rec':>11}  {'ΔUnansw-Rec':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_variants.items():
        delta = res["unanswerable_recall"] - baseline_urec if name != "baseline" else 0.0
        sign  = "+" if delta >= 0 else ""
        print(
            f"{name:<20} {res['accuracy']:>6.3f} {res['macro_f1']:>8.3f} "
            f"{res['unanswerable_f1']:>10.3f} {res['unanswerable_recall']:>11.3f}  "
            f"{sign}{delta:>11.3f}"
        )

    # ── Per corruption-type unanswerable recall ───────────────────────────────
    def _type_recall(per_sample_list, ct):
        items = [p for p in per_sample_list if p["label"] == "UNANSWERABLE" and p["corruption_type"] == ct]
        return sum(p["correct"] for p in items) / len(items) if items else float("nan")

    col_w = max(len(ct) for ct in ctypes) + 2
    print(f"\n=== Unanswerable recall by corruption type ({short}) ===")
    print(f"{'Variant':<20}", end="")
    for ct in ctypes:
        print(f"  {ct:>{col_w}}", end="")
    print()
    print("-" * (20 + (col_w + 2) * len(ctypes)))
    for name, res in all_variants.items():
        print(f"{name:<20}", end="")
        for ct in ctypes:
            r = _type_recall(res["per_sample"], ct)
            print(f"  {r:>{col_w}.3f}", end="")
        print()

    # ── Per-model plots ───────────────────────────────────────────────────────
    variant_names = list(all_variants.keys())

    # Plot 1: Answerable vs Unanswerable recall by variant
    x     = np.arange(len(variant_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [all_variants[v]["answerable_recall"]   for v in variant_names],
           width, label="Answerable recall",   color="#4C72B0")
    ax.bar(x + width / 2, [all_variants[v]["unanswerable_recall"] for v in variant_names],
           width, label="Unanswerable recall", color="#DD8452")
    ax.axhline(baseline_urec, color="#DD8452", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(variant_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Recall")
    ax.set_title(f"Recall by class and mitigation — {short}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"mitigation_recall_{short}.jpg", dpi=150)
    plt.show()
    print(f"\nSaved → {fig_dir}/mitigation_recall_{short}.jpg")

    # Plot 2: Key metrics across variants (line chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in ["macro_f1", "unanswerable_f1", "accuracy"]:
        ax.plot(variant_names, [all_variants[v][m] for v in variant_names],
                marker="o", label=m.replace("_", " "))
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"Metrics across mitigations — {short}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(fig_dir / f"mitigation_metrics_{short}.jpg", dpi=150)
    plt.show()
    print(f"Saved → {fig_dir}/mitigation_metrics_{short}.jpg")

    # Plot 3: Per corruption-type unanswerable recall heatmap
    recall_matrix = [
        [_type_recall(all_variants[v]["per_sample"], ct) for ct in ctypes]
        for v in variant_names
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(ctypes) * 2.5), len(variant_names) + 1))
    im = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ctypes))); ax.set_xticklabels(ctypes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(variant_names))); ax.set_yticklabels(variant_names, fontsize=9)
    for i in range(len(variant_names)):
        for j in range(len(ctypes)):
            v = recall_matrix[i][j]
            ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Unanswerable recall")
    ax.set_title(f"Unanswerable recall by mitigation and corruption type — {short}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"mitigation_by_type_{short}.jpg", dpi=150)
    plt.show()
    print(f"Saved → {fig_dir}/mitigation_by_type_{short}.jpg")

    # ── Qualitative error analysis ─────────────────────────────────────────────
    print(f"\n=== QUALITATIVE ERROR ANALYSIS  ({short}) ===")
    baseline_idx = {(p["questionId"], p["label"]): p for p in baseline["per_sample"]}

    for mit_name in MITIGATIONS:
        if mit_name not in model_mit:
            continue
        mit_idx = {(p["questionId"], p["label"]): p for p in model_mit[mit_name]["per_sample"]}

        fixes       = []
        still_wrong = []
        regressions = []
        for key, bp in baseline_idx.items():
            mp = mit_idx.get(key)
            if not mp:
                continue
            if bp["label"] == "UNANSWERABLE" and not bp["correct"] and mp["correct"]:
                fixes.append((bp["corruption_type"], question_idx.get(key, "?")))
            if mp["label"] == "UNANSWERABLE" and not mp["correct"]:
                still_wrong.append((mp["corruption_type"], question_idx.get(key, "?")))
            if bp["label"] == "ANSWERABLE" and bp["correct"] and not mp["correct"]:
                regressions.append((bp["corruption_type"], question_idx.get(key, "?")))

        print(f"\n  ── {mit_name} ──")
        print(f"     Fixes (baseline→correct)  : {len(fixes)}")
        print(f"     Still wrong (UNANSW missed): {len(still_wrong)}")
        print(f"     Regressions (ANSW broken)  : {len(regressions)}")
        if fixes:
            print("     Sample fixes:")
            for ct, q in fixes[:3]:
                print(f"       [{ct}] {q!r}")
        if still_wrong:
            print("     Still failing:")
            for ct, q in still_wrong[:3]:
                print(f"       [{ct}] {q!r}")
        if regressions:
            print("     Regressions introduced:")
            for ct, q in regressions[:3]:
                print(f"       [{ct}] {q!r}")

    best_mit = max(model_mit, key=lambda m: model_mit[m]["unanswerable_recall"])
    best_res = model_mit[best_mit]
    print(f"\n  Best mitigation : {best_mit}  "
          f"Unansw-Rec {best_res['unanswerable_recall']:.3f}  "
          f"(Δ {best_res['unanswerable_recall'] - baseline_urec:+.3f})")

# ── Cross-model delta chart (only when evaluating multiple models) ────────────
if len(target_models) > 1:
    mit_names = list(MITIGATIONS.keys())
    x         = np.arange(len(mit_names))
    width = 0.8 / len(target_models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, tm in enumerate(target_models):
        base_urec = baseline_all[tm]["unanswerable_recall"]
        deltas = [
            mit_results[tm][m]["unanswerable_recall"] - base_urec
            if m in mit_results.get(tm, {}) else float("nan")
            for m in mit_names
        ]
        offset = (i - len(target_models) / 2 + 0.5) * width
        ax.bar(x + offset, deltas, width, label=tm.split("/")[-1])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(mit_names, rotation=15, ha="right")
    ax.set_ylabel("Δ Unanswerable Recall vs Baseline")
    ax.set_title("Mitigation improvement by model (cross-model comparison)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "mitigation_delta_crossmodel.jpg", dpi=150)
    plt.show()
    print(f"\nSaved → {fig_dir}/mitigation_delta_crossmodel.jpg")

print(f"\nPart 3 complete.")
print(f"Results  → {CONFIG['mitigation_results_path']}")
print(f"Figures  → {fig_dir}/")
