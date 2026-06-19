import argparse
import gc
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)
from tqdm import tqdm
import torch
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # ---- input (produced by pypart1.py) ----
    "dataset_path": "data/corrupted_dataset.json",
    "images_dir":   "data/images",
    # ---- models to benchmark (loaded/unloaded one at a time) ----
    # Must NOT include the judge model used in Part 1 (Qwen2.5-VL-7B-Instruct)
    "models": [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "HuggingFaceTB/SmolVLM-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "OpenGVLab/InternVL3-4B",
    ],
    # ---- output ----
    "data_dir":    "data",
    "results_path": "data/benchmark_results.json",
    "figures_dir":  "data/figures",
    # ---- misc ----
    "seed": 42,
}

# ── Reproducibility + GPU check ───────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true",
                    help="Delete cached results and rerun all models from scratch.")
args = parser.parse_args()

set_seed(CONFIG["seed"])
Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["figures_dir"]).mkdir(parents=True, exist_ok=True)

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")

# ── Build benchmark dataset ───────────────────────────────────────────────────
print(f"\nLoading verified dataset from {CONFIG['dataset_path']} …")
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
    # ANSWERABLE row — original question
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
    # UNANSWERABLE row — corrupted question, same image
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

# ── Inference prompt ──────────────────────────────────────────────────────────
BENCHMARK_PROMPT = (
    "You are evaluating a document question-answering system.\n"
    "Look at the document image carefully.\n"
    "Can the following question be answered SOLELY from the content visible in this document?\n\n"
    "Question: {question}\n\n"
    "Reply with ONE word only: ANSWERABLE or UNANSWERABLE"
)

_MAX_SIDE = 2048
BATCH_SIZE = 8

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

def _apply_chat_template(processor, messages: list) -> str:
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    except (ValueError, AttributeError):
        return processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

def _build_inputs(processor, model_name: str, pil_images: list, prompt_texts: list, input_device):
    if "InternVL" in model_name:
        # InternVL3 expects <image>\n prefix in text rather than content-list format
        messages_batch = [
            [{"role": "user", "content": f"<image>\n{pt}"}]
            for pt in prompt_texts
        ]
        texts = [
            processor.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages_batch
        ]
        return processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(input_device)
    messages_batch = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pt}]}]
        for pt in prompt_texts
    ]
    texts = [_apply_chat_template(processor, m) for m in messages_batch]
    return processor(text=texts, images=[[img] for img in pil_images], return_tensors="pt", padding=True).to(input_device)

def predict_batch(processor, model, input_device,
                  pil_images: list, questions: list, model_name: str = "") -> list:
    pil_images   = [_prepare_image(img) for img in pil_images]
    prompt_texts = [BENCHMARK_PROMPT.format(question=q) for q in questions]

    inputs = _build_inputs(processor, model_name, pil_images, prompt_texts, input_device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(pil_images)):
        resp = processor.decode(out[i][input_len:], skip_special_tokens=True).strip().upper()
        results.append("UNANSWERABLE" if "UNANSWERABLE" in resp else "ANSWERABLE")
    return results

# ── Per-model evaluation loop ─────────────────────────────────────────────────
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
    ModelClass = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForCausalLM
    ModelClass = AutoModelForCausalLM

all_results = {}   # model_name → {predictions, labels, per_sample}

results_path = Path(CONFIG["results_path"])
if args.force and results_path.exists():
    results_path.unlink()
    print(f"Deleted cache: {results_path}")

if results_path.exists():
    with open(results_path) as f:
        all_results = json.load(f)
    print(f"\nResuming: {list(all_results.keys())} already done.")

for model_name in CONFIG["models"]:
    if model_name in all_results:
        print(f"\nSkipping {model_name} (already in results).")
        continue

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = ModelClass.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    input_device = next(model.parameters()).device


    per_sample = []
    for chunk in tqdm(list(_chunked(samples, BATCH_SIZE)), desc=f"Evaluating {model_name.split('/')[-1]}"):
        imgs  = [Image.open(s["image_path"]) for s in chunk]
        preds = predict_batch(processor, model, input_device, imgs,
                              [s["question"] for s in chunk], model_name=model_name)
        for s, pred in zip(chunk, preds):
            per_sample.append({
                "questionId":      s["questionId"],
                "corruption_type": s["corruption_type"],
                "label":           s["label"],
                "prediction":      pred,
                "correct":         pred == s["label"],
            })

    labels  = [s["label_int"] for s in samples]
    preds   = [1 if p["prediction"] == "ANSWERABLE" else 0 for p in per_sample]

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    prec_u, rec_u, f1_u, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=0, zero_division=0
    )

    all_results[model_name] = {
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

    print(f"  Accuracy         : {acc:.3f}")
    print(f"  Macro F1         : {all_results[model_name]['macro_f1']:.3f}")
    print(f"  Unanswerable F1  : {f1_u:.3f}  ← key metric")

    # Save after every model so a crash doesn't lose prior work
    with open(CONFIG["results_path"], "w") as f:
        json.dump(all_results, f, indent=2)

    # Unload before loading the next model
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {model_name} unloaded.")

# ── Metrics summary ───────────────────────────────────────────────────────────
print("\n\n=== BENCHMARK SUMMARY ===")
print(f"{'Model':<40} {'Acc':>6} {'MacroF1':>8} {'Unansw-F1':>10} {'Unansw-Rec':>11}")
print("-" * 80)
for name, res in all_results.items():
    short = name.split("/")[-1]
    print(f"{short:<40} {res['accuracy']:>6.3f} {res['macro_f1']:>8.3f} "
          f"{res['unanswerable_f1']:>10.3f} {res['unanswerable_recall']:>11.3f}")

# ── Per corruption-type recall (unanswerable class only) ──────────────────────
print("\n=== Unanswerable recall by corruption type ===")
ctypes = sorted(set(s["corruption_type"] for s in samples if s["label"] == "UNANSWERABLE"))
print(f"{'Model':<35}", end="")
for ct in ctypes:
    print(f"  {ct[:18]:>18}", end="")
print()
print("-" * (35 + 20 * len(ctypes)))

for name, res in all_results.items():
    short = name.split("/")[-1][:34]
    print(f"{short:<35}", end="")
    unansw = [p for p in res["per_sample"] if p["label"] == "UNANSWERABLE"]
    for ct in ctypes:
        ct_items = [p for p in unansw if p["corruption_type"] == ct]
        recall = sum(p["correct"] for p in ct_items) / len(ct_items) if ct_items else 0.0
        print(f"  {recall:>18.3f}", end="")
    print()

# ── Plots ─────────────────────────────────────────────────────────────────────
fig_dir = Path(CONFIG["figures_dir"])
model_names = list(all_results.keys())
short_names = [n.split("/")[-1] for n in model_names]

# 1. Overall metrics bar chart
metrics = ["accuracy", "macro_f1", "unanswerable_f1", "unanswerable_recall"]
x = np.arange(len(model_names))
width = 0.18
fig, ax = plt.subplots(figsize=(12, 5))
for i, m in enumerate(metrics):
    vals = [all_results[n][m] for n in model_names]
    ax.bar(x + i * width, vals, width, label=m.replace("_", " "))
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(short_names, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Overall benchmark metrics per model")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "overall_metrics.jpg", dpi=150)
plt.show()
print(f"Saved → {fig_dir}/overall_metrics.jpg")

# 2. Per-type unanswerable recall heatmap
recall_matrix = []
for name in model_names:
    row = []
    unansw = [p for p in all_results[name]["per_sample"] if p["label"] == "UNANSWERABLE"]
    for ct in ctypes:
        ct_items = [p for p in unansw if p["corruption_type"] == ct]
        row.append(sum(p["correct"] for p in ct_items) / len(ct_items) if ct_items else 0.0)
    recall_matrix.append(row)

fig, ax = plt.subplots(figsize=(max(6, len(ctypes) * 2), len(model_names) + 1))
im = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(ctypes)))
ax.set_xticklabels(ctypes, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(short_names, fontsize=9)
for i in range(len(model_names)):
    for j in range(len(ctypes)):
        ax.text(j, i, f"{recall_matrix[i][j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Unanswerable recall")
ax.set_title("Unanswerable recall by model and corruption type")
plt.tight_layout()
plt.savefig(fig_dir / "per_type_recall.jpg", dpi=150)
plt.show()
print(f"Saved → {fig_dir}/per_type_recall.jpg")

# 3. Confusion matrices
n_models = len(model_names)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
if n_models == 1:
    axes = [axes]
labels_int = [s["label_int"] for s in samples]
for ax, name in zip(axes, model_names):
    preds_int = [1 if p["prediction"] == "ANSWERABLE" else 0
                 for p in all_results[name]["per_sample"]]
    cm = confusion_matrix(labels_int, preds_int)
    disp = ConfusionMatrixDisplay(cm, display_labels=["UNANSW", "ANSW"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name.split("/")[-1], fontsize=9)
plt.suptitle("Confusion matrices", fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / "confusion_matrices.jpg", dpi=150)
plt.show()
print(f"Saved → {fig_dir}/confusion_matrices.jpg")

print("\nPart 2 complete.")
print(f"Results saved → {CONFIG['results_path']}")
