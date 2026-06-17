"""
Unanswerable Question Detection — Complete Pipeline

Usage
-----
    uv run python pipeline.py --part 1          # corruption + judge
    uv run python pipeline.py --part 2          # benchmark (4 VLMs)
    uv run python pipeline.py --part 3          # mitigation (5 strategies)
    uv run python pipeline.py                   # all three parts in sequence

    uv run python pipeline.py --part 1 --window-size 3  # multi-page windowing
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIG = {
    # Dataset — DocVQA-2026 val has 25 docs / 80 questions (nested schema)
    "dataset_name":  "VLR-CVC/DocVQA-2026",
    "dataset_split": "val",
    "window_size":   1,          # 1 = preview image; >1 = N pages concatenated

    # Corruption
    "combined_ratio": 0.50,      # probability of also attempting combined corruption

    # Judge (must NOT be one of the benchmarked models)
    "judge_model":   "Qwen/Qwen2.5-VL-7B-Instruct",

    # Benchmark models (Part 2)
    # Florence-2 is the Vision Transformer representative
    "benchmark_models": [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ],

    # Paths
    "images_dir":              "data/images",
    "corrupted_dataset_path":  "data/corrupted_dataset.json",
    "benchmark_results_path":  "data/benchmark_results.json",
    "mitigation_results_path": "data/mitigation_results.json",
    "figures_dir":             "data/figures",

    # Evaluation
    "bootstrap_n":  1000,
    "seed":         42,
}

MODEL_SHORT = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-1.6",
    "google/gemma-3-4b-it":              "Gemma-4b",
    "google/gemma-3-12b-it":             "Gemma-12b",
    "Qwen/Qwen2.5-VL-3B-Instruct":      "Qwen2.5-VL-3B",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS  (deferred heavy imports happen inside functions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gc
import json
import random
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_window(pages: list[Image.Image], anchor: int, size: int) -> Image.Image:
    """Concatenate <size> pages around anchor into a single tall image."""
    n = len(pages)
    half = size // 2
    start = max(0, min(anchor - half, n - size))
    end   = min(n, start + size)
    chunk = [p.convert("RGB") for p in pages[start:end]]
    if len(chunk) == 1:
        return chunk[0]
    w = max(p.width  for p in chunk)
    h = sum(p.height for p in chunk)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for p in chunk:
        canvas.paste(p, (0, y)); y += p.height
    return canvas


def load_docvqa(cfg: dict) -> list[dict]:
    """
    Flatten DocVQA-2026's nested document→questions schema.

    Each document item contains:
        preview   — small composite thumbnail of all pages
        document  — list of full-resolution page images
        questions / answers — dicts of lists (question_id, question/answer)

    Image saved per document:
        window_size=1  → preview  (full-doc overview, ~361×512 px)
        window_size>1  → first N pages concatenated vertically
    """
    from datasets import load_dataset

    rng = random.Random(cfg["seed"])
    images_dir = Path(cfg["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)
    window_size = cfg.get("window_size", 1)

    print(f"Loading {cfg['dataset_name']} ({cfg['dataset_split']}) …")
    ds = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])

    samples: list[dict] = []
    for item in ds:
        doc_id  = item["doc_id"]
        n_pages = len(item["document"])
        ans_map = dict(zip(item["answers"]["question_id"], item["answers"]["answer"]))

        suffix   = f"_w{window_size}" if window_size > 1 else ""
        img_path = images_dir / f"{doc_id}{suffix}.jpg"
        if not img_path.exists():
            if window_size == 1:
                item["preview"].convert("RGB").save(img_path)
            else:
                pages = [p.convert("RGB") for p in item["document"][:window_size]]
                _build_window(pages, 0, window_size).save(img_path)

        for qid, question in zip(item["questions"]["question_id"],
                                  item["questions"]["question"]):
            answer = ans_map.get(qid, "")
            if not answer or answer in ("None", "null", ""):
                continue
            samples.append({
                "idx":        qid,
                "question":   question,
                "image_path": str(img_path),
                "doc_id":     doc_id,
                "n_pages":    n_pages,
            })

    rng.shuffle(samples)
    print(f"  {len(samples)} questions from {len(ds)} documents (window_size={window_size})")
    return samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — CORRUPTION STRATEGIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ENTITY_POOL = {
    "DATE":    ["January 2020","March 2019","December 2021","Q3 2018","FY2023","August 2016"],
    "GPE":     ["Paris","Berlin","Tokyo","Sydney","Toronto","Mumbai","Seoul","Nairobi"],
    "ORG":     ["Microsoft","Apple","Amazon","IBM","Google","Samsung","Toyota","Nestlé"],
    "PERSON":  ["John Smith","Sarah Johnson","Michael Lee","Emma Davis","Carlos Ruiz"],
    "MONEY":   ["$1,500","$42,000","$3.2 million","€50,000","£12,800","$750"],
    "PERCENT": ["23%","67%","91%","15%","48%","82%","37%"],
    "CARDINAL":["7","42","156","2,300","18","500","1,250"],
    "ORDINAL": ["second","fourth","seventh","third","fifth"],
    "TIME":    ["9:00 AM","3:30 PM","midnight","noon","6:45 AM"],
}

_ELEMENT_SWAPS = {
    "table":    ["figure","chart","graph","diagram"],
    "figure":   ["table","chart","graph"],
    "chart":    ["table","figure","graph"],
    "graph":    ["table","figure","chart"],
    "footnote": ["appendix","header","caption"],
    "appendix": ["footnote","caption"],
    "caption":  ["footnote","appendix"],
    "list":     ["table","figure"],
    "diagram":  ["table","figure"],
    "image":    ["table","figure"],
    "photo":    ["table","figure"],
}

_LAYOUT_SWAPS = {
    "top":"bottom","bottom":"top","left":"right","right":"left",
    "first":"last","last":"first","upper":"lower","lower":"upper",
    "above":"below","below":"above","previous":"next","next":"previous",
    "beginning":"end","end":"beginning","front":"back","back":"front",
}

_NUM_PCTS   = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
_NUM_COUNTS = [2,3,5,7,10,12,15,20,25,30,50,75,100,150,200,500]


_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable,"-m","spacy","download","en_core_web_sm"],
                           check=True, capture_output=True)
            import spacy
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _nlp_entity(q: str, rng: random.Random) -> Optional[str]:
    doc = _get_nlp()(q)
    candidates = [e for e in doc.ents if e.label_ in _ENTITY_POOL]
    if not candidates:
        return None
    ent  = rng.choice(candidates)
    pool = [v for v in _ENTITY_POOL[ent.label_] if v.lower() != ent.text.lower()]
    if not pool:
        return None
    return q[:ent.start_char] + rng.choice(pool) + q[ent.end_char:]


def _element(q: str, rng: random.Random) -> Optional[str]:
    lower = q.lower()
    hits  = []
    for elem, repls in _ELEMENT_SWAPS.items():
        for m in re.finditer(r'\b' + re.escape(elem) + r'\b', lower):
            repl = rng.choice([r for r in repls if r != elem])
            hits.append((m.start(), m.end(), repl, q[m.start():m.end()]))
    if not hits:
        return None
    start, end, repl, orig = rng.choice(hits)
    if orig.isupper():   repl = repl.upper()
    elif orig[0].isupper(): repl = repl.capitalize()
    return q[:start] + repl + q[end:]


def _layout(q: str, rng: random.Random) -> Optional[str]:
    lower = q.lower()
    hits  = []
    for kw, repl in _LAYOUT_SWAPS.items():
        for m in re.finditer(r'\b' + re.escape(kw) + r'\b', lower):
            orig = q[m.start():m.end()]
            r    = repl.upper() if orig.isupper() else (repl.capitalize() if orig[0].isupper() else repl)
            hits.append((m.start(), m.end(), r))
    if not hits:
        return None
    start, end, r = rng.choice(hits)
    return q[:start] + r + q[end:]


def _numerical(q: str, rng: random.Random) -> Optional[str]:
    patterns = [
        (re.compile(r'\b(\d+\.?\d*)\s*%'), "pct"),
        (re.compile(r'[$€£]\s*(\d[\d,]*\.?\d*)'), "cur"),
        (re.compile(r'(?<!\d)(?!(?:19|20)\d\d\b)(\d{1,3})(?!\d)'), "cnt"),
    ]
    rng.shuffle(patterns)
    for pat, kind in patterns:
        ms = list(pat.finditer(q))
        if not ms:
            continue
        m = rng.choice(ms)
        try:
            orig = float(m.group(1).replace(",",""))
        except ValueError:
            continue
        if kind == "pct":
            pool = [v for v in _NUM_PCTS if abs(v-orig) > 10]
            if pool:
                return q[:m.start()] + f"{rng.choice(pool)}%" + q[m.end():]
        elif kind == "cur":
            sym  = q[m.start()]
            pool = [v for v in _NUM_COUNTS if abs(v-orig) > 3]
            if pool:
                return q[:m.start()] + f"{sym}{rng.choice(pool):,}" + q[m.end():]
        elif kind == "cnt":
            pool = [v for v in _NUM_COUNTS if abs(v-orig) > 2]
            if pool:
                return q[:m.start(1)] + str(rng.choice(pool)) + q[m.end(1):]
    return None


_FUNS = {
    "nlp_entity": _nlp_entity,
    "element":    _element,
    "layout":     _layout,
    "numerical":  _numerical,
}


def _combined(q: str, rng: random.Random) -> Optional[tuple[str, list[str]]]:
    """Apply exactly two corruption types to one question."""
    order = list(_FUNS.keys()); rng.shuffle(order)
    applied, current = [], q
    for t in order:
        if len(applied) >= 2: break
        r = _FUNS[t](current, rng)
        if r and r != current:
            current = r; applied.append(t)
    return (current, applied) if len(applied) >= 2 else None


def generate_candidates(samples: list[dict], combined_ratio: float, rng: random.Random) -> list[dict]:
    """Try every corruption type for every question. Returns all unique candidates."""
    candidates:   list[dict] = []
    seen_keys:    set[str]   = set()   # (idx, ctype)
    seen_texts:   set[str]   = set()   # exact corrupted question strings

    for s in samples:
        q = s["question"]

        for ctype, fn in _FUNS.items():
            c = fn(q, rng)
            if c is None or c == q: continue
            key = f"{s['idx']}|{ctype}"
            if key in seen_keys or c in seen_texts: continue
            seen_keys.add(key); seen_texts.add(c)
            candidates.append({**s, "original_question": q, "corrupted_question": c,
                                "corruption_type": ctype})

        if rng.random() < combined_ratio:
            res = _combined(q, rng)
            if res:
                c, types = res
                key = f"{s['idx']}|combined_{'_'.join(types)}"
                if key not in seen_keys and c not in seen_texts:
                    seen_keys.add(key); seen_texts.add(c)
                    candidates.append({**s, "original_question": q, "corrupted_question": c,
                                        "corruption_type": "combined_" + "_".join(types)})
    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — JUDGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_JUDGE_PROMPT = (
    "You are a strict quality reviewer for document question-answering datasets.\n\n"
    "Original question : {original}\n"
    "Corrupted question: {corrupted}\n"
    "Corruption applied: {ctype}\n\n"
    "Look at the document image. Is the CORRUPTED question genuinely unanswerable?\n"
    "(Unanswerable = the specific entity, element, or position it references is NOT in the document.)\n\n"
    "Answer with exactly one word: YES or NO"
)


def _load_vlm(model_name: str):
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    except (ValueError, ImportError):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    return model, processor


def _unload_vlm(model, processor):
    import torch
    del model, processor; gc.collect(); torch.cuda.empty_cache()


def infer(model, processor, model_name: str, image: Image.Image,
          prompt: str, max_new_tokens: int = 10) -> str:
    import torch
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    apply_fn = getattr(processor,"apply_chat_template",None) or processor.tokenizer.apply_chat_template
    text     = apply_fn(messages, add_generation_prompt=True, tokenize=False)
    device   = next(model.parameters()).device
    inputs   = processor(text=text, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dec = getattr(processor,"decode",None) or processor.tokenizer.decode
    return dec(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_label(response: str) -> str:
    upper = response.upper().strip()
    for line in upper.split("\n"):
        if line.startswith("VERDICT:"):
            rest = line[8:].strip()
            if "UNANSWERABLE" in rest: return "UNANSWERABLE"
            if "ANSWERABLE"   in rest: return "ANSWERABLE"
    for line in reversed(upper.split("\n")):
        line = line.strip()
        if not line: continue
        if "UNANSWERABLE" in line: return "UNANSWERABLE"
        if "ANSWERABLE"   in line: return "ANSWERABLE"
    if "UNANSWERABLE" in upper: return "UNANSWERABLE"
    if "ANSWERABLE"   in upper: return "ANSWERABLE"
    return "ANSWERABLE"


def run_judge(candidates: list[dict], cfg: dict) -> list[dict]:
    from tqdm import tqdm
    print(f"  Loading judge: {cfg['judge_model']}")
    model, processor = _load_vlm(cfg["judge_model"])
    verified = []
    for item in tqdm(candidates, desc="Judge"):
        img    = Image.open(item["image_path"]).convert("RGB")
        prompt = _JUDGE_PROMPT.format(
            original=item["original_question"],
            corrupted=item["corrupted_question"],
            ctype=item["corruption_type"],
        )
        resp = infer(model, processor, cfg["judge_model"], img, prompt, max_new_tokens=5)
        if resp.strip().upper().startswith("YES"):
            verified.append(item)
    _unload_vlm(model, processor)
    return verified


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def part1(cfg: dict):
    Path(cfg["images_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["figures_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60 + "\nPART 1 — Corruption pipeline\n" + "="*60)
    rng = random.Random(cfg["seed"])

    samples = load_docvqa(cfg)

    print(f"\nGenerating candidates (all types per question) …")
    candidates = generate_candidates(samples, cfg["combined_ratio"], rng)
    dist = Counter("combined" if c["corruption_type"].startswith("combined_") else c["corruption_type"]
                   for c in candidates)
    print(f"  Candidates: {len(candidates)}")
    for t, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {t:20s}: {n}")

    print(f"\nJudge verification …")
    verified = run_judge(candidates, cfg)

    out = Path(cfg["corrupted_dataset_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(verified, f, indent=2)

    dist2 = Counter("combined" if v["corruption_type"].startswith("combined_") else v["corruption_type"]
                    for v in verified)
    print(f"\n{'='*60}\nPart 1 — Summary\n{'='*60}")
    print(f"  Source questions      : {len(samples)}")
    print(f"  Candidates generated  : {len(candidates)}")
    print(f"  Verified unanswerable : {len(verified)}")
    print(f"  Type distribution:")
    for t, n in sorted(dist2.items(), key=lambda x: -x[1]):
        print(f"    {t:20s}: {n:3d}  ({n/max(len(verified),1)*100:.0f}%)")
    print(f"  Saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2 — BENCHMARK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASELINE_PROMPT = (
    "Look at this document image carefully.\n"
    "Can the following question be answered based on the content visible in the document?\n\n"
    "Question: {question}\n\n"
    "Answer with one word only: ANSWERABLE or UNANSWERABLE"
)

POSITIVE = "UNANSWERABLE"
LABELS   = ["ANSWERABLE", "UNANSWERABLE"]


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, labels=[POSITIVE],
                 pos_label=POSITIVE, average="binary", zero_division=0)
    return {"accuracy":float(acc),"precision":float(p),"recall":float(r),"f1":float(f)}


def bootstrap_ci(y_true, y_pred, metric: str, n=1000, seed=42) -> tuple[float,float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    rng = np.random.default_rng(seed)
    ns  = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, ns, ns)
        yt  = [y_true[i] for i in idx]
        yp  = [y_pred[i] for i in idx]
        try:
            if metric == "accuracy":
                scores.append(accuracy_score(yt,yp))
            else:
                p,r,f,_ = precision_recall_fscore_support(yt,yp,labels=[POSITIVE],
                             pos_label=POSITIVE,average="binary",zero_division=0)
                scores.append({"precision":p,"recall":r,"f1":f}[metric])
        except Exception:
            pass
    if not scores: return 0.0, 0.0
    return float(np.percentile(scores,2.5)), float(np.percentile(scores,97.5))


def per_type_recall(y_true, y_pred, ctypes, target_type) -> float:
    idxs = [i for i,(l,ct) in enumerate(zip(y_true,ctypes))
            if l == POSITIVE and (ct == target_type or
               (ct.startswith("combined_") and target_type in ct))]
    if not idxs: return float("nan")
    return sum(y_pred[i]==POSITIVE for i in idxs)/len(idxs)


def build_eval_set(verified: list[dict], seed: int):
    rng = random.Random(seed)
    samples = []
    for item in verified:
        samples.append({"image_path":item["image_path"],"question":item["original_question"],
                         "label":"ANSWERABLE","corruption_type":item["corruption_type"],"idx":item["idx"]})
        samples.append({"image_path":item["image_path"],"question":item["corrupted_question"],
                         "label":"UNANSWERABLE","corruption_type":item["corruption_type"],"idx":item["idx"]})
    rng.shuffle(samples)
    return samples


def part2(cfg: dict):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    print("\n" + "="*60 + "\nPART 2 — Benchmark\n" + "="*60)

    with open(cfg["corrupted_dataset_path"]) as f:
        verified = json.load(f)
    eval_samples = build_eval_set(verified, cfg["seed"])
    ground_truth = [s["label"] for s in eval_samples]
    ctypes       = [s["corruption_type"] for s in eval_samples]
    print(f"Eval set: {len(eval_samples)} samples  "
          f"(A={ground_truth.count('ANSWERABLE')}, U={ground_truth.count('UNANSWERABLE')})")

    all_preds: dict[str,list[str]] = {}

    for model_name in cfg["benchmark_models"]:
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        model, processor = _load_vlm(model_name)
        preds = []
        for s in tqdm(eval_samples, desc=short):
            img    = Image.open(s["image_path"]).convert("RGB")
            prompt = BASELINE_PROMPT.format(question=s["question"])
            raw    = infer(model, processor, model_name, img, prompt, max_new_tokens=10)
            preds.append(parse_label(raw))
        all_preds[model_name] = preds
        _unload_vlm(model, processor)
        acc = sum(p==g for p,g in zip(preds,ground_truth))/len(ground_truth)
        print(f"  accuracy={acc:.3f}")

    # Save
    out = Path(cfg["benchmark_results_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"eval_samples":eval_samples,"predictions":all_preds}, f, indent=2)

    # Metrics
    print(f"\n{'='*60}\nPart 2 — Results (UNANSWERABLE as positive class)\n{'='*60}")
    all_ctypes = sorted({("combined" if ct.startswith("combined_") else ct) for ct in ctypes})
    rows = []
    for model_name in cfg["benchmark_models"]:
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        preds = all_preds[model_name]
        m     = compute_metrics(ground_truth, preds)
        ci_lo = {k: bootstrap_ci(ground_truth,preds,k,cfg["bootstrap_n"],cfg["seed"])[0] for k in m}
        ci_hi = {k: bootstrap_ci(ground_truth,preds,k,cfg["bootstrap_n"],cfg["seed"])[1] for k in m}
        rows.append({"model":short, **m,
                     **{f"{k}_lo":v for k,v in ci_lo.items()},
                     **{f"{k}_hi":v for k,v in ci_hi.items()}})

    df = pd.DataFrame(rows).set_index("model")
    print(df[["accuracy","precision","recall","f1"]].round(3).to_string())

    # Per-type recall
    print(f"\nUNANSWERABLE recall per corruption type:")
    for model_name in cfg["benchmark_models"]:
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        preds = all_preds[model_name]
        line  = "  " + f"{short:12s}"
        for ct in all_ctypes:
            r = per_type_recall(ground_truth, preds, ctypes, ct)
            line += f"  {ct}={r:.2f}" if not (r!=r) else f"  {ct}=—"
        print(line)

    _plot_part2(df.reset_index(), all_preds, ground_truth, ctypes, all_ctypes, cfg)
    print(f"\n  Results saved → {out}")


def _plot_part2(df, all_preds, ground_truth, ctypes, all_ctypes, cfg):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fdir = Path(cfg["figures_dir"])
    fdir.mkdir(parents=True, exist_ok=True)
    model_labels = df["model"].tolist()

    # Overall metrics bar chart
    metrics = ["accuracy","precision","recall","f1"]
    colors  = ["steelblue","tomato","seagreen","goldenrod"]
    fig, axes = plt.subplots(1,4,figsize=(17,5))
    for ax,(metric,color) in zip(axes,zip(metrics,colors)):
        vals = df[metric].values
        x    = np.arange(len(model_labels))
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor="white")
        if f"{metric}_lo" in df.columns:
            lo = df[f"{metric}_lo"].values; hi = df[f"{metric}_hi"].values
            ax.errorbar(x, vals, yerr=[vals-lo, hi-vals],
                        fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold")
        ax.set_ylim(0,1.1); ax.axhline(0.5,color="gray",linestyle="--",linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].legend(handles=[plt.Line2D([0],[0],color="gray",linestyle="--")],
                   labels=["random baseline"], fontsize=8)
    plt.suptitle("Benchmark — UNANSWERABLE as positive class (95% CI)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = fdir/"overall_metrics.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # Per-type recall heatmap
    recall_matrix = np.zeros((len(model_labels), len(all_ctypes)))
    model_names   = list(all_preds.keys())
    for i,mn in enumerate(model_names):
        for j,ct in enumerate(all_ctypes):
            recall_matrix[i,j] = per_type_recall(ground_truth, all_preds[mn], ctypes, ct)
    fig, ax = plt.subplots(figsize=(max(6,len(all_ctypes)*2), max(4,len(model_labels)*1.2)))
    im = ax.imshow(recall_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Recall (UNANSWERABLE)")
    ax.set_xticks(range(len(all_ctypes))); ax.set_xticklabels(all_ctypes, fontsize=10)
    ax.set_yticks(range(len(model_labels))); ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title("Recall per Corruption Type", fontsize=13, fontweight="bold")
    for i in range(len(model_labels)):
        for j in range(len(all_ctypes)):
            v = recall_matrix[i,j]
            ax.text(j,i,"—" if v!=v else f"{v:.2f}", ha="center",va="center", fontsize=11,
                    color="black" if 0.3<v<0.8 else "white")
    plt.tight_layout()
    p = fdir/"per_type_recall.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # Confusion matrices
    labels = ["ANSWERABLE","UNANSWERABLE"]
    fig, axes = plt.subplots(1,len(model_names), figsize=(5*len(model_names),4))
    if len(model_names)==1: axes=[axes]
    for ax, mn in zip(axes, model_names):
        cm = confusion_matrix(ground_truth, all_preds[mn], labels=labels)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_yticks([0,1]); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(MODEL_SHORT.get(mn, mn.split("/")[-1]), fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                        color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold"); plt.tight_layout()
    p = fdir/"confusion_matrices.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3 — MITIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = {
    "cot": {
        "name": "Chain-of-Thought", "max_new_tokens": 256,
        "prompt": (
            "Look at this document image carefully.\n\n"
            "Question: {question}\n\n"
            "Think step by step:\n"
            "1. What entity, document element, or spatial position does the question reference?\n"
            "2. Is it clearly present and visible in the document?\n"
            "3. Can the question be fully answered?\n\n"
            "Last line must be exactly one word: ANSWERABLE or UNANSWERABLE"
        ),
    },
    "few_shot_cot": {
        "name": "Few-Shot + CoT", "max_new_tokens": 350,
        "prompt": (
            "You judge whether a question can be answered from a document image.\n\n"
            "EXAMPLE 1 — NLP entity (UNANSWERABLE):\n"
            "Q: What was the unemployment rate in June 2019?\n"
            "Step 1: References date 'June 2019'. Step 2: Document shows 2022-2023 only. Step 3: Absent.\n"
            "Answer: UNANSWERABLE\n\n"
            "EXAMPLE 2 — Element (UNANSWERABLE):\n"
            "Q: What is the label on the x-axis of Figure 2?\n"
            "Step 1: References Figure 2. Step 2: Document has only tables. Step 3: Element missing.\n"
            "Answer: UNANSWERABLE\n\n"
            "EXAMPLE 3 — Layout (UNANSWERABLE):\n"
            "Q: What text appears at the bottom-left?\n"
            "Step 1: References bottom-left. Step 2: Bottom-left is blank. Step 3: No content there.\n"
            "Answer: UNANSWERABLE\n\n"
            "EXAMPLE 4 — Numerical (UNANSWERABLE):\n"
            "Q: Which product had a 92% satisfaction rate?\n"
            "Step 1: References 92%. Step 2: Document shows 78%, 84%, 61% — no 92%. Step 3: Absent.\n"
            "Answer: UNANSWERABLE\n\n"
            "EXAMPLE 5 — ANSWERABLE:\n"
            "Q: What is the title of this document?\n"
            "Step 1: Asks for title. Step 2: Large bold title visible at top. Step 3: Present and clear.\n"
            "Answer: ANSWERABLE\n\n"
            "---\nNow evaluate:\nQuestion: {question}\n\n"
            "Step 1: Step 2: Step 3:\nAnswer (one word): ANSWERABLE or UNANSWERABLE"
        ),
    },
    "knowledge_injection": {
        "name": "Knowledge Injection", "max_new_tokens": 20,
        "prompt": (
            "Examine this document image and determine whether the question can be answered.\n\n"
            "Apply this checklist:\n"
            "  [ENTITY]    Does the specific name/date/value appear in the document?\n"
            "  [ELEMENT]   Does the referenced element type (table/figure/chart) exist here?\n"
            "  [LAYOUT]    Is there content at the specified position (top/bottom/left/right)?\n"
            "  [NUMERICAL] Does the exact quantity or percentage appear in the document?\n\n"
            "UNANSWERABLE if any check fails.\n\n"
            "Question: {question}\n\n"
            "Answer (one word): ANSWERABLE or UNANSWERABLE"
        ),
    },
    "role_based": {
        "name": "Role-Based", "max_new_tokens": 20,
        "prompt": (
            "You are an adversarial document analyst. Your mission: find reasons why questions\n"
            "CANNOT be answered. You are skeptical by default.\n\n"
            "Question: {question}\n\n"
            "Challenge every claim:\n"
            "— Is the named entity/value actually present?\n"
            "— Does the referenced element type actually exist?\n"
            "— Is the stated location actually populated?\n"
            "— Is the exact quantity/percentage actually there?\n\n"
            "Resist the urge to answer. Only say ANSWERABLE if evidence is unmistakable.\n\n"
            "Final verdict (one word): ANSWERABLE or UNANSWERABLE"
        ),
    },
    "self_refine": {
        "name": "Self-Refine", "max_new_tokens": 80,
        "prompt": (
            "Look at this document carefully.\n\n"
            "Question: {question}\n\n"
            "Step 1 — Attempt: try to find the answer.\n"
            "Step 2 — Confidence: rate 0-100 (0=clearly absent, 100=unmistakably present).\n"
            "Step 3 — Verdict: UNANSWERABLE if confidence < 60.\n\n"
            "Format:\n"
            "ATTEMPT: <brief answer or 'not found'>\n"
            "CONFIDENCE: <0-100>\n"
            "VERDICT: ANSWERABLE or UNANSWERABLE"
        ),
    },
}


def part3(cfg: dict):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    print("\n" + "="*60 + "\nPART 3 — Mitigation\n" + "="*60)

    with open(cfg["benchmark_results_path"]) as f:
        bench = json.load(f)
    eval_samples   = bench["eval_samples"]
    ground_truth   = [s["label"] for s in eval_samples]
    ctypes         = [s["corruption_type"] for s in eval_samples]
    baseline_preds = bench["predictions"]
    model_names    = list(baseline_preds.keys())

    all_mit_preds: dict[str, dict[str,list[str]]] = {}

    for model_name in model_names:
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        model, processor = _load_vlm(model_name)
        all_mit_preds[model_name] = {}

        for strat_key, strat in STRATEGIES.items():
            print(f"\n  Strategy: {strat['name']}")
            preds = []
            for s in tqdm(eval_samples, desc=f"{short}/{strat_key}"):
                img    = Image.open(s["image_path"]).convert("RGB")
                prompt = strat["prompt"].format(question=s["question"])
                raw    = infer(model, processor, model_name, img, prompt,
                               max_new_tokens=strat["max_new_tokens"])
                preds.append(parse_label(raw))
            all_mit_preds[model_name][strat_key] = preds
            acc = sum(p==g for p,g in zip(preds,ground_truth))/len(ground_truth)
            print(f"    accuracy={acc:.3f}")

        _unload_vlm(model, processor)
        # Checkpoint
        with open(cfg["mitigation_results_path"],"w") as f:
            json.dump({"eval_samples":eval_samples,"baseline_predictions":baseline_preds,
                       "mitigation_predictions":all_mit_preds}, f, indent=2)
        print(f"  Checkpoint saved")

    # Metrics table
    print(f"\n{'='*60}\nPart 3 — Results\n{'='*60}")
    rows = []
    for model_name in model_names:
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        for cond, preds in [("Baseline", baseline_preds[model_name])] + \
                           [(STRATEGIES[k]["name"], all_mit_preds[model_name][k])
                            for k in STRATEGIES]:
            m = compute_metrics(ground_truth, preds)
            rows.append({"model":short,"condition":cond,**m})

    df = pd.DataFrame(rows)
    print("\nF1 pivot:")
    print(df.pivot(index="condition",columns="model",values="f1").round(3).to_string())
    print("\nRecall pivot:")
    print(df.pivot(index="condition",columns="model",values="recall").round(3).to_string())

    _plot_part3(df, eval_samples, ground_truth, ctypes, baseline_preds,
                all_mit_preds, model_names, cfg)


def _plot_part3(df, eval_samples, ground_truth, ctypes, baseline_preds,
                all_mit_preds, model_names, cfg):
    import numpy as np
    import matplotlib.pyplot as plt

    fdir = Path(cfg["figures_dir"])
    palette = {
        "Baseline":"#888888","Chain-of-Thought":"#4472C4","Few-Shot + CoT":"#ED7D31",
        "Knowledge Injection":"#70AD47","Role-Based":"#9B59B6","Self-Refine":"#E74C3C",
    }
    conditions   = df["condition"].unique().tolist()
    model_labels = df["model"].unique().tolist()
    x = np.arange(len(model_labels))
    n = len(conditions)
    w = 0.7/n

    for metric in ["accuracy","f1","recall"]:
        fig, ax = plt.subplots(figsize=(11,5))
        for k,cond in enumerate(conditions):
            color  = palette.get(cond, "#333333")
            vals   = [df[(df["condition"]==cond)&(df["model"]==m)][metric].values[0]
                      for m in model_labels]
            offset = (k-(n-1)/2)*w
            bars   = ax.bar(x+offset, vals, w, label=cond, color=color, alpha=0.87, edgecolor="white")
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_title(f"{metric.capitalize()} — UNANSWERABLE as positive class",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0,1.12); ax.axhline(0.5,color="gray",linestyle="--",linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(model_labels, rotation=20, ha="right")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        p = fdir/f"mitigation_{metric}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved → {p}")

    # Per-type recall: baseline vs best strategy per model
    all_ctypes = sorted({("combined" if ct.startswith("combined_") else ct) for ct in ctypes})
    fig, axes  = plt.subplots(1,len(model_names), figsize=(6*len(model_names),5))
    if len(model_names)==1: axes=[axes]
    for ax, model_name in zip(axes, model_names):
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        best_key = max(STRATEGIES, key=lambda k:
            df[(df["condition"]==STRATEGIES[k]["name"])&(df["model"]==short)]["f1"].values[0])
        best_preds = all_mit_preds[model_name][best_key]
        xpos = np.arange(len(all_ctypes)); w2 = 0.35
        base_r = [per_type_recall(ground_truth, baseline_preds[model_name], ctypes, ct)
                  for ct in all_ctypes]
        best_r = [per_type_recall(ground_truth, best_preds, ctypes, ct) for ct in all_ctypes]
        ax.bar(xpos-w2/2, base_r, w2, label="Baseline", color="#888888", alpha=0.85)
        ax.bar(xpos+w2/2, best_r, w2, label=f"Best ({STRATEGIES[best_key]['name']})",
               color="#4472C4", alpha=0.85)
        ax.set_title(short, fontsize=11, fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(all_ctypes, fontsize=9)
        ax.set_ylim(0,1.1); ax.set_ylabel("Recall (UNANSWERABLE)")
        ax.axhline(0.5,color="gray",linestyle="--",linewidth=0.7)
        for xi,(b,m) in enumerate(zip(base_r,best_r)):
            if b==b: ax.text(xi-w2/2, b+0.02, f"{b:.2f}", ha="center", fontsize=8)
            if m==m: ax.text(xi+w2/2, m+0.02, f"{m:.2f}", ha="center", fontsize=8)
        if ax is axes[0]: ax.legend(fontsize=9)
    plt.suptitle("Per-Type Recall: Baseline vs Best Mitigation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = fdir/"mitigation_per_type_recall.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Unanswerable Question Detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--part", choices=["1","2","3","all"], default="all",
                        help="Which part to run (default: all)")
    parser.add_argument("--window-size", type=int, default=CONFIG["window_size"],
                        help="Pages per inference window for multi-page mode")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model shortnames to benchmark "
                             "(llava, gemma4b, gemma12b, florence2)")
    args = parser.parse_args()

    cfg = {**CONFIG, "window_size": args.window_size}

    if args.models:
        aliases = {
            "llava":     "llava-hf/llava-v1.6-mistral-7b-hf",
            "gemma4b":   "google/gemma-3-4b-it",
            "gemma12b":  "google/gemma-3-12b-it",
            "qwen3b":    "Qwen/Qwen2.5-VL-3B-Instruct",
        }
        cfg["benchmark_models"] = [
            aliases.get(m.strip(), m.strip()) for m in args.models.split(",")
        ]

    parts = ["1","2","3"] if args.part == "all" else [args.part]
    for p in parts:
        if   p == "1": part1(cfg)
        elif p == "2": part2(cfg)
        elif p == "3": part3(cfg)


if __name__ == "__main__":
    main()
