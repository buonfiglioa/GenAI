import os
import json
import re
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from datasets import load_dataset

# Raise PIL's decompression-bomb limit — DocVQA contains very high-res scans
Image.MAX_IMAGE_PIXELS = None

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    # ---- dataset ----
    "qa_file": "data/val.json",   # DUDE val split (extracted from qas.zip)
    "images_dir": "data/images",  # extracted from images tar; files named {page_id}.png
    "num_samples": 300,
    # ---- corruption ----
    # combined = apply two different types to the same question simultaneously
    "corruption_distribution": {"nlp_entity": 0.25, "element": 0.20, "layout": 0.20, "combined": 0.35},
    # ---- judge model ----
    # Must differ from the models benchmarked in Part 2 to avoid circular evaluation.
    "judge_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    # ---- output ----
    "data_dir": "data",
    # ---- misc ----
    "seed": 42,
    # Pages shown per window when the answer spans a multi-page document.
    # 1 = only the answer page; increase to show surrounding context pages.
    "window_size": 1,
}

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)

# ── GPU check ─────────────────────────────────────────────────────────────────
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")
else:
    print("  WARNING: no CUDA device — inference will be very slow on CPU.")

# ── Load DUDE QA data ─────────────────────────────────────────────────────────
print(f"\nLoading DUDE QAs from {CONFIG['qa_file']} …")
with open(CONFIG["qa_file"]) as f:
    raw = json.load(f)

all_records = raw["data"]
print(f"  Total records in file : {len(all_records)}")

# Keep only records with a real answer (filter out the single null entry)
answerable = [
    r for r in all_records
    if r["answers"] and not all(a.strip().lower() in ("", "none", "null") for a in r["answers"])
]
print(f"  Answerable records    : {len(answerable)}")

_element_kws = set(["table","figure","chart","graph","footnote","section",
                    "appendix","diagram","box","list"])
_layout_kws  = set(["top","bottom","left","right","upper","lower","first","last",
                    "header","footer","above","below","beginning","end","next","previous"])

def _is_corruptible(q: str) -> bool:
    words = set(re.findall(r'\b\w+\b', q.lower()))
    return (bool(re.search(r'\b\d+\b', q))
            or bool(words & _element_kws)
            or bool(words & _layout_kws))

corruptible = [r for r in answerable if _is_corruptible(r["question"])]
print(f"  Corruptible records   : {len(corruptible)}  "
      f"({len(corruptible)/len(answerable):.0%} of answerable)")

# Sample from corruptible only — every sampled record will produce a candidate
n = min(CONFIG["num_samples"], len(corruptible))
records = random.sample(corruptible, n)
print(f"  Sampled               : {len(records)}")

# Quick sanity check on structure
r0 = records[0]
answer_page = r0["page_ids"][r0["answer_page_idx"]]
print(f"\nExample record:")
print(f"  questionId      : {r0['questionId']}")
print(f"  question        : {r0['question']}")
print(f"  answers         : {r0['answers']}")
print(f"  doc_id          : {r0['doc_id']}")
print(f"  page_ids        : {r0['page_ids'][:4]}{'...' if len(r0['page_ids']) > 4 else ''}")
print(f"  answer_page_idx : {r0['answer_page_idx']}  →  page '{answer_page}'")
print(f"  expected image  : {CONFIG['images_dir']}/{answer_page}.jpg")

# ── Corruption resources ──────────────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")
print(f"\nspaCy {spacy.__version__} ready.")

ENTITY_POOLS = {
    "DATE":     ["January 2020", "March 2019", "December 2021", "Q3 2022",
                 "FY2023", "2018", "2024", "April 2021", "June 2017"],
    "CARDINAL": ["42", "157", "3200", "12", "88", "1000", "250", "7", "500"],
    "MONEY":    ["$500", "$1,200", "€2,500", "$75,000", "£300", "$10 million"],
    "ORG":      ["Acme Corp", "Global Industries", "TechSolutions Inc.",
                 "National Bureau", "Allied Group", "Pacific Holdings"],
    "PERSON":   ["John Smith", "Maria Garcia", "Robert Johnson", "Emily Chen", "David Miller"],
    "GPE":      ["New York", "London", "Tokyo", "Berlin", "Sydney", "Chicago", "Paris"],
    "PERCENT":  ["15%", "42%", "3.5%", "99%", "0.1%", "67.8%", "28%"],
    "ORDINAL":  ["first", "second", "third", "fourth", "fifth"],
    "LOC":      ["North America", "Eastern Europe", "the Pacific", "Southeast Asia"],
    "TIME":     ["9:00 AM", "3:30 PM", "midnight", "noon"],
}

ELEMENT_REPLACEMENTS = {
    "table":    ["figure", "chart", "graph", "appendix", "footnote"],
    "figure":   ["table", "chart", "diagram", "graph"],
    "chart":    ["table", "figure", "graph", "diagram"],
    "graph":    ["table", "figure", "chart", "diagram"],
    "footnote": ["table", "figure", "appendix", "section"],
    "section":  ["table", "figure", "appendix", "chapter"],
    "appendix": ["table", "figure", "section", "footnote"],
    "diagram":  ["table", "figure", "chart", "graph"],
    "box":      ["table", "figure", "section", "chart"],
    "list":     ["table", "figure", "chart", "section"],
}

LAYOUT_REPLACEMENTS = {
    "top":       ["bottom", "middle"],
    "bottom":    ["top", "middle"],
    "left":      ["right", "center"],
    "right":     ["left", "center"],
    "upper":     ["lower", "middle"],
    "lower":     ["upper", "middle"],
    "first":     ["last", "second"],
    "last":      ["first", "second"],
    "header":    ["footer", "body"],
    "footer":    ["header", "body"],
    "above":     ["below", "beside"],
    "below":     ["above", "beside"],
    "beginning": ["end", "middle"],
    "end":       ["beginning", "middle"],
    "next":      ["previous", "last"],
    "previous":  ["next", "following"],
}


def pick_replacement(pool: list, original: str):
    candidates = [x for x in pool if x.lower() != original.lower()]
    return random.choice(candidates) if candidates else None


# ── Individual corruption functions ──────────────────────────────────────────
def corrupt_nlp_entity(question: str):
    """Replace a spaCy-detected named entity with a pool value of the same type.
    Falls back to replacing the first inline integer if no entity is found."""
    doc = nlp(question)
    candidates = [(e.text, e.label_) for e in doc.ents if e.label_ in ENTITY_POOLS]
    if candidates:
        orig_text, label = random.choice(candidates)
        replacement = pick_replacement(ENTITY_POOLS[label], orig_text)
        if replacement:
            corrupted = question.replace(orig_text, replacement, 1)
            if corrupted != question:
                return {"question": corrupted,
                        "detail": {"original": orig_text, "replacement": replacement,
                                   "entity_type": label}}
    # Fallback: swap the first standalone integer (e.g. "Table 1" -> "Table 4")
    m = re.search(r'\b(\d+)\b', question)
    if m:
        orig_num = m.group(1)
        pool = [str(i) for i in range(1, 20) if str(i) != orig_num]
        replacement = random.choice(pool)
        corrupted = question[:m.start()] + replacement + question[m.end():]
        return {"question": corrupted,
                "detail": {"original": orig_num, "replacement": replacement,
                           "entity_type": "CARDINAL"}}
    return None


def corrupt_element(question: str):
    """Replace the first document-element keyword (table, figure, chart, …)."""
    q_lower = question.lower()
    for element, pool in ELEMENT_REPLACEMENTS.items():
        if re.search(r'\b' + re.escape(element) + r'\b', q_lower):
            replacement = random.choice(pool)
            corrupted = re.sub(r'\b' + re.escape(element) + r'\b',
                               replacement, question, count=1, flags=re.IGNORECASE)
            if corrupted.lower() != question.lower():
                return {"question": corrupted,
                        "detail": {"original": element, "replacement": replacement}}
    return None


def corrupt_layout(question: str):
    """Replace the first spatial/layout keyword (top, left, first, header, …)."""
    q_lower = question.lower()
    for term, pool in LAYOUT_REPLACEMENTS.items():
        if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
            replacement = random.choice(pool)
            corrupted = re.sub(r'\b' + re.escape(term) + r'\b',
                               replacement, question, count=1, flags=re.IGNORECASE)
            if corrupted.lower() != question.lower():
                return {"question": corrupted,
                        "detail": {"original": term, "replacement": replacement}}
    return None


CORRUPTION_FNS = {
    "nlp_entity": corrupt_nlp_entity,
    "element":    corrupt_element,
    "layout":     corrupt_layout,
}

# ── Combined corruption ───────────────────────────────────────────────────────
COMBINED_PAIRS = [
    ("nlp_entity", "element"),
    ("nlp_entity", "layout"),
    ("element",    "layout"),
]

def corrupt_combined(question: str):
    """Try each pair; return (result_dict, 'typeA+typeB') for the first pair that succeeds."""
    for type1, type2 in COMBINED_PAIRS:
        r1 = CORRUPTION_FNS[type1](question)
        if r1 is None:
            continue
        r2 = CORRUPTION_FNS[type2](r1["question"])
        if r2 is None:
            continue
        detail = {
            "step1": {"type": type1, **r1["detail"]},
            "step2": {"type": type2, **r2["detail"]},
        }
        return {"question": r2["question"], "detail": detail}, f"{type1}+{type2}"
    return None, None


# ── apply_corruption: unified entry point ────────────────────────────────────
FALLBACK_ORDER = ["nlp_entity", "element", "layout"]

def apply_corruption(question: str, preferred_type: str):
    if preferred_type == "combined":
        result, used = corrupt_combined(question)
        if result is not None:
            return result, used
        for ctype in FALLBACK_ORDER:
            r = CORRUPTION_FNS[ctype](question)
            if r is not None:
                return r, ctype
        return None, None

    # Try preferred single type
    result = CORRUPTION_FNS[preferred_type](question)
    if result is not None:
        return result, preferred_type
    # Try remaining single types
    for ctype in FALLBACK_ORDER:
        if ctype == preferred_type:
            continue
        r = CORRUPTION_FNS[ctype](question)
        if r is not None:
            return r, ctype
    # Last resort: combined
    result, used = corrupt_combined(question)
    if result is not None:
        return result, used
    return None, None


print("Corruption functions ready.")
print("Types available:", FALLBACK_ORDER + ["combined"])

# ── Generate corrupted candidates ─────────────────────────────────────────────
candidates_path = Path(CONFIG["data_dir"]) / "corrupted_candidates.json"

corruption_types  = list(CONFIG["corruption_distribution"].keys())
corruption_weights = list(CONFIG["corruption_distribution"].values())

if candidates_path.exists():
    print(f"\nLoading cached candidates from {candidates_path} …")
    with open(candidates_path) as f:
        corrupted_data = json.load(f)
    print(f"  Loaded {len(corrupted_data)} candidates.")
    print("  Distribution:", dict(Counter(d["corruption_type"] for d in corrupted_data)))
else:
    corrupted_data = []
    skipped = 0

    for record in tqdm(records, desc="Corrupting"):
        question = record["question"]
        preferred = random.choices(corruption_types, weights=corruption_weights, k=1)[0]
        result, used_type = apply_corruption(question, preferred)

        if result is None:
            skipped += 1
            continue

        answer_page = record["page_ids"][record["answer_page_idx"]]

        corrupted_data.append({
            "questionId":         record["questionId"],
            "doc_id":             record["doc_id"],
            "answer_page":        answer_page,          # page_id → image filename
            "page_ids":           record["page_ids"],   # kept for windowed multi-page use
            "answer_page_idx":    record["answer_page_idx"],
            "original_question":  question,
            "original_answers":   record["answers"],
            "corrupted_question": result["question"],
            "corruption_type":    used_type,
            "corruption_detail":  result["detail"],
            "judge_verdict":      None,                 # filled in by the judge step
        })

    print(f"\nSampled        : {len(records)}")
    print(f"Corrupted      : {len(corrupted_data)}")
    print(f"Skipped        : {skipped}  (no keyword matched any corruption type)")
    print("Distribution   :", dict(Counter(d["corruption_type"] for d in corrupted_data)))

    with open(candidates_path, "w") as f:
        json.dump(corrupted_data, f, indent=2)
    print(f"Saved → {candidates_path}")

# ── Sanity-check: one example per corruption type ─────────────────────────────
print("\n=== One example per corruption type ===")
seen_types = set()
for item in corrupted_data:
    ctype = item["corruption_type"]
    if ctype in seen_types:
        continue
    seen_types.add(ctype)
    det = item["corruption_detail"]
    print(f"\n[{ctype}]")
    print(f"  Original  : {item['original_question']}")
    print(f"  Corrupted : {item['corrupted_question']}")
    if "step1" in det:   # combined
        print(f"  Step 1    : '{det['step1']['original']}' → '{det['step1']['replacement']}' ({det['step1']['type']})")
        print(f"  Step 2    : '{det['step2']['original']}' → '{det['step2']['replacement']}' ({det['step2']['type']})")
    else:
        print(f"  Change    : '{det['original']}' → '{det['replacement']}'")

# ── LLM-as-a-Judge ────────────────────────────────────────────────────────────
dataset_path = Path(CONFIG["data_dir"]) / "corrupted_dataset.json"

if dataset_path.exists():
    print(f"\nFinal dataset already exists at {dataset_path} — skipping judge.")
    with open(dataset_path) as f:
        verified_data = json.load(f)
    print(f"  Loaded {len(verified_data)} verified samples.")
else:
    # ── Check image availability before loading the model ────────────────────
    images_dir = Path(CONFIG["images_dir"])
    missing, available = [], []
    for item in corrupted_data:
        img_path = images_dir / f"{item['answer_page']}.jpg"
        (available if img_path.exists() else missing).append(item)

    print(f"\nImage availability: {len(available)} found / {len(missing)} missing")
    if missing:
        print(f"  WARNING: {len(missing)} records have no image yet.")
        print(f"  These will be skipped — extract the images tar first.")
        print(f"  First missing: {missing[0]['answer_page']}.jpg")

    to_judge = available  # only judge records where the image exists

    # ── Load judge model ──────────────────────────────────────────────────────
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
        ModelClass = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM
        ModelClass = AutoModelForCausalLM

    print(f"\nLoading judge: {CONFIG['judge_model']} …")
    processor = AutoProcessor.from_pretrained(CONFIG["judge_model"])
    judge_model = ModelClass.from_pretrained(
        CONFIG["judge_model"],
        device_map="auto",    # spreads across all available GPUs automatically
        torch_dtype=torch.bfloat16,
    )
    judge_model.eval()
    print("Ready.  Device map:", judge_model.hf_device_map)

    # ── Judge prompt ──────────────────────────────────────────────────────────
    JUDGE_PROMPT = (
        "You are a quality-control system for a document question-answering benchmark.\n"
        "Examine the document image carefully.\n"
        "Can the following question be answered SOLELY from the content visible in this document?\n\n"
        "Question: {question}\n\n"
        "Reply with ONE word only: ANSWERABLE or UNANSWERABLE"
    )

    _MAX_SIDE = 2048   # Qwen2.5-VL downsamples internally anyway; cap avoids OOM
    _input_device = next(judge_model.parameters()).device

    def _prepare_image(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def judge(pil_image: Image.Image, question: str) -> str:
        pil_image = _prepare_image(pil_image)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": JUDGE_PROMPT.format(question=question)},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[pil_image], return_tensors="pt").to(_input_device)
        with torch.no_grad():
            out = judge_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(out[0][input_len:], skip_special_tokens=True).strip().upper()
        return "UNANSWERABLE" if "UNANSWERABLE" in response else "ANSWERABLE"

    verified_data = []
    rejected = 0

    for item in tqdm(to_judge, desc="Judging"):
        img_path = images_dir / f"{item['answer_page']}.jpg"
        img = Image.open(img_path).convert("RGB")
        verdict = judge(img, item["corrupted_question"])
        item["judge_verdict"] = verdict
        if verdict == "UNANSWERABLE":
            verified_data.append(item)
        else:
            rejected += 1

    print(f"\nJudged          : {len(to_judge)}")
    print(f"Kept (unanswerable) : {len(verified_data)}")
    print(f"Rejected (answerable): {rejected}")
    print(f"Pass rate       : {len(verified_data) / max(len(to_judge), 1):.1%}")
    print("Kept by type    :", dict(Counter(d["corruption_type"] for d in verified_data)))

    # ── Unload judge to free GPU memory ──────────────────────────────────────
    import gc
    del judge_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Judge model unloaded.")

    with open(dataset_path, "w") as f:
        json.dump(verified_data, f, indent=2)
    print(f"Saved {len(verified_data)} samples → {dataset_path}")

print(f"\nPart 1 complete. Final dataset: {len(verified_data)} unanswerable samples.")
print(f"Distribution: {dict(Counter(d['corruption_type'] for d in verified_data))}")

