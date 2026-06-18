"""
One-shot data preparation script for Part 1 and Part 2.

Downloads everything needed from lmms-lab/DocVQA (HuggingFace):
  - data/docvqa_val.json   QA annotations for the validation split
  - data/images/*.jpg      Document page images (validation + test)

Run once before pypart1.py:
    uv run python download_data.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

IMAGES_DIR = Path("data/images")
QA_FILE    = Path("data/docvqa_val.json")
HF_REPO    = ("lmms-lab/DocVQA", "DocVQA")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
QA_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Validation split: save QA + images ───────────────────────────────────────
print("Loading validation split …")
val_ds = load_dataset(*HF_REPO, split="validation", streaming=True)

qa_records = []
saved = skipped = 0

for row in tqdm(val_ds, desc="validation"):
    doc_id  = row["ucsf_document_id"]
    page_no = int(row["ucsf_document_page_no"]) - 1   # 1-indexed → 0-indexed
    stem    = f"{doc_id}_p{page_no}"
    out_path = IMAGES_DIR / f"{stem}.jpg"

    if not out_path.exists():
        row["image"].convert("RGB").save(out_path, "JPEG", quality=95)
        saved += 1
    else:
        skipped += 1

    qa_records.append({
        "question_id":    row["questionId"],
        "question":       row["question"],
        "valid_answers":  row["answers"],
        "image_id":       doc_id,
        "image_name":     [stem],
        "answer_page_idx": 0,
    })

print(f"  Images — saved: {saved}  skipped: {skipped}")
print(f"  QA records: {len(qa_records)}")

with open(QA_FILE, "w") as f:
    json.dump(qa_records, f)
print(f"  QA saved → {QA_FILE}")

# ── Test split: images only (no answers available) ────────────────────────────
print("\nLoading test split (images only) …")
test_ds = load_dataset(*HF_REPO, split="test", streaming=True)

saved = skipped = 0
for row in tqdm(test_ds, desc="test"):
    doc_id  = row["ucsf_document_id"]
    page_no = int(row["ucsf_document_page_no"]) - 1
    out_path = IMAGES_DIR / f"{doc_id}_p{page_no}.jpg"

    if not out_path.exists():
        row["image"].convert("RGB").save(out_path, "JPEG", quality=95)
        saved += 1
    else:
        skipped += 1

print(f"  Images — saved: {saved}  skipped: {skipped}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_images = len(list(IMAGES_DIR.glob("*.jpg")))
print(f"\nDone. {total_images} images in {IMAGES_DIR}  |  {len(qa_records)} QA records in {QA_FILE}")
