"""
Diagnostic tool — inspect results and spot-check raw model outputs.

No-GPU analysis of saved results:
    uv run python inspect_outputs.py --distributions
    uv run python inspect_outputs.py --distributions --file data/mitigation_results.json

Re-run a small batch to see raw text (needs GPU):
    uv run python inspect_outputs.py --model llava --strategy self_refine --n 5
    uv run python inspect_outputs.py --model qwen3b --strategy cot --n 3

Test parse_label on any string:
    uv run python inspect_outputs.py --parse "ATTEMPT: found it  CONFIDENCE: 45  VERDICT: not sure"
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

# Suppress the "Setting pad_token_id to eos_token_id" info log from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


# ── Model / strategy aliases ──────────────────────────────────────────────────

MODEL_ALIASES = {
    "llava":    "llava-hf/llava-v1.6-mistral-7b-hf",
    "gemma4b":  "google/gemma-3-4b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "qwen3b":   "Qwen/Qwen2.5-VL-3B-Instruct",
}

MODEL_SHORT = {v: k for k, v in MODEL_ALIASES.items()}
MODEL_SHORT["Qwen/Qwen2.5-VL-7B-Instruct"] = "qwen7b-judge"


# ── Reuse parse_label from pipeline ──────────────────────────────────────────

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


def _explain_parse(response: str) -> str:
    upper = response.upper().strip()
    for line in upper.split("\n"):
        if line.startswith("VERDICT:"):
            rest = line[8:].strip()
            if "UNANSWERABLE" in rest: return "matched VERDICT: line → UNANSWERABLE"
            if "ANSWERABLE"   in rest: return "matched VERDICT: line → ANSWERABLE"
    for line in reversed(upper.split("\n")):
        line = line.strip()
        if not line: continue
        if "UNANSWERABLE" in line: return f"matched last line containing UNANSWERABLE"
        if "ANSWERABLE"   in line: return f"matched last line containing ANSWERABLE"
    if "UNANSWERABLE" in upper: return "fallback: UNANSWERABLE found anywhere in text"
    if "ANSWERABLE"   in upper: return "fallback: ANSWERABLE found anywhere in text"
    return "no match → default ANSWERABLE"


# ── --parse mode ──────────────────────────────────────────────────────────────

def cmd_parse(text: str):
    label = parse_label(text)
    reason = _explain_parse(text)
    print(f"\nInput : {text!r}")
    print(f"Label : {label}")
    print(f"Reason: {reason}\n")


# ── --distributions mode ──────────────────────────────────────────────────────

def cmd_distributions(file: str):
    path = Path(file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    ground_truth = [s["label"] for s in data["eval_samples"]]
    n = len(ground_truth)
    n_pos = ground_truth.count("UNANSWERABLE")
    n_neg = ground_truth.count("ANSWERABLE")
    print(f"\nEval set: {n} samples  (ANSWERABLE={n_neg}, UNANSWERABLE={n_pos})\n")

    def _show(label: str, preds: list[str], indent: str = "  "):
        dist = Counter(preds)
        correct = sum(p == g for p, g in zip(preds, ground_truth))
        acc = correct / n
        # If all predictions are the same → flag it
        flag = " ← ALL SAME" if len(dist) == 1 else ""
        print(f"{indent}{label:35s}  acc={acc:.3f}  "
              f"A={dist['ANSWERABLE']:3d}  U={dist['UNANSWERABLE']:3d}{flag}")

    # Benchmark predictions
    if "predictions" in data:
        print("── Benchmark (baseline) ────────────────────────────────────")
        for model_id, preds in data["predictions"].items():
            short = MODEL_SHORT.get(model_id, model_id.split("/")[-1])
            _show(short, preds)

    # Mitigation predictions
    if "mitigation_predictions" in data:
        print("\n── Mitigation ──────────────────────────────────────────────")
        for model_id, strategies in data["mitigation_predictions"].items():
            short = MODEL_SHORT.get(model_id, model_id.split("/")[-1])
            print(f"\n  {short}")
            if "baseline_predictions" in data:
                _show("baseline", data["baseline_predictions"][model_id], "    ")
            for strat_key, preds in strategies.items():
                _show(strat_key, preds, "    ")

    print()


# ── --model / --strategy spot-check mode ─────────────────────────────────────

STRATEGY_PROMPTS = {
    "baseline": {
        "max_new_tokens": 10,
        "prompt": (
            "Look at this document image carefully.\n"
            "Can the following question be answered based on the content visible in the document?\n\n"
            "Question: {question}\n\n"
            "Answer with one word only: ANSWERABLE or UNANSWERABLE"
        ),
    },
    "cot": {
        "max_new_tokens": 256,
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
        "max_new_tokens": 350,
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
            "Step 1: References 92%. Step 2: Document shows 78%, 84%, 61%. Step 3: Absent.\n"
            "Answer: UNANSWERABLE\n\n"
            "EXAMPLE 5 — ANSWERABLE:\n"
            "Q: What is the title of this document?\n"
            "Step 1: Asks for title. Step 2: Large bold title visible at top. Step 3: Present.\n"
            "Answer: ANSWERABLE\n\n"
            "---\nNow evaluate:\nQuestion: {question}\n\n"
            "Answer (one word): ANSWERABLE or UNANSWERABLE"
        ),
    },
    "knowledge_injection": {
        "max_new_tokens": 20,
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
        "max_new_tokens": 20,
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
        "max_new_tokens": 80,
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


def cmd_spotcheck(model_alias: str, strategy: str, n: int,
                  results_file: str, seed: int):
    import random
    from PIL import Image
    from pipeline import _load_vlm, _unload_vlm, infer

    model_name = MODEL_ALIASES.get(model_alias, model_alias)
    strat = STRATEGY_PROMPTS.get(strategy)
    if strat is None:
        print(f"Unknown strategy '{strategy}'. Choose: {list(STRATEGY_PROMPTS)}")
        return

    # Load eval samples
    path = Path(results_file)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        eval_samples = data["eval_samples"]
    else:
        print(f"Results file not found: {path}  (run pipeline.py --part 2 first)")
        return

    rng = random.Random(seed)
    samples = rng.sample(eval_samples, min(n, len(eval_samples)))

    print(f"\nModel    : {model_name}")
    print(f"Strategy : {strategy}  (max_new_tokens={strat['max_new_tokens']})")
    print(f"Samples  : {n}\n")
    print("─" * 70)

    print("Loading model …")
    model, processor = _load_vlm(model_name)

    for i, s in enumerate(samples, 1):
        img    = Image.open(s["image_path"]).convert("RGB")
        prompt = strat["prompt"].format(question=s["question"])
        raw    = infer(model, processor, model_name, img, prompt,
                       max_new_tokens=strat["max_new_tokens"])
        label  = parse_label(raw)
        reason = _explain_parse(raw)
        correct = "✓" if label == s["label"] else "✗"

        print(f"\n[{i}/{n}]  true={s['label']:13s}  pred={label:13s}  {correct}")
        print(f"  question  : {s['question']}")
        print(f"  ctype     : {s['corruption_type']}")
        print(f"  raw output: {raw!r}")
        print(f"  parse     : {reason}")

    _unload_vlm(model, processor)
    print("\n" + "─" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect pipeline outputs without rerunning everything",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--distributions", action="store_true",
                        help="Show label distribution for every model/strategy")
    parser.add_argument("--file", default="data/mitigation_results.json",
                        help="Results JSON to analyse (default: mitigation_results.json)")
    parser.add_argument("--model", default=None,
                        help="Model alias for spot-check (llava, gemma4b, gemma12b, qwen3b)")
    parser.add_argument("--strategy", default="baseline",
                        help="Strategy for spot-check (default: baseline)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples to spot-check (default: 5)")
    parser.add_argument("--results", default="data/benchmark_results.json",
                        help="benchmark_results.json for spot-check eval set")
    parser.add_argument("--parse", default=None, metavar="TEXT",
                        help="Run parse_label on TEXT and explain the decision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.parse is not None:
        cmd_parse(args.parse)
    elif args.distributions:
        cmd_distributions(args.file)
    elif args.model:
        cmd_spotcheck(args.model, args.strategy, args.n, args.results, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
