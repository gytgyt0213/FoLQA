#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import string
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import defaultdict
_punc = set(string.punctuation)

_BRACE = re.compile(r"answer is \{([^{}]*)\}")

def extract_last_brace(raw: str) -> str:
    if not raw:
        return ""
    matches = _BRACE.findall(raw)
    return matches[-1].strip() if matches else raw.strip()

def load_llm_preds_csv(csv_path: Union[str, Path],
                       q_field: str = "question",
                       ans_field: str = "final_answer") -> Dict[str, str]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(p)
    q2pred: Dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get(q_field) or "").strip()
            raw = (row.get(ans_field) or "").strip()
            matches = _BRACE.findall(raw)
            pred = matches[-1].strip() if matches else raw
            if q:
                q2pred[q] = pred
    return q2pred

def norm(s: str) -> str:
    s = (s or "").casefold().strip()
    s = ''.join(ch for ch in s if ch not in _punc)
    s = re.sub(r'\s+', ' ', s)
    return s

def relaxed_match(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    return na == nb or (na and nb and (na in nb or nb in na))

def split_pred(ans_text: str) -> List[str]:
    if ans_text is None:
        return []
    parts = re.split(r'\s*(?:,|，|;|；|、| and | AND )\s*', ans_text.strip())
    return [p for p in parts if p]

def score_one(pred_items: List[str], gold_items: List[str]) -> Tuple[int, float, float, float]:
    em = 0
    for p in pred_items:
        if any(relaxed_match(p, g) for g in gold_items):
            em = 1
            break

    hit = 0
    used = [False]*len(gold_items)
    for p in pred_items:
        for j, g in enumerate(gold_items):
            if not used[j] and relaxed_match(p, g):
                used[j] = True
                hit += 1
                break

    P = hit / len(pred_items) if pred_items else 0.0
    R = hit / len(gold_items) if gold_items else 0.0
    F1 = (2*P*R / (P+R)) if (P > 0 and R > 0) else 0.0
    return em, P, R, F1

def get_logic_type(x) -> str:
    if isinstance(x, list) and x:
        return str(x[-1])
    if isinstance(x, str):
        return x
    return ""

def read_json_list(path: Union[str, Path]) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p} should be a list of samples.")
    return data

def evaluate_by_logic(samples: List[dict], llm_preds: Dict[str, str]):
    overall_stats = defaultdict(float)
    logic_stats = defaultdict(lambda: defaultdict(float))
    wrong_indices = []

    for idx, item in enumerate(samples, start=1):
        q = (item.get("question") or "").strip()
        logic_type = get_logic_type(item.get("logic_query", ""))

        ans_field = item.get("answer", [])
        if isinstance(ans_field, list):
            gold_names = [a.strip() for a in ans_field if isinstance(a, str) and a.strip()]
        elif isinstance(ans_field, str):
            gold_names = [ans_field.strip()] if ans_field.strip() else []
        else:
            gold_names = []

        pred_raw = llm_preds.get(q, "")
        pred_items = split_pred(pred_raw)

        em, P, R, F1 = score_one(pred_items, gold_names)

        if em == 0:
            wrong_indices.append(idx)

        overall_stats["EM_sum"] += em
        overall_stats["P_sum"] += P
        overall_stats["R_sum"] += R
        overall_stats["F1_sum"] += F1
        overall_stats["N"] += 1

        logic_stats[logic_type]["EM_sum"] += em
        logic_stats[logic_type]["P_sum"] += P
        logic_stats[logic_type]["R_sum"] += R
        logic_stats[logic_type]["F1_sum"] += F1
        logic_stats[logic_type]["N"] += 1

    overall_result = {
        "EM": overall_stats["EM_sum"] / overall_stats["N"] if overall_stats["N"] else 0.0,
        "Precision": overall_stats["P_sum"] / overall_stats["N"] if overall_stats["N"] else 0.0,
        "Recall": overall_stats["R_sum"] / overall_stats["N"] if overall_stats["N"] else 0.0,
        "F1": overall_stats["F1_sum"] / overall_stats["N"] if overall_stats["N"] else 0.0,
        "Total": int(overall_stats["N"])
    }

    logic_results = {}
    for logic_type, stats in logic_stats.items():
        n = stats["N"]
        logic_results[logic_type] = {
            "EM": stats["EM_sum"] / n if n else 0.0,
            "Precision": stats["P_sum"] / n if n else 0.0,
            "Recall": stats["R_sum"] / n if n else 0.0,
            "F1": stats["F1_sum"] / n if n else 0.0,
            "Total": int(n)
        }

    return overall_result, logic_results, wrong_indices


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def write_errors_csv(path: Path, samples: List[dict], wrong_indices: List[int], q2pred: Dict[str, str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "question", "logic_type", "gold_names", "pred_items", "pred_raw"])
        for idx in wrong_indices:
            item = samples[idx - 1]
            q = (item.get("question") or "").strip()
            logic_type = get_logic_type(item.get("logic_query", ""))

            ans_field = item.get("answer", [])
            if isinstance(ans_field, list):
                gold_names = [a.strip() for a in ans_field if isinstance(a, str) and a.strip()]
            elif isinstance(ans_field, str):
                gold_names = [ans_field.strip()] if ans_field.strip() else []
            else:
                gold_names = []

            pred_raw = q2pred.get(q, "")
            pred_items = split_pred(pred_raw)
            writer.writerow([
                idx, q, logic_type,
                "|".join(gold_names),
                "|".join(map(str, pred_items)),
                pred_raw
            ])


def main():
    path_dataset = r"" # dataset path
    path_result  = r"" # eval result path
    output_path  = r"" # output path

    ap = argparse.ArgumentParser(description="Evaluate with single dataset JSON & single prediction CSV.")
    ap.add_argument("--json",  type=Path, default=Path(path_dataset), help="Gold JSON (single)")
    ap.add_argument("--csv",   type=Path, default=Path(path_result),  help="Prediction CSV (single)")
    ap.add_argument("--out_dir", type=Path, default=Path(output_path),
                    help="Output directory")
    ap.add_argument("--q_field", type=str, default="question",     help="CSV field name for question")
    ap.add_argument("--ans_field", type=str, default="final_answer", help="CSV field name for prediction")

    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_json_list(args.json)
    (out_dir / "conflicts_gold.json").write_text("[]", encoding="utf-8")

    q2pred = load_llm_preds_csv(args.csv, q_field=args.q_field, ans_field=args.ans_field)
    (out_dir / "conflicts_pred.json").write_text("[]", encoding="utf-8")

    overall, by_logic, wrong_indices = evaluate_by_logic(samples, q2pred)

    (out_dir / "overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "by_logic.json").write_text(json.dumps(by_logic, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "wrong_indices.json").write_text(json.dumps(wrong_indices, ensure_ascii=False, indent=2), encoding="utf-8")

    write_errors_csv(out_dir / "errors.csv", samples, wrong_indices, q2pred)

    run_meta = {
        "inputs": {
            "json": str(args.json.resolve()),
            "csv": str(args.csv.resolve()),
        },
        "sha256": {
            "json": sha256_of_file(args.json),
            "csv": sha256_of_file(args.csv),
        },
        "q_field": args.q_field,
        "ans_field": args.ans_field,
        "counts": {
            "samples_merged": len(samples),
            "preds_merged": len(q2pred),
            "wrong_count": len(wrong_indices)
        }
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Overall:")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\nBy Logic Query Type:")
    print(json.dumps(by_logic, ensure_ascii=False, indent=2))
    print("\nWrong Question Indices (count={}):".format(len(wrong_indices)))
    print(wrong_indices)
    print(f"\nResults saved in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
