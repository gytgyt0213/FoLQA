#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

BRACE_RE = re.compile(r"\{([^}]*)\}", flags=re.S)
SPLIT_RE = re.compile(r"\s*(?:&&&|\|\|\|)\s*")

def norm_q(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def read_csv_simple(csv_path: Path) -> Dict[str, str]:
    q2mo: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if "question" not in rdr.fieldnames or "model_output" not in rdr.fieldnames:
            raise ValueError(f"{csv_path} is missing required columns: question, model_output")
        for row in rdr:
            q = norm_q(row.get("question", ""))
            mo = row.get("model_output", "")
            if q and q not in q2mo:
                q2mo[q] = mo
    return q2mo

def load_json_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} top-level structure must be a list")
    return data

def extract_logic_and_meta(model_output: Optional[str], original_q: str) -> Tuple[str, List[str]]:
    q = original_q.strip()
    if not model_output:
        return q, [q]

    m = BRACE_RE.search(model_output)
    if not m:
        return q, [q]

    inside = (m.group(1) or "").strip()
    if not inside:
        return q, [q]

    parts = [p.strip() for p in SPLIT_RE.split(inside) if p.strip()]
    if len(parts) >= 2:
        return "{" + inside + "}", parts
    else:
        return q, [q]

def main():
    simple_result_path = r""
    json_data_path = r""
    output_path = r""

    parser = argparse.ArgumentParser(description="Merge LLM simplified results into original dataset (add logic_expression & meta_query).")
    parser.add_argument("--csv", default=simple_result_path, help="Path to simple_result CSV (question,model_output).")
    parser.add_argument("--json", default=json_data_path, help="Path to original JSON dataset (list).")
    parser.add_argument("--out", default=output_path, help="Path to output JSON file.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    json_path = Path(args.json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q2mo = read_csv_simple(csv_path)
    records = load_json_records(json_path)

    augmented: List[dict] = []
    stats_total = 0
    stats_found = 0
    stats_fallback = 0
    stats_no_csv_match = 0

    for rec in records:
        stats_total += 1
        q_orig = rec.get("question", "")
        q_key = norm_q(q_orig)
        mo = q2mo.get(q_key)

        if mo is None:
            stats_no_csv_match += 1
            logic_expr, meta_list = q_orig, [q_orig]
        else:
            logic_expr, meta_list = extract_logic_and_meta(mo, q_orig)
            if logic_expr == q_orig and meta_list == [q_orig]:
                stats_fallback += 1
            else:
                stats_found += 1

        new_rec = dict(rec)
        new_rec["logic_expression"] = logic_expr
        new_rec["meta_query"] = meta_list
        augmented.append(new_rec)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved: {out_path.resolve()}")
    print(f"Total records: {stats_total}")

if __name__ == "__main__":
    main()