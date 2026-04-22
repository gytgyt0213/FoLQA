#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Any

def parse_2in_query(q: tuple) -> Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int, int]]]:
    assert isinstance(q, tuple) and len(q) == 2, f"Unexpected 2in query shape: {q}"
    p1, p2 = q
    assert isinstance(p1[0], int) and isinstance(p2[0], int)
    assert isinstance(p1[1], tuple) and isinstance(p2[1], tuple)
    return p1, p2

def to_relational_path_for_2in(p1, p2) -> List[Dict[str, Any]]:
    (s1, rels1), (s2, rels2) = p1, p2
    return [
        {"start": int(s1), "relation": [int(x) for x in rels1]},
        {"start": int(s2), "relation": [int(x) for x in rels2]},
    ]

def to_logic_query_list_for_2in(p1, p2, placeholder: int, op_name: str) -> List[Any]:
    (s1, rels1), (s2, rels2) = p1, p2
    return [
        [int(s1), [int(x) for x in rels1]],
        [int(s2), [int(x) for x in rels2]],
        int(placeholder),
        op_name
    ]

def main():

    query_path = "../FB15k/train-pin-queries.pkl"
    answer_path = "../FB15k/train-pin-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/pin.json"


    ap = argparse.ArgumentParser(description="Convert 2IN queries+answers PKL to a single JSON list file.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to 2in-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to answers.pkl")
    ap.add_argument("--out", default=out_path, help="Output JSON file path")
    ap.add_argument("--op_name", default="2-chain ni", help="Operation name (e.g., '2chain-not-inter')")
    ap.add_argument("--placeholder", type=int, default=0, help="Placeholder value in logic_query")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answer ids before writing")
    ap.add_argument("--indent", type=int, default=2, help="Indent level for JSON pretty-print")
    args = ap.parse_args()

    queries_obj = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj = pickle.load(open(args.answers_pkl, "rb"))

    records = []

    for sk, query_set in queries_obj.items():
        for q in query_set:
            if not (isinstance(q, tuple) and len(q) == 2):
                continue
            ans = answers_obj.get(q)
            if ans is None:
                continue

            p1, p2 = parse_2in_query(q)
            relational_path = to_relational_path_for_2in(p1, p2)
            logic_query = to_logic_query_list_for_2in(p1, p2, args.placeholder, args.op_name)

            answers_list = sorted(int(x) for x in ans) if args.sort_answers else list(map(int, ans))

            records.append({
                "logic_query": logic_query,
                "relational_path": relational_path,
                "type": "inter",
                "answer": answers_list
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=args.indent)

    print(f"✅ Done. Wrote {len(records)} records to {out_path}")

if __name__ == "__main__":
    main()
