#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

STRUCT_KEY_2P = ('e', ('r', 'r'))
STRUCT_KEY_3P = ('e', ('r', 'r', 'r'))

def list2tuple_deep(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(list2tuple_deep(i) for i in x)
    if isinstance(x, tuple):
        return tuple(list2tuple_deep(i) for i in x)
    return x

def is_khop_query(q: Any, k: int) -> bool:
    if not (isinstance(q, tuple) and len(q) == 2):
        return False
    e, rels = q
    if not (isinstance(e, int) and isinstance(rels, tuple) and len(rels) == k):
        return False
    return all(isinstance(x, int) for x in rels)

def emit_record(e: int, rels: Tuple[int, ...], answers, sort_answers: bool) -> Dict[str, Any]:
    rels_list = [int(r) for r in rels]
    logic_name = f"{len(rels)}-hop"
    answer_list = [] if answers is None else [int(x) for x in answers]
    if sort_answers and answer_list:
        answer_list.sort()
    relational_path = [{
        "start": int(e),
        "relation": rels_list
    }]

    return {
        "logic_query": [[int(e), rels_list, 0], logic_name],
        "relational_path": relational_path,
        "type": "no_logic",
        "answer": answer_list
    }

def main():
    query_path = r"../FB15k/train-2p-queries.pkl"
    answer_path = r"../FB15k/train-2p-tp-answers.pkl"
    out_path = r"../FB15k/transformed_answers/2p.json"

    ap = argparse.ArgumentParser(description="Convert 2p/3p queries+answers PKL to a single JSON list.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to queries PKL")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to answers PKL")
    ap.add_argument("--out", default=out_path, help="Output JSON file")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    ap.add_argument("--include", choices=["2p", "3p", "both"], default="both",
                    help="Which patterns to convert")
    args = ap.parse_args()

    queries_obj: Dict[Any, List[Any]] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    want_2p = args.include in ("2p", "both")
    want_3p = args.include in ("3p", "both")

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_no_ans = 0

    for struct_key, query_list in queries_obj.items():
        if struct_key == STRUCT_KEY_2P and not want_2p:
            continue
        if struct_key == STRUCT_KEY_3P and not want_3p:
            continue
        if struct_key not in (STRUCT_KEY_2P, STRUCT_KEY_3P):
            continue

        k = 2 if struct_key == STRUCT_KEY_2P else 3

        for q_raw in query_list:
            total += 1
            q = list2tuple_deep(q_raw)
            if not is_khop_query(q, k):
                skipped_shape += 1
                continue

            e, rels = q
            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue

            rec = emit_record(e, rels, ans if ans is not None else [], args.sort_answers)
            records.append(rec)
            ok += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=args.indent)

    print(f"✅ Done. wrote {len(records)} to {out_path}")
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
