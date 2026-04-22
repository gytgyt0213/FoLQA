#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

STRUCT_KEY_2PI = (('e', ('r', 'r')), ('e', ('r', 'r')))

def list2tuple_deep(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(list2tuple_deep(i) for i in x)
    if isinstance(x, tuple):
        return tuple(list2tuple_deep(i) for i in x)
    return x

def is_path(term: Any) -> bool:
    return (
        isinstance(term, tuple) and len(term) == 2 and
        isinstance(term[0], int) and isinstance(term[1], tuple) and
        all(isinstance(r, int) for r in term[1]) and len(term[1]) >= 1
    )

def parse_2pi_sample(q: Any) -> Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]:
    assert isinstance(q, tuple) and len(q) == 2, f"2pi sample must be 2-tuple, got: {q}"
    a, b = q
    assert is_path(a) and is_path(b), f"2pi branches must be paths: {q}"
    if len(a[1]) != 2 or len(b[1]) != 2:
        raise ValueError(f"Not a 2pi sample (both sides must be 2-hop): {q}")
    e1, rels1 = int(a[0]), (int(a[1][0]), int(a[1][1]))
    e2, rels2 = int(b[0]), (int(b[1][0]), int(b[1][1]))
    return (e1, rels1), (e2, rels2)

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    default_q = "../FB15k/train-2pi-queries.pkl"
    default_a = "../FB15k/train-2pi-tp-answers.pkl"
    default_out = "../FB15k/transformed_answers/2pi.json"

    ap = argparse.ArgumentParser(description="Convert 2PI (2-hop ∩ 2-hop) queries+answers PKL to JSON list.")
    ap.add_argument("--queries_pkl", default=default_q, help="Path to 2pi *-queries.pkl")
    ap.add_argument("--answers_pkl", default=default_a, help="Path to 2pi *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=default_out, help="Output JSON file (array of records)")
    ap.add_argument("--op_name", default="2-chian i", help="Operation name in logic_query")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_2pi = skipped_no_ans = 0

    for sk, query_set in queries_obj.items():
        query_iter = list(query_set) if not isinstance(query_set, list) else query_set

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            if not (isinstance(q, tuple) and len(q) == 2 and is_path(q[0]) and is_path(q[1])):
                skipped_shape += 1
                continue

            try:
                (e1, (r1, r2)), (e2, (r3, r4)) = parse_2pi_sample(q)
            except Exception:
                skipped_not_2pi += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            record = {
                "logic_query": [
                    [e1, [r1, r2]],
                    [e2, [r3, r4]],
                    0,
                    args.op_name
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1, r2]},
                    {"start": e2, "relation": [r3, r4]},
                ],
                "type": "inter",
                "answer": answer_list
            }
            records.append(record)
            ok += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=args.indent)

    print(f"✅ Done. Wrote {len(records)} records to {out_path}")
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-2pi): {skipped_not_2pi}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
