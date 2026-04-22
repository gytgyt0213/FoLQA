#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

STRUCT_KEY_PI = (('e', ('r', 'r')), ('e', ('r',)))

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

def parse_pi_sample(q: Any) -> Tuple[Tuple[int, Tuple[int, ...]], Tuple[int, Tuple[int, ...]]]:

    assert isinstance(q, tuple) and len(q) == 2, f"pi sample must be 2-tuple, got: {q}"
    a, b = q
    assert is_path(a) and is_path(b), f"pi branches must be paths: {q}"
    len_a, len_b = len(a[1]), len(b[1])
    if len_a == 2 and len_b == 1:
        return a, b
    if len_a == 1 and len_b == 2:
        return b, a
    raise ValueError(f"Not a pi sample (expected 2-hop ∩ 1-hop): {q}")

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    query_path = r"../FB15k/train-pi-queries.pkl"
    answer_path = r"../FB15k/train-pi-tp-answers.pkl"
    out_path = r"../FB15k/transformed_answers/pi.json"

    ap = argparse.ArgumentParser(description="Convert PI (2-hop ∩ 1-hop) queries+answers PKL to JSON list.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to pi *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to pi *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=out_path, help="Output JSON file (array of records)")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_pi = skipped_no_ans = 0

    for sk, query_set in queries_obj.items():
        if isinstance(query_set, list):
            query_iter = query_set
        else:
            query_iter = list(query_set)

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)
            if not (isinstance(q, tuple) and len(q) == 2 and is_path(q[0]) and is_path(q[1])):
                skipped_shape += 1
                continue

            try:
                two_hop, one_hop = parse_pi_sample(q)
            except Exception:
                skipped_not_pi += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            (e2, rels2) = two_hop
            (e1, rels1) = one_hop

            record = {
                "logic_query": [
                    [int(e2), [int(x) for x in rels2]],
                    [int(e1), [int(x) for x in rels1]],
                    0,
                    "chain i"
                ],
                "relational_path": [
                    {"start": int(e2), "relation": [int(x) for x in rels2]},
                    {"start": int(e1), "relation": [int(x) for x in rels1]},
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
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-pi): {skipped_not_pi}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
