#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

def list2tuple_deep(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(list2tuple_deep(i) for i in x)
    if isinstance(x, tuple):
        return tuple(list2tuple_deep(i) for i in x)
    return x

def is_path(term: Any) -> bool:
    return (
        isinstance(term, tuple) and len(term) == 2
        and isinstance(term[0], int)
        and isinstance(term[1], tuple) and len(term[1]) >= 1
        and all(isinstance(r, int) for r in term[1])
    )

def is_union_marker(term: Any) -> bool:
    return isinstance(term, tuple) and len(term) == 1 and term[0] == -1

def is_suffix(term: Any) -> bool:
    return isinstance(term, tuple) and len(term) >= 1 and all(isinstance(r, int) for r in term)

def parse_up_sample(q: Any) -> Tuple[int, int, int, int, int]:
    assert isinstance(q, tuple) and len(q) == 2, f"up sample must be 2-tuple, got: {q}"
    left, suffix = q
    assert isinstance(left, tuple) and len(left) == 3, f"left must be 3-tuple (path,path,(-1,)): {q}"
    p1, p2, marker = left
    assert is_path(p1) and is_path(p2) and is_union_marker(marker), f"left must be two paths and (-1,): {q}"
    if len(p1[1]) != 1 or len(p2[1]) != 1:
        raise ValueError(f"Not an up sample (both branches must be 1-hop): {q}")
    assert is_suffix(suffix) and len(suffix) == 1, f"suffix must be a single relation like (S,), got: {suffix}"

    e1, r1 = int(p1[0]), int(p1[1][0])
    e2, r2 = int(p2[0]), int(p2[1][0])
    s = int(suffix[0])
    return e1, r1, e2, r2, s

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    query_path = "../FB15k/train-up-queries.pkl"
    answer_path = "../FB15k/train-up-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/up.json"

    ap = argparse.ArgumentParser(description="Convert UP (union then single-hop chain) queries+answers PKL to JSON list.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to up *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to up *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=out_path, help="Output JSON file (array of records)")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_up = skipped_no_ans = 0

    for _, query_set in queries_obj.items():
        query_iter = query_set if isinstance(query_set, list) else list(query_set)
        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)
            if not (isinstance(q, tuple) and len(q) == 2 and
                    isinstance(q[0], tuple) and len(q[0]) == 3 and
                    is_path(q[0][0]) and is_path(q[0][1]) and is_union_marker(q[0][2]) and
                    is_suffix(q[1])):
                skipped_shape += 1
                continue

            try:
                e1, r1, e2, r2, s = parse_up_sample(q)
            except Exception:
                skipped_not_up += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            record = {
                "logic_query": [
                    [e1, [r1]],
                    [e2, [r2]],
                    s,
                    0,
                    "u chain"
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1, s]},
                    {"start": e2, "relation": [r2, s]}
                ],
                "type": "union",
                "answer": answer_list
            }
            records.append(record)
            ok += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=args.indent)

    print(f"✅ Done. Wrote {len(records)} records to {out_path}")
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-up): {skipped_not_up}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
