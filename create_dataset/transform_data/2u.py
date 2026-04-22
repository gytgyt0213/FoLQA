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

def parse_2u_sample(q: Any) -> Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]]]:
    assert isinstance(q, tuple) and len(q) == 3, f"2u sample must be 3-tuple, got: {q}"
    a, b, marker = q
    assert is_union_marker(marker), f"3rd element must be union marker (-1,), got: {marker}"
    assert is_path(a) and is_path(b), f"first two elements must be paths, got: {q}"
    if len(a[1]) != 1 or len(b[1]) != 1:
        raise ValueError(f"Not a 2u sample (both branches must be 1-hop): {q}")

    e1, rels1 = int(a[0]), (int(a[1][0]),)
    e2, rels2 = int(b[0]), (int(b[1][0]),)
    return (e1, rels1), (e2, rels2)

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    query_path = "../FB15k/train-2u-queries.pkl"
    answer_path = "../FB15k/train-2u-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/2u.json"

    ap = argparse.ArgumentParser(description="Convert 2U (2-hop ∪ 2-hop) queries+answers PKL to JSON list.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to 2u *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to 2u *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=out_path, help="Output JSON file (array of records)")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_2u = skipped_no_ans = 0

    for _, query_set in queries_obj.items():
        query_iter = query_set if isinstance(query_set, list) else list(query_set)

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)
            if not (isinstance(q, tuple) and len(q) == 3 and is_path(q[0]) and is_path(q[1]) and is_union_marker(q[2])):
                skipped_shape += 1
                continue

            try:
                (e1, (r1,)), (e2, (r2,)) = parse_2u_sample(q)
            except Exception:
                skipped_not_2u += 1
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
                    0,
                    "2u"
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1]},
                    {"start": e2, "relation": [r2]}
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
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-2u): {skipped_not_2u}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
