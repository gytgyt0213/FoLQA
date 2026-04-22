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

def parse_pu_sample(q: Any) -> Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int]]]:
    assert isinstance(q, tuple) and len(q) == 3, f"pu sample must be 3-tuple, got: {q}"
    a, b, u = q
    assert is_union_marker(u), f"3rd element must be union marker (-1,), got: {u}"
    assert is_path(a) and is_path(b), f"first two elements must be paths, got: {q}"

    def hops(p) -> int: return len(p[1])
    if hops(a) == 2 and hops(b) == 1:
        two, one = a, b
    elif hops(a) == 1 and hops(b) == 2:
        two, one = b, a
    else:
        raise ValueError(f"Not a pu sample (expect 2-hop and 1-hop): {q}")

    e2, (r1, r2) = int(two[0]), (int(two[1][0]), int(two[1][1]))
    e1, (r3,)   = int(one[0]), (int(one[1][0]),)
    return (e2, (r1, r2)), (e1, (r3,))

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    query_path = "../FB15k/train-pu-queries.pkl"
    answer_path = "../FB15k/train-pu-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/pu.json"

    ap = argparse.ArgumentParser(description="Convert PU (2-hop ∪ 1-hop) queries+answers PKL to JSON list.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to pu *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to pu *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=out_path, help="Output JSON file (array of records)")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_pu = skipped_no_ans = 0

    for _, query_set in queries_obj.items():
        query_iter = query_set if isinstance(query_set, list) else list(query_set)

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            if not (isinstance(q, tuple) and len(q) == 3 and is_path(q[0]) and is_path(q[1]) and is_union_marker(q[2])):
                skipped_shape += 1
                continue
            try:
                (e2, (r1, r2)), (e1, (r3,)) = parse_pu_sample(q)
            except Exception:
                skipped_not_pu += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            record = {
                "logic_query": [
                    [e2, [r1, r2]],
                    [e1, [r3]],
                    0,
                    "chain u"
                ],
                "relational_path": [
                    {"start": e2, "relation": [r1, r2]},
                    {"start": e1, "relation": [r3]}
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
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-pu): {skipped_not_pu}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
