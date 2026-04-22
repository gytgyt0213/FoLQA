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
        isinstance(term, tuple) and len(term) == 2 and
        isinstance(term[0], int) and isinstance(term[1], tuple) and
        all(isinstance(r, int) for r in term[1]) and len(term[1]) >= 1
    )

def parse_ip_sample(q: Any) -> Tuple[int, int, int, int, int]:
    assert isinstance(q, tuple) and len(q) == 2, f"ip sample must be 2-tuple, got: {q}"
    left, suffix = q
    assert isinstance(left, tuple) and len(left) == 2, f"left must be 2 paths: {q}"
    p1, p2 = left
    assert is_path(p1) and is_path(p2), f"both branches must be paths: {q}"
    if len(p1[1]) != 1 or len(p2[1]) != 1:
        raise ValueError(f"Not ip (branches must be 1-hop): {q}")
    assert isinstance(suffix, tuple) and len(suffix) == 1 and isinstance(suffix[0], int), \
        f"suffix must be a single relation like (S,), got: {suffix}"

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
    default_path = "../FB15k/train-ip-queries.pkl"
    answer_path = "../FB15k/train-ip-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/ip.json"

    ap = argparse.ArgumentParser(description="Convert IP queries+answers to required JSON format.")
    ap.add_argument("--queries_pkl", default=default_path, help="Path to ip *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to ip *-answers.pkl")
    ap.add_argument("--out", default=out_path, help="Output JSON file")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped = 0

    for _, query_set in queries_obj.items():
        query_iter = list(query_set) if not isinstance(query_set, list) else query_set

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            try:
                e1, r1, e2, r2, s = parse_ip_sample(q)
            except Exception:
                skipped += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            record = {
                "logic_query": [
                    [e1, [r1]],
                    [e2, [r2]],
                    s,
                    0,
                    "i chain"
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1, s]},
                    {"start": e2, "relation": [r2, s]}
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
    print(f"Total: {total}, OK: {ok}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
