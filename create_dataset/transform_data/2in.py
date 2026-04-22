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
        len(term[1]) >= 1 and all(isinstance(r, int) for r in term[1])
    )

def parse_inp_like_sample(q: Any) -> Tuple[Tuple[int, Tuple[int, ...]], Tuple[int, Tuple[int, ...]]]:
    assert isinstance(q, tuple) and len(q) == 2, f"inp sample must be 2-tuple, got: {q}"
    a, b = q
    assert is_path(a) and is_path(b), f"both branches must be paths: {q}"

    def is_neg_path(p) -> bool:
        return len(p[1]) >= 1 and int(p[1][-1]) == -2

    a_neg, b_neg = is_neg_path(a), is_neg_path(b)

    if a_neg == b_neg:
        raise ValueError(f"Not an inp-like (one and only one branch must end with -2): {q}")

    pos = b if a_neg else a
    neg = a if a_neg else b

    pos_e = int(pos[0]); pos_rels = tuple(int(x) for x in pos[1])
    neg_e = int(neg[0]); neg_rels = tuple(int(x) for x in neg[1])
    return (pos_e, pos_rels), (neg_e, neg_rels)

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    query_path = "../FB15k/train-2in-queries.pkl"
    answer_path = "../FB15k/train-2in-tp-answers.pkl"
    out_path = "../FB15k/transformed_answers/2in.json"

    ap = argparse.ArgumentParser(description="Convert INP-like (chain ni) queries+answers to the required JSON format.")
    ap.add_argument("--queries_pkl", default=query_path, help="Path to inp *-queries.pkl")
    ap.add_argument("--answers_pkl", default=answer_path, help="Path to inp *-answers.pkl (tp/fp/fn all OK)")
    ap.add_argument("--out", default=out_path, help="Output JSON file (array of records)")
    ap.add_argument("--sort_answers", action="store_true", help="Sort answers ascending")
    ap.add_argument("--allow_empty_answer", action="store_true", help="Keep items with empty/missing answers")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_inp = skipped_no_ans = 0

    for _, query_set in queries_obj.items():
        query_iter = query_set if isinstance(query_set, list) else list(query_set)

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            if not (isinstance(q, tuple) and len(q) == 2 and is_path(q[0]) and is_path(q[1])):
                skipped_shape += 1
                continue

            try:
                pos_path, neg_path = parse_inp_like_sample(q)
            except Exception:
                skipped_not_inp += 1
                continue
            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            pos_e, pos_rels = pos_path
            neg_e, neg_rels = neg_path
            record = {
                "logic_query": [
                    [pos_e, [int(x) for x in pos_rels]],
                    [neg_e, [int(x) for x in neg_rels]],
                    0,
                    "chain ni"
                ],
                "relational_path": [
                    {"start": pos_e, "relation": [int(x) for x in pos_rels]},
                    {"start": neg_e, "relation": [int(x) for x in neg_rels]}
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
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-inp): {skipped_not_inp}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
