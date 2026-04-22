#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

def list2tuple_deep(x: Any) -> Any:
    if isinstance(x, list):  return tuple(list2tuple_deep(i) for i in x)
    if isinstance(x, tuple): return tuple(list2tuple_deep(i) for i in x)
    return x

def is_path(term: Any) -> bool:
    return (
        isinstance(term, tuple) and len(term) == 2 and
        isinstance(term[0], int) and isinstance(term[1], tuple) and
        all(isinstance(r, int) for r in term[1]) and len(term[1]) >= 1
    )

def parse_2nu_sample(q: Any) -> Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int, int]]]:
    assert isinstance(q, tuple) and len(q) == 3, f"2nu must be 3-tuple, got {q}"
    a, b, flag = q
    assert isinstance(flag, tuple) and len(flag) == 1 and flag[0] == -1, f"union flag missing: {q}"
    assert is_path(a) and is_path(b), f"branches must be paths: {q}"
    e1, rels1 = a
    if len(rels1) != 1 or rels1[0] == -2:
        raise ValueError(f"1st branch must be (r) without neg: {q}")
    e2, rels2 = b
    if len(rels2) != 2 or rels2[1] != -2:
        raise ValueError(f"2nd branch must be (r,-2): {q}")

    return (int(e1), (int(rels1[0]),)), (int(e2), (int(rels2[0]), -2))

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers: arr.sort()
    return arr

def main():
    default_q_path = "../FB15k/train-nu-queries.pkl"
    default_a_path = "../FB15k/train-nu-tp-answers.pkl"
    default_out_path = "../FB15k/transformed_answers/2nu.json"

    ap = argparse.ArgumentParser(description="Convert 2nu (1-hop ∪ (1-hop with NOT)) PKL to JSON.")
    ap.add_argument("--queries_pkl", default=default_q_path, help="../2nu-queries.pkl")
    ap.add_argument("--answers_pkl", default=default_a_path, help="../2nu-*-answers.pkl (tp/fp/fn)")
    ap.add_argument("--out", default=default_out_path, help="Output JSON file")
    ap.add_argument("--op_name", default="chain nu", help="Operation name in logic_query")
    ap.add_argument("--sort_answers", action="store_true")
    ap.add_argument("--allow_empty_answer", action="store_true")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args()

    queries_obj: Dict[Any, Any] = pickle.load(open(args.queries_pkl, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(args.answers_pkl, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skip_shape = skip_not_2nu = skip_no_ans = 0

    for _, query_set in queries_obj.items():
        query_iter = list(query_set) if not isinstance(query_set, list) else query_set
        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            if not (isinstance(q, tuple) and len(q) == 3 and is_path(q[0]) and is_path(q[1])):
                skip_shape += 1
                continue
            try:
                (e1, (r1,)), (e2, (r2, _neg)) = parse_2nu_sample(q)
            except Exception:
                skip_not_2nu += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not args.allow_empty_answer:
                skip_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, args.sort_answers)

            record = {
                "logic_query": [
                    [e1, [r1]],
                    [e2, [r2, "n"]],
                    [ "u" ],
                    args.op_name
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1]},
                    {"start": e2, "relation": [r2, -2]}
                ],
                "type": "chian nu",
                "answer": answer_list
            }
            records.append(record)
            ok += 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=args.indent), encoding="utf-8")

    print(f"✅ Done. Wrote {len(records)} records to {out}")
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skip_shape}, Skipped(not-2nu): {skip_not_2nu}, Skipped(no-ans): {skip_no_ans}")

if __name__ == "__main__":
    main()
