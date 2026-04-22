#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

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

def parse_2i_sample(q: Any) -> Tuple[Tuple[int, Tuple[int]], Tuple[int, Tuple[int]]]:

    assert isinstance(q, tuple) and len(q) == 2, f"2i sample must be 2-tuple, got: {q}"
    a, b = q
    assert is_path(a) and is_path(b), f"2i branches must be paths: {q}"
    if len(a[1]) != 1 or len(b[1]) != 1:
        raise ValueError(f"Not a 2i sample (both sides must be 1-hop): {q}")
    e1, rels1 = int(a[0]), (int(a[1][0]),)
    e2, rels2 = int(b[0]), (int(b[1][0]),)
    return (e1, rels1), (e2, rels2)

def answers_to_list(ans: Iterable[int], sort_answers: bool) -> List[int]:
    arr = [int(x) for x in ans]
    if sort_answers:
        arr.sort()
    return arr

def main():
    queries_pkl_path = "../FB15k/train-2i-queries.pkl"# path of logic query
    answers_pkl_path = "../FB15k/train-2i-tp-answers.pkl"# path of answers
    out_file_path = "../FB15k/transformed_answers/2i.json"# path of output json file

    sort_answers = True
    allow_empty_answer = False
    indent = 2
    op_name = "2i"

    queries_obj: Dict[Any, Any] = pickle.load(open(queries_pkl_path, "rb"))
    answers_obj: Dict[Any, Any] = pickle.load(open(answers_pkl_path, "rb"))

    records: List[Dict[str, Any]] = []
    total = ok = skipped_shape = skipped_not_2i = skipped_no_ans = 0

    for sk, query_set in queries_obj.items():
        query_iter = list(query_set) if not isinstance(query_set, list) else query_set

        for q_raw in query_iter:
            total += 1
            q = list2tuple_deep(q_raw)

            if not (isinstance(q, tuple) and len(q) == 2 and is_path(q[0]) and is_path(q[1])):
                skipped_shape += 1
                continue

            try:
                (e1, (r1,)), (e2, (r2,)) = parse_2i_sample(q)
            except Exception:
                skipped_not_2i += 1
                continue

            ans = answers_obj.get(q)
            if ans is None and not allow_empty_answer:
                skipped_no_ans += 1
                continue
            answer_list = [] if ans is None else answers_to_list(ans, sort_answers)

            record = {
                "logic_query": [
                    [e1, [r1]],
                    [e2, [r2]],
                    0,
                    op_name
                ],
                "relational_path": [
                    {"start": e1, "relation": [r1]},
                    {"start": e2, "relation": [r2]},
                ],
                "type": "inter",
                "answer": answer_list
            }
            records.append(record)
            ok += 1

    out_path = Path(out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=indent)

    print(f"✅ Done. Wrote {len(records)} records to {out_path}")
    print(f"Total: {total}, OK: {ok}, Skipped(shape): {skipped_shape}, Skipped(not-2i): {skipped_not_2i}, Skipped(no-ans): {skipped_no_ans}")

if __name__ == "__main__":
    main()
