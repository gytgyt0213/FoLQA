import random
import re
import json, time, pathlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set
from collections import defaultdict, OrderedDict
import csv
import argparse

_OP_TOKENS = {"&&&", "|||"}
_BRACE_RE = re.compile(r"\{([^}]*)\}")


def _tokenize(expr: str):
    i = 0
    while i < len(expr):
        if expr[i] in "()":
            yield expr[i]
            i += 1
        elif expr[i : i + 3] in _OP_TOKENS:
            yield expr[i : i + 3]
            i += 3
        else:
            j = i
            while j < len(expr) and expr[j] not in "()":
                if expr[j : j + 3] in _OP_TOKENS:
                    break
                j += 1
            yield expr[i:j].strip()
            i = j


def _parse(tokens, k=0):
    def term(i):
        if tokens[i] == "(":
            node, nxt = _parse(tokens, i + 1)
            if nxt >= len(tokens) or tokens[nxt] != ")":
                raise ValueError("Unmatched '(' in logic expression")
            return node, nxt + 1
        return tokens[i], i + 1

    node, k = term(k)
    while k < len(tokens) and tokens[k] in _OP_TOKENS:
        op = tokens[k]
        t2, k = term(k + 1)
        if isinstance(node, dict) and node["op"] == op:
            node["terms"].append(t2)
        else:
            node = {"op": op, "terms": [node, t2]}
    return node, k


def _stringify(node, parent=None):
    if isinstance(node, str):
        return node
    s = node["op"].join(_stringify(t, node["op"]) for t in node["terms"])
    return f"({s})" if parent and parent != node["op"] else s


def _dedup_ast(node):
    if isinstance(node, str):
        return node
    seen, terms = set(), []
    for t in node["terms"]:
        t = _dedup_ast(t)
        key = _stringify(t)
        if key not in seen:
            seen.add(key)
            terms.append(t)
    return terms[0] if len(terms) == 1 else {"op": node["op"], "terms": terms}


def dedup_logic(model_output: str) -> str:
    """
    From the LLM's answer block, extract the first `{ ... }` and deduplicate it.
    Return the original answer text with only the content inside braces replaced
    by the deduplicated expression. If no braces are found, return the original text.
    """
    m = _BRACE_RE.search(model_output)
    if not m:
        return model_output.strip()

    inside = m.group(1).strip()
    try:
        ast, _ = _parse(list(_tokenize(inside)))
        new_inside = _stringify(_dedup_ast(ast))
    except Exception:
        new_inside = inside

    return "{" + new_inside + "}"


def _gather_leaves(node):
    if isinstance(node, str):
        return [node.strip()]
    leaves: List[str] = []
    for t in node["terms"]:
        leaves.extend(_gather_leaves(t))
    return leaves


def split_meta_queries(logic_expr_text: str) -> List[str]:
    """
    Given an answer text that contains `{...}`, extract and deduplicate all leaf
    meta-queries (preserving the order of first appearance).
    """
    m = _BRACE_RE.search(logic_expr_text)
    if not m:
        return []
    try:
        ast, _ = _parse(list(_tokenize(m.group(1).strip())))
    except Exception:
        return []

    seen, uniq = set(), []
    for q in _gather_leaves(ast):
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def load_eval_rows(
    meta_csv: Union[str, Path],
    entity_csv: Union[str, Path],
    limit_q: Optional[int] = None,
    *,
    dedup_meta: bool = True,
    sample_questions: bool = False,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Read the new-format meta_csv (question, model_output), parse logic expressions
    and leaf meta-queries, then combine with entity_csv (Question, EntityIDs) to
    produce evaluation rows.

    Each returned row contains:
        question
        logic_expression   — the full answer text after deduplication
        meta_query         — one leaf meta-query
        subQuestion        — same as meta_query (placeholder for backward compatibility)
        EntityIDs
    """
    q2mids: Dict[str, str] = OrderedDict()
    with Path(entity_csv).open(encoding="utf-8", newline="") as f_ent:
        rdr_ent = csv.DictReader(f_ent)
        for r in rdr_ent:
            q = r["Question"].strip()
            mids_str = r["EntityIDs"].strip()
            if q in q2mids:
                merged = q2mids[q].split("&") + mids_str.split("&")
                uniq = []
                for m in merged:
                    m = m.strip()
                    if m and m not in uniq:
                        uniq.append(m)
                q2mids[q] = "&".join(uniq)
            else:
                q2mids[q] = mids_str

    rows: List[Dict[str, object]] = []
    q2info: Dict[str, Dict[str, object]] = OrderedDict()

    with Path(meta_csv).open(encoding="utf-8", newline="") as f_meta:
        rdr_meta = csv.DictReader(f_meta)
        for r in rdr_meta:
            question = r["question"].strip()
            model_output = r["model_output"]

            if limit_q is not None and question not in q2info and len(q2info) >= limit_q:
                break

            logic_expr = dedup_logic(model_output)

            meta_qs = split_meta_queries(logic_expr)
            if not meta_qs:
                meta_qs = [question]

            info = q2info.setdefault(
                question,
                {
                    "question": question,
                    "logic_expression": logic_expr,
                    "meta_query": [],
                },
            )

            if dedup_meta:
                seen = set(info["meta_query"])
                for mq in meta_qs:
                    if mq not in seen:
                        seen.add(mq)
                        info["meta_query"].append(mq)
            else:
                info["meta_query"].extend(meta_qs)

            if len(info["meta_query"]) == 1:
                info["meta_query"] = [question]

    for q, info in q2info.items():
        mids = q2mids.get(q, "")
        rows.append(
            {
                "question": q,
                "logic_expression": info["logic_expression"],
                "meta_query": info["meta_query"],
                "EntityIDs": mids,
            }
        )

    if sample_questions and num_samples is not None and num_samples < len(rows):
        random.seed(seed)
        keep_qs = set(random.sample([r["question"] for r in rows], num_samples))
        rows = [r for r in rows if r["question"] in keep_qs]

    return rows


def load_eval_rows_original(
    meta_csv: Union[str, Path],
    entity_csv: Union[str, Path],
    limit_q: Optional[int] = None,
    *,
    sample_questions: bool = False,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Same as the original version, but forces meta_query to be [question].
    Returned fields:
        question
        logic_expression   — original answer text (optionally still deduplicated)
        meta_query         — [question] with a single item
        EntityIDs
    """
    q2mids: Dict[str, str] = OrderedDict()
    with Path(entity_csv).open(encoding="utf-8", newline="") as f_ent:
        rdr_ent = csv.DictReader(f_ent)
        for r in rdr_ent:
            q = r["Question"].strip()
            mids_str = r["EntityIDs"].strip()
            merged = (q2mids.get(q, "") + "&" + mids_str).strip("&")
            uniq = []
            for m in merged.split("&"):
                m = m.strip()
                if m and m not in uniq:
                    uniq.append(m)
            q2mids[q] = "&".join(uniq)

    rows: List[Dict[str, object]] = []
    with Path(meta_csv).open(encoding="utf-8", newline="") as f_meta:
        rdr_meta = csv.DictReader(f_meta)
        seen_q = set()

        for r in rdr_meta:
            question = r["question"].strip()

            if limit_q is not None and question not in seen_q and len(seen_q) >= limit_q:
                break
            seen_q.add(question)

            logic_expr = dedup_logic(r["model_output"])

            rows.append(
                {
                    "question":        question,
                    "logic_expression": logic_expr,
                    "meta_query":      [question],
                    "EntityIDs":       q2mids.get(question, ""),
                }
            )

    if sample_questions and num_samples is not None and num_samples < len(rows):
        random.seed(seed)
        keep_qs = set(random.sample([r["question"] for r in rows], num_samples))
        rows = [r for r in rows if r["question"] in keep_qs]

    return rows


def load_json_data_original(
    json_path: Union[str, Path],
    limit_q: Optional[int] = None,
    *,
    sample_questions: bool = False,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Read a JSON list file (each element is one sample) and output row by row
    (no deduplication, no aggregation). Fields required:
      - question: str
      - StartEntityIDs: List[int]

    The returned structure is consistent with the original:
        {
            "question":        question,
            "logic_expression": question,
            "meta_query":      [question],
            "EntityIDs":       "id1&id2&..."
        }
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array, got {type(data).__name__}")

    rows: List[Dict[str, object]] = []
    produced = 0

    for rec in data:
        if not isinstance(rec, dict):
            continue

        q = rec.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()

        raw_ids = rec.get("StartEntityIDs", [])
        ids: List[int] = []
        if isinstance(raw_ids, list):
            for v in raw_ids:
                try:
                    if isinstance(v, str) and v.startswith(("m.", "g.")):
                        v = v.split(".", 1)[1]
                    ids.append(int(v))
                except Exception:
                    continue

        rows.append(
            {
                "question": q,
                "logic_expression": q,
                "meta_query": [q],
                "EntityIDs": "&".join(str(x) for x in ids) if ids else "",
            }
        )
        produced += 1

        if limit_q is not None and produced >= limit_q:
            break

    if sample_questions and num_samples is not None and num_samples < len(rows):
        random.seed(seed)
        rows = random.sample(rows, num_samples)

    return rows


def load_json_data(
    json_path: Union[str, Path],
    limit_q: Optional[int] = None,
    *,
    sample_questions: bool = False,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Read a JSON list file (each element is one sample) and output row by row
    (no deduplication, no merging).
    Required fields:
      - question: str
      - StartEntityIDs: List[int]  (no deduplication; preserve order)
      - logic_expression: str      (fallback to question if missing)
      - meta_query: List[str]      (fallback to [question] if missing/invalid)

    Each returned row contains:
        {
            "question":         question,
            "logic_expression": logic_expression,
            "meta_query":       meta_query,
            "EntityIDs":        "id1&id2&..."
        }
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array, got {type(data).__name__}")

    rows: List[Dict[str, object]] = []
    produced = 0

    for rec in data:
        if not isinstance(rec, dict):
            continue

        q = rec.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()

        logic_expr = rec.get("logic_expression", q)
        if not isinstance(logic_expr, str):
            logic_expr = str(logic_expr)

        mq = rec.get("meta_query", [q])
        if isinstance(mq, str):
            mq = [mq]
        if not (isinstance(mq, list) and all(isinstance(x, str) for x in mq) and mq):
            mq = [q]

        raw_ids = rec.get("StartEntityIDs", [])
        ids: List[int] = []
        if isinstance(raw_ids, list):
            for v in raw_ids:
                try:
                    if isinstance(v, str) and v.startswith(("m.", "g.")):
                        v = v.split(".", 1)[1]
                    ids.append(int(v))
                except Exception:
                    continue

        rows.append(
            {
                "question": q,
                "logic_expression": logic_expr,
                "meta_query": mq,
                "EntityIDs": "&".join(str(x) for x in ids) if ids else "",
            }
        )
        produced += 1

        if limit_q is not None and produced >= limit_q:
            break

    if sample_questions and num_samples is not None and num_samples < len(rows):
        random.seed(seed)
        rows = random.sample(rows, num_samples)

    return rows
