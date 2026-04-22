from __future__ import annotations

import argparse
import csv
import gc
import itertools
import os
import random
import re
import json, time, pathlib
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterable, Set, Any
import difflib

import openai
import requests
import torch
from sentence_transformers import SentenceTransformer, util

from KGResource import KGResources
from entity_meta_join import load_json_data
from Logical_Not_Detector import NegationDetector, NegationDetector_t5

random.seed(42)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_URL   = f"{OLLAMA_HOST}/api/generate"

print("OLLAMA_HOST: ", OLLAMA_HOST)
print("OLLAMA_MODEL: ", OLLAMA_MODEL)
print("OLLAMA_URL: ", OLLAMA_URL)
print("- " * 80)

warnings.filterwarnings("ignore")

_END_TAG_RE       = re.compile(r"\bend\s+of\s+answer\b", re.I | re.S)
_FINAL_ANS_RE     = re.compile(r"Final\s+Answer\s+is\s*(.*)", re.I | re.S)
_BRACES_SPAN_RE   = re.compile(r"\{[^{}]*\}", re.S)

ANSWER_RE = re.compile(r"So\s+the\s+answer\s+is\s*\{[^{}]*\}", re.I)


class BGERelationMatcherLocal:
    def __init__(self, model_path: str = "../../models/BGE-m3", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        else:
            dev = str(device).strip().lower()
            if dev.isdigit():
                if torch.cuda.is_available() and int(dev) < torch.cuda.device_count():
                    device = f"cuda:{dev}"
                else:
                    device = "cpu"
            elif dev.startswith("cuda:"):
                if not torch.cuda.is_available():
                    device = "cpu"
                else:
                    idx = int(dev.split(":", 1)[1])
                    if idx >= torch.cuda.device_count():
                        device = "cpu"
            elif dev not in ("cpu", "cuda"):
                device = "cpu"

        self.device = device
        print(f"[INFO] Loading local BGE model from: {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)

    def match(self, question: str, relation_texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        question_embedding = self.model.encode(question, normalize_embeddings=True)
        relation_embeddings = self.model.encode(relation_texts, normalize_embeddings=True)
        scores = util.cos_sim(question_embedding, relation_embeddings)[0].cpu().tolist()
        ranked_results = sorted(zip(relation_texts, scores), key=lambda x: x[1], reverse=True)
        return ranked_results[:top_k]


def append_record(out_path: pathlib.Path,
                  question: str,
                  logic_expression: str,
                  meta_paths: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        "question": question,
        "logic_expression": logic_expression,
        "meta_paths": meta_paths,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with out_path.open("at", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_template(path: os.PathLike) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8").rstrip()


def pick_alias(alias_list, q):
    q_low = q.lower()
    for a in alias_list:
        if re.search(r'\b' + re.escape(a.lower()) + r'\b', q_low):
            return a
    return max(alias_list, key=lambda a: difflib.SequenceMatcher(None, a.lower(), q_low).ratio())


def call_llm(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "stop": ["end of answer"],
        "options": {
            "temperature": 0.5,
            "num_predict": 524,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


def build_relation_prompt(question: str, relations: Iterable[str]) -> str:
    relations_sorted = sorted(relations)
    return (
        "\n\n"
        f"Q: {question}\n"
        f"Relations: {', '.join(relations_sorted)}\n"
        "Answer:"
    )


def try_cot_ans(question, cot_prmpt: str):
    cur_process = f"Q: {question}\nAnswer:"
    prompt = cot_prmpt + cur_process
    answer = call_llm(prompt)
    print("CoT Reasoning:")
    print(cur_process, answer)
    print("- " * 40)
    return cur_process, answer.split("end of answer")[0].strip()


def parse_llm_answer(raw_answer: str):
    ans = re.sub(r"end of answer.*$", "", raw_answer, flags=re.I).strip()
    ans = ans.split("Answer:")[-1].strip()
    m = re.match(r"\{(Yes|No)\}", ans, flags=re.I)
    if m:
        yes_no = m.group(1).capitalize()
        reasoning = ans[m.end():].lstrip(" .")
    else:
        yes_no, reasoning = "Unknown", ans
    return yes_no, reasoning

def try_ans(question, triples, enough_ans_prompt: str):
    triple_lines = []
    seen = set()
    for path in triples:
        for h, r, t in path:
            line = f"({h}, {r}, {t})"
            if line not in seen:
                triple_lines.append(line)
                seen.add(line)
    if not triple_lines:
        triple_lines.append("no information found in kg")

    known_block = "Knowledge Triples:\n" + "\n".join(triple_lines)
    cur_process = f"Q: {question}\n{known_block}\n<|start_header_id|>assistant<|end_header_id|>\nAnswer:"
    prompt = enough_ans_prompt + cur_process
    answer = call_llm(prompt)

    yesOrNo, reasoning_process = parse_llm_answer(answer)
    print("Try to answer by KG info:")
    print(cur_process, answer)
    print("- " * 40)
    return cur_process, yesOrNo, reasoning_process


def is_entity_id(s: str) -> bool:
    s = s.lstrip()
    return len(s) > 2 and s.startswith(("m.", "g."))


def pick_readable_name(
    value: str,
    question: str,
    qid2name: Dict[str, List[str]],
    force_primary: bool = False,
) -> Optional[str]:
    if not is_entity_id(value):
        return value
    aliases = qid2name.get(value, [])
    if not aliases:
        return None
    if force_primary:
        return aliases[0]
    q_low = question.lower()
    for a in aliases:
        if re.search(r'\b' + re.escape(a.lower()) + r'\b', q_low):
            return a
    return aliases[0]


def triples_to_readable(triples, question, kg):
    readable = []
    for h, r, t in triples:
        h_name = kg.ent_name(h)
        t_name = kg.ent_name(t)
        if h_name is None or t_name is None:
            continue
        readable.append((h_name, r, t_name))
    return readable


def _parse_eids(entityIDs: str) -> List[int]:
    out = []
    for tok in entityIDs.split("&"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith(("g.", "m.")):
            tok = tok.split(".", 1)[1]
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def get_meta_query_answer(
    sub_question: str,
    entityIDs: str,
    max_depth: int,
    KG: "KGResources",
    matcher: Any,
    logical_not_dector: Any,
    enough_ans_tpl: str,
    cot_prompt_tpl: str,
    top_k: int = 3,
    *,
    ref_tail_set: Optional[Set[int]] = None,
    and_mode: bool = False,
) -> Dict[str, Any]:

    TOTAL_SAMPLE_CAP = 200

    def _balanced_sample_for_llm(layers: List[List[Tuple[int, str, int]]],
                                 cap: int = TOTAL_SAMPLE_CAP) -> List[Tuple[int, str, int]]:
        if not layers or cap <= 0:
            return []
        sampled: List[Tuple[int, str, int]] = []
        n_layers = len(layers)
        per_layer_cap = max(1, (cap + n_layers - 1) // n_layers)
        for layer in layers:
            if not layer:
                continue
            buckets: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)
            for trip in layer:
                buckets[trip[1]].append(trip)
            n_rels = max(1, len(buckets))
            per_rel_cap = max(1, per_layer_cap // n_rels)

            chosen = []
            leftovers = []
            for rel_name, trips in buckets.items():
                if len(trips) <= per_rel_cap:
                    chosen.extend(trips)
                else:
                    chosen.extend(random.sample(trips, per_rel_cap))
                    remain = [t for t in trips if t not in chosen]
                    leftovers.extend(remain)

            if len(chosen) < per_layer_cap and leftovers:
                need = per_layer_cap - len(chosen)
                if len(leftovers) <= need:
                    chosen.extend(leftovers)
                else:
                    chosen.extend(random.sample(leftovers, need))

            sampled.extend(chosen)

        if len(sampled) > cap:
            sampled = random.sample(sampled, cap)
        return sampled

    def _remove_triples(layers: List[List[Tuple[int, str, int]]],
                        exclude: Set[Tuple[int, str, int]]) -> List[List[Tuple[int, str, int]]]:
        if not exclude:
            return layers
        new_layers: List[List[Tuple[int, str, int]]] = []
        for layer in layers:
            if not layer:
                new_layers.append([])
            else:
                new_layers.append([t for t in layer if t not in exclude])
        return new_layers

    sub_q: str = sub_question
    start_entities: List[int] = _parse_eids(entityIDs)
    if not start_entities:
        _, reasoning = try_cot_ans(sub_q, cot_prompt_tpl)
        return {
            "meta_query": sub_q,
            "triples_paths": [],
            "reasoning_process": reasoning,
            "relational_paths": [],
            "latest_tail_ids": [],
        }

    layers: List[List[Tuple[int, str, int]]] = []
    frontier: Set[int] = set(start_entities)
    cur_depth = 0
    latest_tail_ids: List[int] = []

    path_buckets: Dict[Tuple[str, ...], Set[int]] = {tuple(): set(start_entities)}
    relational_paths: List[List[str]] = []

    while cur_depth < max_depth:
        relation_texts: List[str] = []
        for h in frontier:
            for rel_id, nbr in KG.one_hop(h):
                r_text = KG.rel_name(rel_id)
                relation_texts.append(r_text)

        if not relation_texts:
            _, reasoning = try_cot_ans(sub_q, cot_prompt_tpl)
            return {
                "meta_query": sub_q,
                "triples_paths": [],
                "reasoning_process": reasoning,
                "relational_paths": relational_paths,
                "latest_tail_ids": latest_tail_ids,
            }

        relation_texts = list(dict.fromkeys(relation_texts))
        top_rels = matcher.match(sub_q, relation_texts, top_k=top_k)
        chosen_rels = [r for r, _ in top_rels]

        new_layer_raw: List[Tuple[int, str, int]] = []
        new_frontier_raw: Set[int] = set()
        next_path_buckets: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)

        for path, tails in path_buckets.items():
            for h in tails:
                for rel_id, nbr in KG.one_hop(h):
                    r_text = KG.rel_name(rel_id)
                    if r_text not in chosen_rels:
                        continue
                    new_layer_raw.append((h, r_text, nbr))
                    new_frontier_raw.add(nbr)
                    next_path_buckets[path + (r_text,)].add(nbr)

        if not new_layer_raw:
            _, reasoning = try_cot_ans(sub_q, cot_prompt_tpl)
            return {
                "meta_query": sub_q,
                "triples_paths": [],
                "reasoning_process": reasoning,
                "relational_paths": relational_paths,
                "latest_tail_ids": latest_tail_ids,
            }

        relational_paths = [list(p) for p in next_path_buckets.keys()]

        need_not = False
        if logical_not_dector is not None and relational_paths:
            det_inputs = [logical_not_dector.build_input(sub_q, p) for p in relational_paths]
            det_out = logical_not_dector.predict(det_inputs)
            need_not = any((s.lower() == "yes") for s in det_out)

        if need_not:
            new_layer_display = [(h, f"not {r}", t) for (h, r, t) in new_layer_raw]

            next_path_buckets_not: Dict[Tuple[str, ...], Set[int]] = {}
            for p, tails in next_path_buckets.items():
                next_path_buckets_not[p] = KG.logical_not(set(tails))

            frontier = set().union(*next_path_buckets_not.values()) if next_path_buckets_not else set()
            path_buckets = next_path_buckets_not
        else:
            new_layer_display = new_layer_raw
            frontier = new_frontier_raw
            path_buckets = next_path_buckets

        layers.append(new_layer_display)
        latest_tail_ids = sorted({t for (_, _, t) in new_layer_display})

        cap = TOTAL_SAMPLE_CAP
        sampled_triples: List[Tuple[int, str, int]] = []

        if and_mode and ref_tail_set:
            last_layer = layers[-1] if layers else []
            priority_pool = [trip for trip in last_layer if trip[2] in ref_tail_set]

            if len(priority_pool) >= cap:
                sampled_triples = random.sample(priority_pool, cap)
            else:
                sampled_triples = list(priority_pool)
                cap_left = cap - len(sampled_triples)
                exclude_set = set(sampled_triples)
                remaining_layers = _remove_triples(layers, exclude_set)
                filler = _balanced_sample_for_llm(remaining_layers, cap=cap_left)
                sampled_triples.extend(filler)
        else:
            sampled_triples = _balanced_sample_for_llm(layers, cap=cap)

        if not sampled_triples:
            triples_flat = list(itertools.chain.from_iterable(layers))
            sampled_triples = triples_flat[:cap] if len(triples_flat) > cap else triples_flat

        readable_triples = [
            [(KG.ent_name(h), r, KG.ent_name(t)) for (h, r, t) in sampled_triples]
        ]

        _, yes_or_no, reasoning = try_ans(sub_q, readable_triples, enough_ans_tpl)
        if yes_or_no.lower().strip().startswith("yes"):
            return {
                "meta_query": sub_q,
                "triples_paths": layers,
                "reasoning_process": reasoning,
                "relational_paths": relational_paths,
                "latest_tail_ids": latest_tail_ids,
            }

        cur_depth += 1
        torch.cuda.empty_cache()
        gc.collect()

    _, reasoning = try_cot_ans(sub_q, cot_prompt_tpl)
    return {
        "meta_query": sub_q,
        "triples_paths": [],
        "reasoning_process": reasoning,
        "relational_paths": relational_paths,
        "latest_tail_ids": latest_tail_ids,
    }


def _parse_logic_expression(logic_experision: str):
    if '&&&' in logic_experision:
        op = 'Logical AND (Integrate the answers to multiple sub-problems to find the same set of results)'
        sub_qs = [q.strip() for q in logic_experision.split('&&&') if q.strip()]
    elif '|||' in logic_experision:
        op = 'Logical OR (Compare or merge the answers to multiple sub-problems)'
        sub_qs = [q.strip() for q in logic_experision.split('|||') if q.strip()]
    else:
        raise ValueError("logic_experision must contain '&&&' or '|||'")

    if not sub_qs:
        raise ValueError("Failed to parse any sub-questions")

    return op, sub_qs


def try_summary_ans(
        question: str,
        answer_process: dict,
        logic_experision: str,
        summary_prompt: str
    ):
    logic_op, sub_questions = _parse_logic_expression(logic_experision)

    prompt_lines = [f"Q: {question}",
                    f"Logic Operation: {logic_op}"]

    for idx, sq in enumerate(answer_process.keys(), 1):
        reasoning = answer_process.get(sq, "None reasoning process")
        prompt_lines.append(f"sub-question {idx}: {sq}")
        prompt_lines.append(f"sub-question {idx} reasoning process: {reasoning}")

    prompt_lines.append("<|eot_id|>")
    prompt_lines.append("<|start_header_id|>assistant<|end_header_id|>")
    filled_prompt = "\n".join(prompt_lines)

    full_prompt = summary_prompt.strip() + "\n" + filled_prompt

    answer = call_llm(full_prompt)
    print("Try to Summarize:")
    print("### prompt:\n", filled_prompt)
    print("### answer:\n", answer)

    return filled_prompt, answer


def main():
    ENTITIES_PATH = r""
    META_QUERY_PATH = r""
    json_path = r"../../simple_question/results/FB15k/test_llma3-8b_outputs.json"# simple quetsion result path

    BGE_m3_path = r"../../models/BGE-m3"
    PLM_path = "../../models/t5-base/"
    Logical_Not_classifer_path = "../../train_classifier_negation/results/fb15k/best_macroF1.pt"# negation detector train result path

    kg_path = r"../KG/FB15k/kb.txt"
    rel2id_path = r"../KG/FB15k/rel2id.txt"
    entId2name_path = r"../create_dataset/FB15k/prompts/id2name_cache.pkl"

    results_csv = Path("./results/fb15k/Mine/Mine_final_answer.csv")
    answer_process_csv = Path(r"./results/fb15k/Mine/Mine_answer_process_csv.csv")

    relation_prompt_path = r"./relation_prompt/filter_best_rel_prompt.txt"
    enough_ans_prompt_path = r"./enough_ans_prompt/prompt.txt"
    cot_prompt_path = r"./CoT/prompt.txt"
    summary_prompt_path = r"./summary/prompt.txt"

    print("Path check:")
    print(f"ENTITIES_PATH     = {Path(ENTITIES_PATH).resolve()}")
    print(f"META_QUERY_PATH   = {Path(META_QUERY_PATH).resolve()}")
    print(f"json_path   = {Path(json_path).resolve()}")
    print(f"BGE_m3_path   = {Path(BGE_m3_path).resolve()}")
    print(f"PLM_path   = {Path(PLM_path).resolve()}")
    print(f"Logical_Not_classifer_path   = {Path(Logical_Not_classifer_path).resolve()}")
    print("")
    print(f"KG_TXT_PATH       = {Path(kg_path).resolve()}")
    print(f"R2N_PATH       = {Path(rel2id_path).resolve()}")
    print(f"Q2N_PATH      = {Path(entId2name_path).resolve()}")
    print("")
    print(f"results_csv           = {results_csv.resolve()}")
    print(f"answer_process_csv          = {Path(answer_process_csv).resolve()}")
    print("")
    print(f"relation_prompt_path       = {Path(relation_prompt_path).resolve()}")
    print(f"enough_ans_prompt_path          = {Path(enough_ans_prompt_path).resolve()}")
    print(f"cot_prompt_path          = {Path(cot_prompt_path).resolve()}")
    print(f"summary_prompt_path          = {Path(summary_prompt_path).resolve()}")
    print("-" * 60)

    DEFAULT_Rleation_TPL = Path(relation_prompt_path)
    DEFAULT_Enough_Ans_TPL = Path(enough_ans_prompt_path)

    DEFAULT_CoT_Prompt = Path(cot_prompt_path)
    DEFAULT_Summary_Prompt = Path(summary_prompt_path)

    parser = argparse.ArgumentParser(description="Top-K relation-path tester")
    parser.add_argument("--joined_csv", default=None)
    parser.add_argument("--entity_csv", default=ENTITIES_PATH)
    parser.add_argument("--meta_csv",   default=META_QUERY_PATH)
    parser.add_argument("--json_path", default=json_path)
    parser.add_argument("--BGE_m3_path", default=BGE_m3_path)
    parser.add_argument("--PLM_path", default=PLM_path)
    parser.add_argument("--Logical_Not_classifer_path", default=Logical_Not_classifer_path)

    parser.add_argument("--kg",         default=kg_path)
    parser.add_argument("--rel2name",         default=rel2id_path)
    parser.add_argument("--qid2name",   default=entId2name_path)

    parser.add_argument("--results_csv",default=results_csv)
    parser.add_argument("--answer_process_csv",default=answer_process_csv)

    parser.add_argument("--sample_questions", default=False, action="store_true")
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_depth", type=int, default=4)

    parser.add_argument("--relation_tpl", type=Path, default=DEFAULT_Rleation_TPL)
    parser.add_argument("--enough_ans_tpl", type=Path, default=DEFAULT_Enough_Ans_TPL)
    parser.add_argument("--cot_prompt_tpl", type=Path, default=DEFAULT_CoT_Prompt)
    parser.add_argument("--summary_prompt_tpl", type=Path, default=DEFAULT_Summary_Prompt)

    parser.add_argument("--cuda_device", default="1")
    args = parser.parse_args()

    t0 = time.time()
    rows = load_json_data(
        args.json_path,
        limit_q=None,
        sample_questions=args.sample_questions,
        num_samples=args.num_samples,
        seed=args.seed
    )

    print(f"[Data] Loaded {len(rows)} rows (unique questions = {len(set(r['question'] for r in rows))}) in {time.time()-t0:.2f}s.")

    matcher = BGERelationMatcherLocal(model_path=args.BGE_m3_path, device=args.cuda_device)
    print("BGE-m3 has been load")

    logical_not_dector = NegationDetector_t5(
        t5_dir=args.PLM_path,
        clf_weights_path=args.Logical_Not_classifer_path,
        device=args.cuda_device,
        max_length=512,
        batch_size=32,
        trans_hidden_dim=256,
        trans_heads=4,
        trans_layers=2,
        trans_dropout=0.1,
    )

    enough_ans_tpl = load_template(args.enough_ans_tpl)
    cot_prompt_tpl = load_template(args.cot_prompt_tpl)
    summary_prompt_tpl = load_template(args.summary_prompt_tpl)
    print(f"Answer prompt has been loaded in {time.time() - t0:.2f}s.")

    args.answer_process_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.answer_process_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "meta_query", "triples", "reasoning_process"])
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "final_answer"])

    kg = KGResources(args.kg, args.rel2name, args.qid2name)

    row_count = 0
    start_time = time.time()
    for row in rows:
        row_count += 1

        question = row["question"]
        logic_expr = row["logic_expression"]
        meta_queries = row["meta_query"]
        entity_ids = row["EntityIDs"]

        try:
            logic_op, _ = _parse_logic_expression(logic_expr)
            is_and = logic_op.startswith("Logical AND")
        except Exception:
            is_and = False

        answer_process = []
        need_original_reasoing = False

        anchor_tail_set: Optional[Set[int]] = None

        for idx, mq in enumerate(meta_queries):
            ref_set_to_use = (anchor_tail_set if (is_and and idx >= 1 and anchor_tail_set) else None)

            res = get_meta_query_answer(
                mq, entity_ids, args.max_depth,
                kg, matcher, logical_not_dector, enough_ans_tpl, cot_prompt_tpl,
                ref_tail_set=ref_set_to_use,
                and_mode=is_and
            )

            triples = res.get("triples_paths") or ["NoneTriples"]
            with open(args.answer_process_csv, "a", newline="", encoding="utf-8") as f_ans:
                writer = csv.writer(f_ans)
                writer.writerow([
                    question,
                    res.get("meta_query"),
                    json.dumps(triples, ensure_ascii=False),
                    res.get("reasoning_process", "None reasoning process")
                ])

            if triples[0] == "NoneTriples" and len(meta_queries) >= 2:
                need_original_reasoing = True
                answer_process.clear()
                break

            if is_and and idx == 0:
                anchor_tail_set = set(res.get("latest_tail_ids", [])) or None

            answer_process.append(res)

        if need_original_reasoing and len(meta_queries) >= 2:
            res = get_meta_query_answer(
                question, entity_ids, args.max_depth,
                kg, matcher, logical_not_dector, enough_ans_tpl, cot_prompt_tpl,
                ref_tail_set=None, and_mode=False
            )

            triples = res.get("triples_paths") or ["NoneTriples"]
            with open(args.answer_process_csv, "a", newline="", encoding="utf-8") as f_ans:
                writer = csv.writer(f_ans)
                writer.writerow([
                    question,
                    res.get("meta_query"),
                    json.dumps(triples, ensure_ascii=False),
                    res.get("reasoning_process", "None reasoning process")
                ])

            answer_process.append(res)

        if len(answer_process) <= 1:
            final_answer = answer_process[0].get("reasoning_process", "None reasoning process")
        else:
            proc_dict = {r["meta_query"]: r["reasoning_process"] for r in answer_process}
            _, final_answer = try_summary_ans(
                question,
                proc_dict,
                logic_expr,
                summary_prompt_tpl
            )

        with open(args.results_csv, "a", newline="", encoding="utf-8") as f_res:
            writer = csv.writer(f_res)
            writer.writerow([question, final_answer])

        if row_count % 5 == 0:
            print(f"{row_count} question have been processed, with a total duration of: {(time.time() - start_time):.2f} s")

    print("All done — incremental CSVs at:", args.answer_process_csv, args.results_csv)


if __name__ == "__main__":
    main()
