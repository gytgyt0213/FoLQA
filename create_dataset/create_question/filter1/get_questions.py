import os
import pickle
import json
import time
from pathlib import Path
import openai

openai.api_key = "Mine"
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")


def map_rel_ids_to_strings(rel_ids, id2rel: dict) -> list:
    out = []
    for rid in rel_ids:
        rel = id2rel.get(rid)
        out.append(rel if rel is not None else f"[REL_{rid}]")
    return out

def get_path(relational_path: dict):
    if not relational_path:
        return None, []
    return relational_path.get("start"), list(relational_path.get("relation") or [])

def process_rel_ids_with_logic_not(rel_ids, id2rel):
    rel_strs = []
    i = 0
    while i < len(rel_ids):
        if i + 1 < len(rel_ids) and rel_ids[i + 1] == -2:
            rel_str = id2rel.get(rel_ids[i], f"[REL_{rel_ids[i]}]") + " (Logic_NOT)"
            rel_strs.append(rel_str)
            i += 2
        else:
            rel_str = id2rel.get(rel_ids[i], f"[REL_{rel_ids[i]}]")
            rel_strs.append(rel_str)
            i += 1
    return rel_strs

def build_prompt_block_from_sample(
    sample: dict,
    id2name: dict,
    id2rel: dict,
    with_question_placeholder: bool = True
) -> str:
    logic_modle_type = sample.get("logic_query")[-1]

    if logic_modle_type in ["2-hop", "3-hop"]:
        start_id, rel_ids = get_path(sample.get("relational_path")[0])
        if start_id is None:
            raise ValueError("sample is missing a valid relational_path/start")
        start_name = id2name.get(start_id, str(start_id))
        rel_strs = map_rel_ids_to_strings(rel_ids, id2rel)
        path_line = " ; ".join(rel_strs)
        lines = [
            f"Start entity: {start_name}",
            f"Relational path: {path_line}",
        ]
        if with_question_placeholder:
            lines.append("Question:")
        return "\n".join(lines)

    elif logic_modle_type in ["chain u", "2-chain u", "chain i", "2-chain i", "2u", "2i"]:
        paths = sample.get("relational_path")
        lines = []
        for path_id, path in enumerate(paths, 1):
            start_id, rel_ids = get_path(path)
            start_name = id2name.get(start_id, str(start_id))
            rel_strs = map_rel_ids_to_strings(rel_ids, id2rel)
            path_line = " ; ".join(rel_strs)
            lines.append(f"Relational Path P{path_id}")
            lines.append(f"- Start entity: {start_name}")
            lines.append(f"- Relational path: {path_line}")
        if with_question_placeholder:
            lines.append("Question:")
        return "\n".join(lines)

    elif logic_modle_type in ["u chain", "i chain"]:
        paths = sample.get("relational_path") or []
        if len(paths) < 2:
            raise ValueError(f"{logic_modle_type} requires at least two initial paths (P1, P2)")
        (start1, rels1) = get_path(paths[0])
        (start2, rels2) = get_path(paths[1])
        if not rels1 or not rels2:
            raise ValueError(f"Each path in {logic_modle_type} requires at least one relation (ending with a shared final_relation)")
        final_rel_id1 = rels1[-1]
        final_rel_id2 = rels2[-1]
        if final_rel_id1 != final_rel_id2:
            raise ValueError(
                f"{logic_modle_type} requires the last relation of both paths to be the same, but got {final_rel_id1} and {final_rel_id2}"
            )
        mid_rels1 = rels1[:-1]
        mid_rels2 = rels2[:-1]
        start_name1 = id2name.get(start1, str(start1))
        start_name2 = id2name.get(start2, str(start2))
        mid_path_line1 = " ; ".join(map_rel_ids_to_strings(mid_rels1, id2rel))
        mid_path_line2 = " ; ".join(map_rel_ids_to_strings(mid_rels2, id2rel))
        final_rel_str = map_rel_ids_to_strings([final_rel_id1], id2rel)[0]
        lines = []
        lines.append("Relational Path P1")
        lines.append(f"- Start entity: {start_name1}")
        lines.append(f"- Relational path (to intersection/union): {mid_path_line1}")
        lines.append("Relational Path P2")
        lines.append(f"- Start entity: {start_name2}")
        lines.append(f"- Relational path (to intersection/union): {mid_path_line2}")
        lines.append(f"Final Relation:")
        lines.append(f"- Relation: {final_rel_str}")
        if with_question_placeholder:
            lines.append("Question:")
        return "\n".join(lines)

    elif logic_modle_type in ["chain ni", "2-chain ni"]:
        logic_query = sample.get("logic_query")
        if not logic_query or len(logic_query) < 2:
            raise ValueError(f"Invalid logic_query: {logic_query}")
        branch1, branch2 = logic_query[0], logic_query[1]
        start1, rels1 = branch1[0], branch1[1]
        start2, rels2 = branch2[0], branch2[1]
        start_name1 = id2name.get(start1, str(start1))
        start_name2 = id2name.get(start2, str(start2))
        rel_strs1 = process_rel_ids_with_logic_not(rels1, id2rel)
        rel_strs2 = process_rel_ids_with_logic_not(rels2, id2rel)
        lines = []
        lines.append("Relational Path P1")
        lines.append(f"- Start entity: {start_name1}")
        lines.append(f"- Relations: {' ; '.join(rel_strs1)}")
        lines.append("Relational Path P2")
        lines.append(f"- Start entity: {start_name2}")
        lines.append(f"- Relations: {' ; '.join(rel_strs2)}")
        if with_question_placeholder:
            lines.append("Question:")
        return "\n".join(lines)

    elif logic_modle_type in ["chain nu", "2-chain nu"]:
        logic_query = sample.get("logic_query")
        if not logic_query or len(logic_query) < 2:
            raise ValueError(f"Invalid logic_query: {logic_query}")
        branch1, branch2 = logic_query[0], logic_query[1]
        start1, rels1 = branch1[0], branch1[1]
        start2, rels2 = branch2[0], branch2[1]
        start_name1 = id2name.get(start1, str(start1))
        start_name2 = id2name.get(start2, str(start2))
        rel_strs1 = process_rel_ids_with_logic_not(rels1, id2rel)
        rel_strs2 = process_rel_ids_with_logic_not(rels2, id2rel)
        lines = []
        lines.append("Relational Path P1")
        lines.append(f"- Start entity: {start_name1}")
        lines.append(f"- Relations: {' ; '.join(rel_strs1)}")
        lines.append("Relational Path P2")
        lines.append(f"- Start entity: {start_name2}")
        lines.append(f"- Relations: {' ; '.join(rel_strs2)}")
        if with_question_placeholder:
            lines.append("Question:")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported logic_modle_type: {logic_modle_type}")

def generate_question(prompt, max_retries=5, delay_seconds=3):
    messages = [
        {"role": "system", "content": "You are a professional question-generation assistant."},
        {"role": "user", "content": prompt}
    ]
    response = None
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages
            )
            if response is not None:
                break
            else:
                print(f"Attempt {attempt + 1}: returned None.")
        except (openai.error.Timeout,
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.InvalidRequestError,
                openai.error.PermissionError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Reached maximum retries; unable to get GPT result.")
                raise e
    if not response:
        raise RuntimeError("GPT call failed; no valid response.")
    content = response['choices'][0]['message'].get('content') or "None"
    time.sleep(0.1)
    return content

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def normalize_id(s: str) -> str:
    if s.startswith("m."):
        return "/m/" + s[2:]
    if s.startswith("g."):
        return "/g/" + s[2:]
    return s

def load_ent2id(path: Path) -> dict:
    mapping = {}
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            mid = normalize_id(parts[0])
            try:
                num_id = int(parts[1])
            except ValueError:
                continue
            mapping[mid] = num_id
    return mapping

def load_mid2firstname_filtered(path: Path, needed_mids: set) -> dict:
    mapping = {}
    if not needed_mids:
        return mapping
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            mid = normalize_id(parts[0])
            if mid not in needed_mids:
                continue
            if mid in mapping:
                continue
            name = parts[1]
            mapping[mid] = name
            if len(mapping) == len(needed_mids):
                break
    return mapping

def build_id2name_fast(ent2id_file: Path, mid2name_file: Path, cache_file: Path = None) -> dict:
    if cache_file and cache_file.exists():
        with cache_file.open('rb') as f:
            return pickle.load(f)
    mid_to_numid = load_ent2id(ent2id_file)
    needed = set(mid_to_numid.keys())
    mid_to_firstname = load_mid2firstname_filtered(mid2name_file, needed)
    id2name = {}
    for mid, numid in mid_to_numid.items():
        name = mid_to_firstname.get(mid)
        if name:
            id2name[numid] = name
    if cache_file:
        with cache_file.open('wb') as f:
            pickle.dump(id2name, f, protocol=pickle.HIGHEST_PROTOCOL)
    return id2name

def load_id2rel(path: Path) -> dict:
    mapping = {}
    with Path(path).open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            rel = parts[0]
            try:
                rid = int(parts[1])
            except ValueError:
                continue
            mapping[rid] = rel
    return mapping

if __name__ == "__main__":
    ent2id_path = Path("../../../KG/FB15k/ent2id.txt")
    rel2id_path = Path("../../../KG/FB15k/rel2id.txt")
    mid2name_path = Path("../../../KG/FB15k/ent2id.txt")
    cache_path = Path("../prompts/id2name_cache.pkl")
    data_path = Path(r"./all_queries.json")

    results_path = Path("./generated_questions.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    processed = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "index" in obj:
                        processed.add(obj.get("index"))
                except json.JSONDecodeError:
                    continue

    chain_prompt_path = r"../prompts_fb/prompt_chain_v3.txt"
    chain_union_prompt_path = r"../prompts_fb/prompt_chain_union_v1.txt"
    union_chain_prompt_path = r"../prompts_fb/prompt_chain_union_v1.txt"
    chain_inter_prompt_path = r"../prompts_fb/prompt_chain_inter_v1.txt"
    inter_chain_prompt_path = r"../prompts_fb/prompt_chain_inter_v1.txt"
    # only wn18rr has
    logic_not_path = r"../prompts_wn18rr/prompt_chain_not_inter_v1.txt"
    logic_not_union_path = r"../prompts_wn18rr/prompt_chain_not_union_v1.txt"

    paths = {
        "chain_prompt": chain_prompt_path,
        "chain_union_prompt": chain_union_prompt_path,
        "chain_inter_prompt": chain_inter_prompt_path,
        "union_chain_prompt": union_chain_prompt_path,
        "inter_chain_prompt": inter_chain_prompt_path,
        "logic_not_prompt": logic_not_path,
        "logic_not_union_prompt": logic_not_union_path,
    }

    prompts = {}
    for key, pth in paths.items():
        with open(pth, "r", encoding="utf-8") as f:
            prompts[key] = f.read().strip()

    id2rel = load_id2rel(rel2id_path)
    print(f"rel2id size: {len(id2rel)}")

    id2name_map = build_id2name_fast(ent2id_path, mid2name_path, cache_file=cache_path)
    print(f"id2name size: {len(id2name_map)}")

    with data_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array (list).")
    print(f"Dataset size: {len(dataset)}")

    start_time = time.time()
    for data_index, data in enumerate(dataset):
        if data_index in processed:
            continue

        prompt_block = build_prompt_block_from_sample(
            data, id2name=id2name_map, id2rel=id2rel,
            with_question_placeholder=True
        )

        logic_query = data.get("logic_query")
        logic_type = logic_query[-1]

        prompt_template = ""
        if logic_type in ["2-hop", "3-hop"]:
            prompt_template = prompts.get("chain_prompt")
        elif logic_type in ["chain u", "2-chain u", "2u"]:
            prompt_template = prompts.get("chain_union_prompt")
        elif logic_type in ["chain i", "2-chain i", "2i"]:
            prompt_template = prompts.get("chain_inter_prompt")
        elif logic_type in ["u chain"]:
            prompt_template = prompts.get("union_chain_prompt")
        elif logic_type in ["i chain"]:
            prompt_template = prompts.get("inter_chain_prompt")
        elif logic_type in ["chain ni", "2-chain ni"]:
            prompt_template = prompts.get("logic_not_prompt")
        elif logic_type in ["chain nu", "2-chain nu"]:
            prompt_template = prompts.get("logic_not_union_prompt")

        if not prompt_template:
            raise ValueError(f"No matching prompt template found: {logic_type}")

        cur_prompt = prompt_template + "\n" + prompt_block

        llm_generated = generate_question(cur_prompt)

        record = {
            "index": data_index,
            "logic_query": logic_query,
            "question": llm_generated,
        }

        if (data_index + 1) % 10 == 0:
            print(f"Processed {data_index + 1} samples, elapsed {(time.time() - start_time):.2f}s")

        append_jsonl(results_path, record)
