from pathlib import Path
import json
import pickle

DATASET_SPLITS = {
    "train": Path(r"../datasets/Mine/FB15k/train.json"),
    "valid": Path("../datasets/Mine/FB15k/vaild.json"),
    "test": Path("../datasets/Mine/FB15k/test.json")
}
output_path = r"./not_classification_data_fb15k"
OUT_DIR = Path(output_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

rel2id_path = r"../KG/FB15k/rel2id.txt"
id2name_pkl_path = r"../create_dataset/FB15k/prompts/id2name_cache.pkl"
rel2id_path = Path(rel2id_path)
id2name_pkl_path = Path(id2name_pkl_path)

def load_id2rel(path: Path) -> dict:
    mapping = {}
    with path.open('r', encoding='utf-8') as f:
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
                mapping[rid] = rel
            except ValueError:
                continue
    return mapping


def load_id2name_from_pkl(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_input(question: str, rel_names: list) -> str:
    parts = ["[CLS]", question.strip()]
    for rel in rel_names:
        parts.append(" " + rel)
    return " ".join(parts)


def extract_clean_relation_ids(rel_ids: list) -> list:
    return [rid for rid in rel_ids if rid != -2]


def path_label(rel_ids: list, idx: int) -> int:
    return int(idx + 1 < len(rel_ids) and rel_ids[idx + 1] == -2)


def process_dataset(split_name: str, data_path: Path, id2rel: dict, output_dir: Path):
    with data_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    results = []
    for sample in raw_data:
        question = sample.get("question", "").strip()
        logic_query = sample.get("logic_query", [])
        relational_paths = sample.get("relational_path", [])

        for path in relational_paths:
            rel_ids = path.get("relation", [])
            clean_rel_ids = extract_clean_relation_ids(rel_ids)

            for i in range(len(clean_rel_ids)):
                rel_segment = clean_rel_ids[:i + 1]
                rel_names = [id2rel.get(rid, f"[REL_{rid}]") for rid in rel_segment]
                label = path_label(rel_ids, i)
                input_text = build_input(question, rel_names)
                results.append({
                    "input": input_text,
                    "label": label
                })

    out_path = output_dir / f"{split_name}_not_classification.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path.name, len(results)

id2rel = load_id2rel(rel2id_path)
id2name = load_id2name_from_pkl(id2name_pkl_path)

output_files = {}
for split, path in DATASET_SPLITS.items():
    filename, count = process_dataset(split, path, id2rel, OUT_DIR)
    output_files[split] = (filename, count)
