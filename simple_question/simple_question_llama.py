import json
import os, csv, warnings, torch, ijson
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests

import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_URL   = f"{OLLAMA_HOST}/api/generate"

print("OLLAMA_HOST: ", OLLAMA_HOST)
print("OLLAMA_MODEL: ", OLLAMA_MODEL)
print("OLLAMA_URL: ", OLLAMA_URL)
print("- "*80)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 3, p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.fc1(x); h = self.act1(h); h = self.drop1(h)
        h = self.fc2(h); h = self.act2(h); h = self.drop2(h)
        logits = self.fc3(h)
        return logits, h


class LogicClassifier:
    def __init__(self, ckpt_path: str, bert_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.embed_dim = ckpt["embed_dim"]
        self.n_classes = ckpt["n_classes"]
        self.type2label = ckpt.get("type2label", {"no_logic": 0, "inter": 1, "union": 2})

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
        self.encoder = BertModel.from_pretrained(bert_dir).to(self.device).eval()
        hidden = self.encoder.config.hidden_size
        if hidden != self.embed_dim:
            print(f"[WARN] BERT hidden_size({hidden}) != ckpt embed_dim({self.embed_dim}). "
                  f"Please ensure the same BERT as used in training.")

        self.model = MLPClassifier(self.embed_dim, hidden_dim=128, out_dim=self.n_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        self.id2relation = {
            int(self.type2label.get("no_logic", 0)): None,
            int(self.type2label.get("inter", 1)): "AND",
            int(self.type2label.get("union", 2)): "OR",
        }

    @torch.no_grad()
    def predict_relation(self, text: str) -> Tuple[Optional[str], bool, float]:
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.encoder(**enc)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        logits, _ = self.model(cls_embed)
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        relation = self.id2relation.get(pred_id, None)
        found = relation is not None
        confidence = float(probs[pred_id].item())
        return relation, found, confidence


def data_load(path: str) -> List[str]:
    path_lower = path.lower()
    questions = []

    if "webqsp" in path_lower:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data.get("Questions", []):
                if "RawQuestion" in obj:
                    questions.append(obj["RawQuestion"])
    elif "cwq" in path_lower:
        with Path(path).open("r", encoding="utf-8") as f:
            for prefix in ("item", "data.item"):
                f.seek(0)
                for obj in ijson.items(f, prefix):
                    if isinstance(obj, dict) and "question" in obj:
                        questions.append(obj["question"])
    elif "grailqa" in path_lower:
        with open(path, "r", encoding="utf-8") as f:
            parser = ijson.items(f, "item")
            for obj in parser:
                if "question" in obj:
                    questions.append(obj["question"])
    else:
        raise ValueError(
            f"Unrecognized dataset type; ensure path contains 'webqsp', 'cwq', or 'grailqa': {path}"
        )

    return questions

def load_ent_id2name_pkl(pkl_path: Path) -> Dict[int, str]:
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{pkl_path} must contain a dict {{int_id: 'Name'}}")
    ent_id2name = {}
    for k, v in raw.items():
        try:
            kid = int(k)
        except Exception:
            continue
        ent_id2name[kid] = str(v)
    if not ent_id2name:
        raise ValueError(f"No names parsed from {pkl_path}")
    return ent_id2name


def load_dataset_with_entity_names(data_path: str, id2name_path: str) -> List[Dict[str, Any]]:
    ent_id2name = load_ent_id2name_pkl(Path(id2name_path))

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    parsed_data = []
    for item in dataset:
        question = item.get("question", "")
        start_entity_ids = item.get("StartEntityIDs", [])
        entity_names = [ent_id2name.get(eid, str(eid)) for eid in start_entity_ids]
        parsed_data.append({
            "question": question,
            "entity_names": entity_names
        })

    return parsed_data

def detect_logic_relation_model(text: str, clf: LogicClassifier) -> Tuple[Optional[str], bool, float]:
    return clf.predict_relation(text)

def call_llm(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "stop": ["end of answer"],
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")

def form_prompt(question, base_prompt, rel):
    cur_question = question.get("question")
    entities_mentioned = question.get("entity_names")
    entity_line = "; ".join(entities_mentioned) if entities_mentioned else ""
    cur_prompt = (
        f"{base_prompt}\n"
        f"Q: {cur_question}\n"
        f"Entities Mentioned: {entity_line}\n"
        f"Logical Operation: {rel}\n"
        f"<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return cur_prompt

def dump_rows(rows: list[dict], path: str):
    if not rows:
        return
    file_exists   = Path(path).exists()
    file_is_empty = (not file_exists) or (Path(path).stat().st_size == 0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERNAMES)
        if file_is_empty:
            writer.writeheader()
        writer.writerows(rows)

if __name__=="__main__":
    CKPT_PATH = "../train_classifier/results/results_bert_fb15k/mlp_epoch20.pt"
    BERT_DIR  = "../models/bert/"

    OR_PROMPT_PATH = "./prompts/or_prompt.txt"
    AND_PROMPT_PATH = "./prompts/and_prompt.txt"
    Ent_id2Name_PATH = r"../create_dataset/FB15k/prompts/id2name_cache.pkl"

    DATA_PATH = "../dataset/Mine/FB15k/test.json"
    OUT_CSV_PATH = "./results/FB15k/test_llma3-8b_outputs.csv"

    CUDA_DEVICE = "1"
    BATCH_SIZE = 10

    print("CKPT_PATH: ", CKPT_PATH)
    print("BERT_DIR: ", BERT_DIR)
    print("OR_PROMPT_PATH: ", OR_PROMPT_PATH)
    print("AND_PROMPT_PATH: ", AND_PROMPT_PATH)
    print("DATA_PATH: ", DATA_PATH)
    print("OUT_CSV_PATH: ", OUT_CSV_PATH)
    print("CUDA_DEVICE: ", CUDA_DEVICE)
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("- "* 80)

    warnings.filterwarnings("ignore")
    HEADERNAMES = ["question", "entity_names", "logical_relation", "model_output"]

    output_dir = Path(OUT_CSV_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    or_prompt = Path(OR_PROMPT_PATH).read_text(encoding="utf-8").rstrip()
    and_prompt = Path(AND_PROMPT_PATH).read_text(encoding="utf-8").rstrip()

    device_str = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() and CUDA_DEVICE != "" else "cpu"
    logic_clf = LogicClassifier(CKPT_PATH, BERT_DIR, device=device_str)

    question_list = load_dataset_with_entity_names(DATA_PATH, Ent_id2Name_PATH)
    rows_buffer = []

    for question in question_list:
        rel, is_logic, conf = detect_logic_relation_model(question.get("question"), logic_clf)
        if is_logic:
            base_prompt = and_prompt if rel == "AND" else or_prompt
            prompt = form_prompt(question, base_prompt, rel)
            model_output = call_llm(prompt)
        else:
            model_output = question.get("question", "")
        rows_buffer.append({
            "question": question.get("question", ""),
            "entity_names": "; ".join(question.get("entity_names", [])),
            "logical_relation": rel if is_logic else "no_logic",
            "model_output": model_output
        })
        if len(rows_buffer) >= BATCH_SIZE:
            dump_rows(rows_buffer, OUT_CSV_PATH)
            rows_buffer.clear()

    dump_rows(rows_buffer, OUT_CSV_PATH)