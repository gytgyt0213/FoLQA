import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, f1_score, accuracy_score

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_stem_for_dir(p: Path):
    stem = p.stem.strip().replace(" ", "_")
    return stem if stem else "unnamed"

model_path = r"./results/fb15k/best_macroF1.pt"
DEFAULT_TEST_JSONS = [
    r"./not_classification_data_fb15k/test_not_classification.json",
]

parser = argparse.ArgumentParser()
parser.add_argument("--test-json", default="./not_classification_data_fb15k/test_not_classification.json",
                    help="Single test JSON path")
parser.add_argument("--test-jsons", nargs="+", default=DEFAULT_TEST_JSONS,
                    help="Multiple test JSON paths (space-separated). If provided, overrides --test-json.")
parser.add_argument("--bert-dir", default="../models/t5-base",
                    help="Path to T5 checkpoint dir")
parser.add_argument("--model-path", default=model_path,
                    help="Path to classifier .pt saved from T5 training")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--max-length", type=int, default=512,
                    help="Must match training's tokenize max_length (training used 512)")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--report-loss", action="store_true", help="Report CE and CE+beta*NCE loss on test set")
parser.add_argument("--beta", type=float, default=0.3, help="Weight for InfoNCE in total loss")
parser.add_argument("--tau", type=float, default=0.07, help="Temperature for InfoNCE")
parser.add_argument("--out-dir", default="./test_results/fb15k/",
                    help="Where to save per-file outputs")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NegationDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.questions = [item["input"] for item in data]
        self.labels = [item["label"] for item in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.transformer(x)
        cls_embed = out[:, 0, :]
        logits = self.classifier(cls_embed)
        return logits, cls_embed

def info_nce_loss(z, labels, tau=0.07):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / tau
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    same = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(z.device) & (~eye)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim) * (~eye)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
    pos = (log_prob * same).sum(dim=1) / (same.sum(dim=1) + 1e-6)
    return -pos.mean()

print("Loading T5 tokenizer & model (encoder-decoder)...")
tokenizer = T5Tokenizer.from_pretrained(args.bert_dir)
encoder = T5ForConditionalGeneration.from_pretrained(args.bert_dir).to(device)
encoder.eval()
embed_dim = encoder.config.d_model
print(f"[INFO] embed_dim={embed_dim}, max_length={args.max_length}")

print("Loading TransformerClassifier (must match training architecture)...")
model = TransformerClassifier(embed_dim).to(device)
state = torch.load(args.model_path, map_location="cpu")
if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()

ce_criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate_one(json_path: str):
    dataset = NegationDataset(json_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    total_ce, total_nce, total_total = 0.0, 0.0, 0.0

    for sentences, labels in loader:
        inputs = tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        B = inputs['input_ids'].size(0)
        decoder_input_ids = torch.full((B, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)

        outputs = encoder(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
        embeds = outputs.encoder_last_hidden_state

        logits, hidden = model(embeds)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels)

        if args.report_loss:
            y = torch.tensor(labels, dtype=torch.long, device=device)
            ce = ce_criterion(logits, y).item()
            nce = info_nce_loss(hidden, y, tau=args.tau).item()
            total_ce += ce * y.size(0)
            total_nce += nce * y.size(0)
            total_total += ((1 - args.beta) * ce + args.beta * nce) * y.size(0)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    pred_counts = np.bincount(all_preds, minlength=2).tolist()
    true_counts = np.bincount(all_labels, minlength=2).tolist()

    result = {
        "n_samples": len(all_labels),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "report_text": report
    }
    if args.report_loss and len(all_labels) > 0:
        n = len(all_labels)
        result["loss"] = {
            "ce": total_ce / n,
            "infonce": total_nce / n,
            "total": total_total / n,
            "beta": args.beta,
            "tau": args.tau
        }
    return result

def main():
    test_files = [Path(p) for p in args.test_jsons] if args.test_jsons else [Path(args.test_json)]
    ensure_dir(Path(args.out_dir))

    for idx, fpath in enumerate(test_files, start=1):
        print("\n" + "="*10 + f" File {idx}/{len(test_files)} " + "="*10)
        print(f"Path: {fpath}")
        if not fpath.exists():
            print(f"[WARN] File not found. Skipping: {fpath}")
            continue

        result = evaluate_one(str(fpath))

        print("\n========== Test Results ==========")
        print(f"Samples    : {result['n_samples']}")
        print(f"Accuracy   : {result['accuracy']:.4f}")
        print(f"Macro F1   : {result['macro_f1']:.4f}")
        print(f"Micro F1   : {result['micro_f1']:.4f}")
        if args.report_loss and "loss" in result:
            print(f"CE Loss    : {result['loss']['ce']:.4f}")
            print(f"InfoNCE    : {result['loss']['infonce']:.4f} (tau={result['loss']['tau']})")
            print(f"Total Loss : {result['loss']['total']:.4f} (beta={result['loss']['beta']})")
        print("\nDetailed Report:")
        print(result["report_text"])
        print("Predicted class distribution:", result["pred_counts"])
        print("True class distribution:", result["true_counts"])

        subdir_name = safe_stem_for_dir(fpath)
        out_subdir = Path(args.out_dir) / subdir_name
        ensure_dir(out_subdir)

        with open(out_subdir / "report.txt", "w", encoding="utf-8") as rf:
            rf.write(result["report_text"])

        effective_args = {
            "test_json": str(fpath),
            "bert_dir": args.bert_dir,
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "gpu": args.gpu,
            "report_loss": args.report_loss,
            "beta": args.beta,
            "tau": args.tau,
        }
        summary = {
            "timestamp": now_iso(),
            "file": str(fpath.resolve()),
            "metrics": {
                "samples": result["n_samples"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "micro_f1": result["micro_f1"],
                "pred_counts": result["pred_counts"],
                "true_counts": result["true_counts"],
            },
            "loss": result.get("loss", None),
            "args": effective_args
        }
        dump_json(summary, out_subdir / "summary.json")

    print("\nDone. Per-file outputs saved under:", Path(args.out_dir).resolve())

if __name__ == "__main__":
    main()
