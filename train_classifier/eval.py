import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel

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

class TextDataset(Dataset):
    def __init__(self, json_path: str, type2label: dict):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{json_path} top-level must be a list")

        questions, labels = [], []
        skipped = 0
        for item in data:
            q = (item.get("question") or "").strip()
            t = (item.get("type") or "").strip()
            if not q or not t or t not in type2label:
                skipped += 1
                continue
            questions.append(q)
            labels.append(type2label[t])

        if skipped > 0:
            print(f"[WARN] {json_path}: skipped {skipped} samples with missing/unknown type.")
        self.questions = questions
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

def per_class_prf1(preds, golds, classes):
    report = {}
    for c in classes:
        tp = np.sum((preds == c) & (golds == c))
        fp = np.sum((preds == c) & (golds != c))
        fn = np.sum((preds != c) & (golds == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        report[int(c)] = {"precision": float(p), "recall": float(r), "f1": float(f1), "support": int(np.sum(golds == c))}
    return report

def macro_micro_f1(preds, golds):
    preds = np.asarray(preds, dtype=int)
    golds = np.asarray(golds, dtype=int)
    classes = sorted(set(golds.tolist()) | set(preds.tolist()))
    f1s = []
    tp = fp = fn = 0
    for c in classes:
        tp_c = np.sum((preds == c) & (golds == c))
        fp_c = np.sum((preds == c) & (golds != c))
        fn_c = np.sum((preds != c) & (golds == c))
        tp += tp_c; fp += fp_c; fn += fn_c
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        f1s.append(f1)
    macro = float(np.mean(f1s)) if f1s else 0.0
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0
    return macro, micro

def info_nce_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / tau
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(z.device)
    mask_self = torch.eye(len(z), dtype=torch.bool, device=z.device)
    mask = mask & ~mask_self
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim) * (~mask_self)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
    mean_log_pos = (log_prob * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    return -mean_log_pos.mean()

@torch.no_grad()
def evaluate(model, encoder, tokenizer, dataset, batch_size=64, beta=0.3, tau=0.07, device="cuda"):
    model.eval(); encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ex = 0
    all_preds, all_golds = [], []

    for sentences, y in loader:
        inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = encoder(**inputs)
        embeds = outputs.last_hidden_state[:, 0, :]

        y = torch.tensor(y, dtype=torch.long, device=device)
        logits, hidden = model(embeds)

        loss = (1 - beta) * ce(logits, y) + beta * info_nce_loss(hidden, y, tau=tau)
        total_loss += loss.item() * y.size(0)
        total_ex   += y.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_golds.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(1, total_ex)
    preds_np = np.array(all_preds, dtype=int)
    golds_np = np.array(all_golds, dtype=int)

    acc = float((preds_np == golds_np).mean()) if total_ex > 0 else 0.0
    macro_f1, micro_f1 = macro_micro_f1(preds_np, golds_np)
    classes = sorted(set(golds_np.tolist()) | set(preds_np.tolist()))
    cls_report = per_class_prf1(preds_np, golds_np, classes)

    conf = np.zeros((len(classes), len(classes)), dtype=int)
    c2i = {c:i for i,c in enumerate(classes)}
    for p, g in zip(preds_np, golds_np):
        conf[c2i[g], c2i[p]] += 1

    return {
        "loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1,
        "per_class": cls_report,
        "classes": classes, "confusion_matrix": conf.tolist(),
        "pred_dist": dict(Counter(all_preds)), "gold_dist": dict(Counter(all_golds)),
    }

def main():
    ckpt_path = r"./results/results_bert_fb15k/best_macroF1.pt"
    train_json_path = r"../dataset/Mine/FB15k/train.json"
    test_json_path = r"../dataset/Mine/FB15k/test.json"
    bert_path = r"../models/bert/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default=test_json_path, help="Path to test.json (same schema as train).")
    parser.add_argument("--train-json", default=train_json_path, help="Path to train.json (used to auto-infer test path)")
    parser.add_argument("--ckpt", default=ckpt_path, help="Path to checkpoint (mlp_epoch*.pt or best_macroF1.pt)")
    parser.add_argument("--bert-dir", default=bert_path, help="Override BERT dir. Default: use ckpt args['bert_dir']")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out-dir", default="./test_results_wn18rr")
    parser.add_argument("--beta", type=float, default=None, help="Override beta for eval. Default: use ckpt args")
    parser.add_argument("--tau",  type=float, default=None, help="Override tau for eval. Default: use ckpt args")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = args.test_json
    if test_path is None:
        p = Path(args.train_json)
        test_path = str(p.with_name("test.json"))
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    embed_dim = ckpt["embed_dim"]
    n_classes = ckpt["n_classes"]
    ckpt_args = ckpt.get("args", {})
    type2label = ckpt.get("type2label", {"no_logic":0, "inter":1, "union":2})

    beta = args.beta if args.beta is not None else float(ckpt_args.get("beta", 0.3))
    tau  = args.tau  if args.tau  is not None else float(ckpt_args.get("tau", 0.07))
    bert_dir = args.bert_dir or ckpt_args.get("bert_dir", "bert-base-uncased")

    test_ds = TextDataset(test_path, type2label)

    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    encoder   = BertModel.from_pretrained(bert_dir).to(device)
    encoder.eval()
    hidden = encoder.config.hidden_size
    if hidden != embed_dim:
        print(f"[WARN] BERT hidden_size({hidden}) and ckpt embed_dim({embed_dim}) mismatch; please ensure the same BERT is used as in training.")

    model = MLPClassifier(embed_dim, hidden_dim=128, out_dim=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    metrics = evaluate(model, encoder, tokenizer, test_ds, batch_size=args.batch, beta=beta, tau=tau, device=device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_dir) / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Test metrics saved to {out_json.resolve()}")
    print(f"loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}  "
          f"macroF1={metrics['macro_f1']:.4f}  microF1={metrics['micro_f1']:.4f}")

if __name__ == "__main__":
    main()
