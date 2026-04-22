#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import argparse
import random
import json
from collections import defaultdict
from pathlib import Path

json_train_default_path = r"../dataset/Mine/FB15k/train.json"
json_vaild_default_path = r"../dataset/Mine/FB15k/valid.json"

bert_path = r"../models/bert/"
parser = argparse.ArgumentParser()
parser.add_argument("--json",        default=json_train_default_path, help="Path to train.json (list of dicts)")
parser.add_argument("--valid-json",  default=json_vaild_default_path, help="Path to vaild.json (same schema). If None, auto-infer from train path.")
parser.add_argument("--epochs",      type=int, default=10)
parser.add_argument("--batch",       type=int, default=32)
parser.add_argument("--val-batch",   type=int, default=16)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--beta",        type=float, default=0.3,   help="weight for InfoNCE")
parser.add_argument("--tau",         type=float, default=0.07,  help="temperature for InfoNCE")
parser.add_argument("--save-every",  type=int, default=2)
parser.add_argument("--out-dir",     default="./results/results_bert_fb15k")
parser.add_argument("--gpu",         type=str,   default="0",    help="CUDA_VISIBLE_DEVICES")
parser.add_argument("--bert-dir",    default=bert_path, help="Local path or HF model identifier for BERT")
args = parser.parse_args()

print("epoch: ", args.epochs)
print("save_every: ", args.save_every)
print("out_dir: ", args.out_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import BertTokenizerFast, BertModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TYPE2LABEL = {
    "no_logic": 0,
    "inter":    1,
    "union":    2,
}

class TextDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{json_path} top-level must be a list")

        questions, labels = [], []
        skipped = 0
        for item in data:
            q = (item.get("question") or "").strip()
            t = (item.get("type") or "").strip()
            if not q or not t or t not in TYPE2LABEL:
                skipped += 1
                continue
            questions.append(q)
            labels.append(TYPE2LABEL[t])

        if skipped > 0:
            print(f"[WARN] {json_path}: skipped {skipped} samples with missing question/type or unknown type.")
        if len(set(labels)) < 2:
            print(f"[WARN] {json_path}: detected fewer than 2 classes; training/evaluation may be unstable.")

        self.questions = questions
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size: int, shuffle: bool = True):
        self.labels  = np.asarray(labels)
        self.bsz     = batch_size
        self.shuffle = shuffle

        self.label2idx = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.label2idx[lab].append(idx)

        self.classes = list(self.label2idx.keys())
        self.n_cls   = len(self.classes)
        assert self.bsz >= self.n_cls, "batch_size must be ≥ number of classes"

        self.num_batches = len(self.labels) // self.bsz

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            for v in self.label2idx.values():
                random.shuffle(v)
        ptr = {c: 0 for c in self.classes}

        for _ in range(self.num_batches):
            base      = self.bsz // self.n_cls
            remainder = self.bsz - base * self.n_cls
            per_class = {c: base for c in self.classes}
            extra_cls = random.sample(self.classes, remainder)
            for c in extra_cls:
                per_class[c] += 1

            batch = []
            for c in self.classes:
                need = per_class[c]
                idx_pool = self.label2idx[c]
                start, end = ptr[c], ptr[c] + need

                if end <= len(idx_pool):
                    selected = idx_pool[start:end]
                    ptr[c] = end
                else:
                    remain = idx_pool[start:]
                    still_need = need - len(remain)
                    refill = random.choices(idx_pool, k=still_need)
                    selected = remain + refill
                    if self.shuffle:
                        random.shuffle(idx_pool)
                    ptr[c] = still_need

                batch.extend(selected)
            if self.shuffle:
                random.shuffle(batch)
            yield batch

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
def evaluate(model, encoder, tokenizer, dataset, batch_size=64, tau=0.07):
    model.eval()
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    all_preds, all_golds = [], []
    total_loss = 0.0
    total_ex   = 0

    for sentences, y in loader:
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = encoder(**inputs)
        embeds = outputs.last_hidden_state[:, 0, :]

        y = torch.tensor(y, dtype=torch.long, device=device)
        logits, hidden = model(embeds)

        loss_ce  = criterion(logits, y)
        loss_nce = info_nce_loss(hidden, y, tau=tau)
        loss     = (1 - args.beta) * loss_ce + args.beta * loss_nce

        total_loss += loss.item() * y.size(0)
        total_ex   += y.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_golds.extend(y.detach().cpu().tolist())

    acc = (np.array(all_preds) == np.array(all_golds)).mean() if total_ex > 0 else 0.0
    macro_f1, micro_f1 = f1_scores(all_preds, all_golds)

    avg_loss = total_loss / max(1, total_ex)
    return {"loss": avg_loss, "acc": float(acc), "macro_f1": macro_f1, "micro_f1": micro_f1}

def f1_scores(preds, golds):
    preds = np.array(preds, dtype=int)
    golds = np.array(golds, dtype=int)
    classes = sorted(set(golds.tolist()) | set(preds.tolist()))
    tp = fp = fn = 0
    f1_list = []
    for c in classes:
        tp_c = np.sum((preds == c) & (golds == c))
        fp_c = np.sum((preds == c) & (golds != c))
        fn_c = np.sum((preds != c) & (golds == c))
        tp += tp_c; fp += fp_c; fn += fn_c
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0
    return float(macro_f1), float(micro_f1)

def train():
    valid_path = args.valid_json
    if valid_path is None:
        p = Path(args.json)
        cand1 = p.with_name("vaild.json")
        cand2 = p.with_name("valid.json")
        valid_path = str(cand1 if cand1.exists() else cand2)

    dataset = TextDataset(args.json)
    val_dataset = TextDataset(valid_path) if valid_path and Path(valid_path).exists() else None
    if val_dataset is None:
        print("[WARN] Validation set not found (vaild.json / valid.json); training only without evaluation.")

    sampler = BalancedBatchSampler(dataset.labels, args.batch, shuffle=True)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    bert_dir = args.bert_dir
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    encoder = BertModel.from_pretrained(bert_dir).to(device)
    encoder.eval()
    embed_dim = encoder.config.hidden_size

    n_classes = len(set(dataset.labels))
    model = MLPClassifier(embed_dim, hidden_dim=128, out_dim=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    best_macro = -1.0
    best_path  = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for step, (sentences, y) in enumerate(loader, 1):
            inputs = tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = encoder(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :]

            y = torch.tensor(y, dtype=torch.long, device=device)
            logits, hidden = model(embeds)

            loss_ce  = criterion(logits, y)
            loss_nce = info_nce_loss(hidden, y, tau=args.tau)
            loss     = (1 - args.beta) * loss_ce + args.beta * loss_nce

            if torch.isnan(loss):
                raise ValueError("Loss became NaN!")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if step % 10 == 0 or step == 1:
                print(f"Epoch[{epoch}/{args.epochs}] Step[{step}/{len(loader)}] "
                      f"CE:{loss_ce.item():.4f}  NCE:{loss_nce.item():.4f}  Total:{loss.item():.4f}")

        if val_dataset is not None:
            val_metrics = evaluate(model, encoder, tokenizer, val_dataset, batch_size=args.val_batch, tau=args.tau)
            print(f"[VAL] Epoch {epoch} | "
                  f"loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.4f}  "
                  f"macroF1={val_metrics['macro_f1']:.4f}  microF1={val_metrics['micro_f1']:.4f}")

        print(f"Epoch {epoch} finished in {time.time() - t0:.1f}s")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "embed_dim": embed_dim,
                "n_classes": n_classes,
                "args": vars(args),
                "type2label": TYPE2LABEL
            }
            path = Path(args.out_dir) / f"mlp_epoch{epoch}.pt"
            torch.save(ckpt, path)
            print(f"[CKPT] Saved to {path.resolve()}")

            if val_dataset is not None:
                macro = val_metrics["macro_f1"]
                if macro > best_macro:
                    best_macro = macro
                    best_path = Path(args.out_dir) / f"best_macroF1.pt"
                    torch.save(ckpt, best_path)
                    print(f"[BEST] New best macroF1={best_macro:.4f} saved to {best_path.resolve()}")

if __name__ == "__main__":
    train()
