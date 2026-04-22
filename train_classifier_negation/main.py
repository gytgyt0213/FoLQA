#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import random
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--json", default="./not_classification_data_fb15k/train_not_classification.json")
parser.add_argument("--valid-json", default="./not_classification_data_fb15k/valid_not_classification.json")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--val-batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--tau", type=float, default=0.07)
parser.add_argument("--save-every", type=int, default=2)
parser.add_argument("--out-dir", default="./results/fb15k")
parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--bert-dir", default="../models/t5-base")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.asarray(labels)
        self.bsz = batch_size
        self.shuffle = shuffle
        self.label2idx = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.label2idx[lab].append(idx)
        self.classes = list(self.label2idx.keys())
        self.n_cls = len(self.classes)
        assert self.bsz >= self.n_cls
        self.num_batches = len(self.labels) // self.bsz

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            for v in self.label2idx.values():
                random.shuffle(v)
        ptr = {c: 0 for c in self.classes}
        for _ in range(self.num_batches):
            base = self.bsz // self.n_cls
            remainder = self.bsz - base * self.n_cls
            per_class = {c: base for c in self.classes}
            for c in random.sample(self.classes, remainder):
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
                    refill = random.choices(idx_pool, k=need - len(remain))
                    selected = remain + refill
                    if self.shuffle:
                        random.shuffle(idx_pool)
                    ptr[c] = need - len(remain)
                batch.extend(selected)
            if self.shuffle:
                random.shuffle(batch)
            yield batch

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    all_preds, all_golds = [], []
    total_loss = 0.0
    for sentences, y in loader:
        inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        B = inputs['input_ids'].size(0)
        decoder_input_ids = torch.full((B, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
        outputs = encoder(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
        embeds = outputs.encoder_last_hidden_state
        y = torch.tensor(y, dtype=torch.long, device=device)
        logits, hidden = model(embeds)
        loss_ce = criterion(logits, y)
        loss_nce = info_nce_loss(hidden, y, tau)
        loss = (1 - args.beta) * loss_ce + args.beta * loss_nce
        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_golds.extend(y.cpu().tolist())
    acc = (np.array(all_preds) == np.array(all_golds)).mean()
    macro_f1, micro_f1 = f1_scores(all_preds, all_golds)
    return {"loss": total_loss / len(dataset), "acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}


def f1_scores(preds, golds):
    from sklearn.metrics import f1_score
    return f1_score(golds, preds, average='macro'), f1_score(golds, preds, average='micro')


def train():
    dataset = NegationDataset(args.json)
    all_labels = np.array(dataset.labels)
    num_classes = len(set(all_labels))
    cls_counts = np.bincount(all_labels, minlength=num_classes)
    cls_weights = (cls_counts.sum() / (num_classes * (cls_counts + 1e-6)))
    cls_weights = torch.tensor(cls_weights, dtype=torch.float, device=device)
    val_dataset = NegationDataset(args.valid_json) if Path(args.valid_json).exists() else None
    sampler = BalancedBatchSampler(dataset.labels, args.batch, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    tokenizer = T5Tokenizer.from_pretrained('../models/t5-base')
    encoder = T5ForConditionalGeneration.from_pretrained('../models/t5-base').to(device)
    encoder.eval()
    embed_dim = encoder.config.d_model
    model = TransformerClassifier(embed_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_macro = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for step, (sentences, y) in enumerate(loader):
            inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            decoder_input_ids = tokenizer(["<pad>"] * len(sentences), padding=True, truncation=True, max_length=1, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                outputs = encoder(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
                embeds = outputs.encoder_last_hidden_state
            y = torch.tensor(y, dtype=torch.long, device=device)
            logits, hidden = model(embeds)
            loss_ce = criterion(logits, y)
            loss_nce = info_nce_loss(hidden, y, tau=args.tau)
            loss = (1 - args.beta) * loss_ce + args.beta * loss_nce
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step % 10 == 0 or step == 1:
                print(f"[E{epoch}] Step {step}/{len(loader)} | Total Loss={loss.item():.4f} | CE={loss_ce.item():.4f} | NCE={loss_nce.item():.4f} | Time={time.time() - t0:.2f}s")
        if val_dataset:
            metrics = evaluate(model, encoder, tokenizer, val_dataset, batch_size=args.val_batch, tau=args.tau)
            print(f"[VAL] Epoch {epoch} | loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} macroF1={metrics['macro_f1']:.4f} microF1={metrics['micro_f1']:.4f}")
            if metrics['macro_f1'] > best_macro:
                best_macro = metrics['macro_f1']
                best_path = Path(args.out_dir) / "best_macroF1.pt"
                torch.save(model.state_dict(), best_path)
                print(f"[BEST] Saved best model to {best_path.resolve()}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = Path(args.out_dir) / f"transformer_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved checkpoint to {ckpt_path.resolve()}")


if __name__ == "__main__":
    train()
