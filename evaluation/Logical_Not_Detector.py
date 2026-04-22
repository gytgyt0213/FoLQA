from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D), src_key_padding_mask: (B, T), True indicates positions to mask.
        """
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_embed = out[:, 0, :]
        logits = self.classifier(cls_embed)
        return logits


class NegationDetector:
    """
    NOT (logical negation) detector.
    - On init: load tokenizer, BERT encoder, and Transformer classifier head weights.
    - build_input: (question, ordered relation-name list) -> input string.
    - predict: classify a batch of prebuilt input strings, returning ["No"/"Yes", ...].
    """
    def __init__(self,
                 bert_dir: str,
                 clf_weights_path: str,
                 device: str = None,
                 max_length: int = 512,
                 batch_size: int = 32,
                 trans_hidden_dim: int = 256,
                 trans_heads: int = 4,
                 trans_layers: int = 2,
                 trans_dropout: float = 0.1):
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
        self.device = torch.device(device)

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
        self.encoder: BertModel = BertModel.from_pretrained(bert_dir).to(self.device)
        self.encoder.eval()
        embed_dim = self.encoder.config.hidden_size

        self.classifier = TransformerClassifier(
            input_dim=embed_dim,
            hidden_dim=trans_hidden_dim,
            num_heads=trans_heads,
            num_layers=trans_layers,
            dropout=trans_dropout,
            num_classes=2,
        ).to(self.device)
        sd = torch.load(clf_weights_path, map_location=self.device)
        self.classifier.load_state_dict(sd)
        self.classifier.eval()

        self.max_length = max_length
        self.batch_size = batch_size
        self.id2label = {0: "No", 1: "Yes"}

    @staticmethod
    def build_input(question: str, relation_names: List[str]) -> str:
        """
        Construct the classifier input string from "question + ordered relation-name list".
        You may adjust the template as needed, as long as train/inference match.
        """
        rel_block = " ".join(relation_names) if relation_names else " "
        return f"[CLS] {question} {rel_block}"

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[str]:
        """
        Input: a batch of prebuilt strings (usually from build_input).
        Output: same-length list of labels ["No"/"Yes", ...].
        """
        preds_out: List[str] = []
        self.encoder.eval()
        self.classifier.eval()

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.encoder(**enc)
            embeds: torch.Tensor = outputs.last_hidden_state

            src_key_padding_mask = (enc["attention_mask"] == 0)

            logits: torch.Tensor = self.classifier(
                embeds, src_key_padding_mask=src_key_padding_mask
            )
            batch_pred_ids = torch.argmax(logits, dim=-1).tolist()
            preds_out.extend(self.id2label[i_] for i_ in batch_pred_ids)

        return preds_out


class NegationDetector_t5:
    """
    NOT (logical negation) detector (T5 version).
    - Init: load T5 tokenizer/model + Transformer classifier head weights.
    - predict(texts): classify a batch of input strings, returning ["No"/"Yes", ...].
    Notes:
      * To exactly reproduce the original T5 evaluation script: during forward, explicitly pass
        decoder_input_ids = <pad>, take only encoder_last_hidden_state, do not pass a padding
        mask to the classifier head, and use out[:, 0, :] as the sentence vector.
      * Keep the input template identical to training; if training used the raw sentence,
        pass the raw sentence here as well.
    """
    def __init__(self,
                 t5_dir: str,
                 clf_weights_path: str,
                 device: Optional[str] = None,
                 max_length: int = 512,
                 batch_size: int = 32,
                 trans_hidden_dim: int = 256,
                 trans_heads: int = 4,
                 trans_layers: int = 2,
                 trans_dropout: float = 0.1):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            dev = str(device).strip().lower()
            if dev.isdigit():
                device = f"cuda:{dev}" if torch.cuda.is_available() and int(dev) < torch.cuda.device_count() else "cpu"
            elif dev.startswith("cuda:"):
                if torch.cuda.is_available():
                    idx = int(dev.split(":", 1)[1])
                    device = dev if idx < torch.cuda.device_count() else "cpu"
                else:
                    device = "cpu"
            elif dev not in ("cpu", "cuda"):
                device = "cpu"
            else:
                device = "cuda:0" if (dev == "cuda" and torch.cuda.is_available()) else "cpu"

        self.device = torch.device(device)

        self.tokenizer = T5Tokenizer.from_pretrained(t5_dir)
        self.encoder = T5ForConditionalGeneration.from_pretrained(t5_dir).to(self.device).eval()
        embed_dim = self.encoder.config.d_model

        self.classifier = TransformerClassifier(
            input_dim=embed_dim,
            hidden_dim=trans_hidden_dim,
            num_heads=trans_heads,
            num_layers=trans_layers,
            dropout=trans_dropout,
            num_classes=2,
        ).to(self.device)
        sd = torch.load(clf_weights_path, map_location="cpu")
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        self.classifier.load_state_dict(sd, strict=True)
        self.classifier.eval()

        self.max_length = max_length
        self.batch_size = batch_size
        self.id2label = {0: "No", 1: "Yes"}

    @staticmethod
    def build_input(question: str, relation_names: List[str]) -> str:
        """
        Construct the classifier input string from "question + ordered relation-name list".
        You may adjust the template as needed, as long as train/inference match.
        """
        rel_block = " ".join(relation_names) if relation_names else " "
        return f"[CLS] {question} {rel_block}"

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[str]:
        """
        Input: a batch of prebuilt strings (typically the raw sentence).
        Output: same-length list of labels ["No"/"Yes", ...].
        """
        preds_out: List[str] = []
        self.encoder.eval()
        self.classifier.eval()

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            B = enc["input_ids"].size(0)
            pad_id = self.tokenizer.pad_token_id
            decoder_input_ids = torch.full((B, 1), pad_id, dtype=torch.long, device=self.device)

            outputs = self.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                decoder_input_ids=decoder_input_ids
            )
            embeds: torch.Tensor = outputs.encoder_last_hidden_state

            logits: torch.Tensor = self.classifier(embeds)

            batch_pred_ids = torch.argmax(logits, dim=-1).tolist()
            preds_out.extend(self.id2label[i_] for i_ in batch_pred_ids)

        return preds_out
