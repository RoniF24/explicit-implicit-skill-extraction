# src/train/train_transformer.py
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR
from src.datasets.synthetic_jsonl import load_jsonl_dataset  # returns List[(text, labels_vector)]

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class SkillDataset(Dataset):
    """
    HuggingFace Trainer dataset for multi-label classification.
    Each item returns tokenized inputs + float labels vector.
    """
    def __init__(self, pairs: List[Tuple[str, List[float]]], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        text, labels = self.pairs[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.float32)
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/splits/train.jsonl")
    ap.add_argument("--val_path", default="data/splits/val.jsonl")
    ap.add_argument("--model_name", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--out_dir", default="outputs/model_distilroberta")
    args = ap.parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data using the SAME extractor (job_description/resume_chunk_text/text) ---
    train_pairs = load_jsonl_dataset(train_path)
    val_pairs = load_jsonl_dataset(val_path)

    print(f"Loaded train: {len(train_pairs)} val: {len(val_pairs)}")
    print("Num labels (skills):", len(GLOBAL_SKILL_VECTOR))

    if len(train_pairs) == 0:
        raise RuntimeError(
            "Train split loaded 0 samples. Check your JSONL keys; "
            "expected job_description/resume_chunk_text/text + skills."
        )

    # --- Model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(GLOBAL_SKILL_VECTOR),
        problem_type="multi_label_classification",  # important: BCEWithLogitsLoss
    )

    train_ds = SkillDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_ds = SkillDataset(val_pairs, tokenizer, max_length=args.max_length)

    # transformers>=4.57 uses eval_strategy (not evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=False,  # CPU
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,  # ok למרות ה-warning
    )

    trainer.train()

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("Saved model to:", final_dir.resolve())


if __name__ == "__main__":
    main()
