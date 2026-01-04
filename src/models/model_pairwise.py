from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_recall_fscore_support

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- IO ----------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "job_description" not in obj or "skills" not in obj:
                raise ValueError(f"Bad schema in {path} line {i}: needs job_description + skills")
            items.append(obj)
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_skills_txt(path: Path) -> List[str]:
    skills: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            skills.append(s)
    if not skills:
        raise ValueError(f"Skills file is empty: {path}")
    return skills


def fingerprint_example(ex: Dict[str, Any]) -> str:
    text = ex["job_description"].strip()
    skills = ex.get("skills", {})
    skills_items = sorted((k, float(v)) for k, v in skills.items())
    payload = json.dumps({"t": text, "s": skills_items}, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# ---------- Split 70/15/15 (use metadata only for better split) ----------
def strat_key(ex: Dict[str, Any]) -> str:
    domain = str(ex.get("domain", "NA"))
    seniority = str(ex.get("seniority", "NA"))
    return f"{domain}__{seniority}"


def split_701515(items: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(items)
    if n < 50:
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train = [items[i] for i in idx[:n_train]]
        val = [items[i] for i in idx[n_train:n_train + n_val]]
        test = [items[i] for i in idx[n_train + n_val:]]
        return train, val, test

    y = np.array([strat_key(x) for x in items])

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), y))

    trainval = [items[i] for i in trainval_idx]
    test = [items[i] for i in test_idx]

    y_tv = y[trainval_idx]
    val_frac = 0.15 / 0.85
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    tr_idx, val_idx = next(sss2.split(np.zeros(len(trainval)), y_tv))

    train = [trainval[i] for i in tr_idx]
    val = [trainval[i] for i in val_idx]
    return train, val, test


# ---------- Pairwise rows ----------
# label mapping: 0=NONE, 1=IMPLICIT, 2=EXPLICIT
def score_to_label(v: float) -> int:
    if v >= 0.99:
        return 2
    if v >= 0.49:
        return 1
    return 0


def build_pairs(
    examples: List[Dict[str, Any]],
    global_skills: List[str],
    neg_ratio: float,
    seed: int,
    min_negs_per_example: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    gset = set(global_skills)

    rows: List[Dict[str, Any]] = []
    for ex in examples:
        text = ex["job_description"]
        sk: Dict[str, float] = ex.get("skills", {})

        positives: List[Tuple[str, int]] = []
        present = set()
        for skill, val in sk.items():
            if skill not in gset:
                continue
            lab = score_to_label(float(val))
            if lab == 0:
                continue
            positives.append((skill, lab))
            present.add(skill)

        neg_candidates = [s for s in global_skills if s not in present]
        n_pos = len(positives)
        n_negs = max(min_negs_per_example, int(round(max(1, n_pos) * neg_ratio)))
        n_negs = min(n_negs, len(neg_candidates))
        negs = rng.sample(neg_candidates, k=n_negs) if n_negs > 0 else []

        for skill, lab in positives:
            rows.append({"skill": skill, "job_description": text, "label": lab})
        for skill in negs:
            rows.append({"skill": skill, "job_description": text, "label": 0})

    rng.shuffle(rows)
    return rows


@dataclass
class PairDataset(torch.utils.data.Dataset):
    encodings: Dict[str, Any]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_pairs(tokenizer, rows: List[Dict[str, Any]], max_length: int) -> PairDataset:
    a = [r["skill"] for r in rows]
    b = [r["job_description"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    enc = tokenizer(a, b, truncation=True, max_length=max_length)
    return PairDataset(encodings=enc, labels=labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "macro_f1": float(f1), "macro_precision": float(p), "macro_recall": float(r)}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_results_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model_name", "output_dir", "best_val_macro_f1", "epochs", "lr", "batch_size", "max_length", "cuda", "gpu"]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="SkillSight - Model Creation (split + train + eval)")
    ap.add_argument("--data", default="data/synthetic_dataset_extra.jsonl", help="Use EXTRA for better split/debug (metadata not used as model input).")
    ap.add_argument("--skills-file", default="src/skills/skills_v1.txt")

    ap.add_argument("--splits-dir", default="data/splits_v1")
    ap.add_argument("--seed", type=int, default=42)

    # actions
    ap.add_argument("--prepare", action="store_true", help="Create 70/15/15 splits.")
    ap.add_argument("--train", action="store_true", help="Train a transformer model.")
    ap.add_argument("--eval", action="store_true", help="Evaluate on TEST (pair-level report).")

    # model/training
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", default="models/roberta_base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--min-negs", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=32)

    args = ap.parse_args()

    data_path = REPO_ROOT / args.data
    skills_path = REPO_ROOT / args.skills_file
    splits_dir = REPO_ROOT / args.splits_dir
    out_dir = REPO_ROOT / args.out

    global_skills = read_skills_txt(skills_path)
    global_set = set(global_skills)

    if args.prepare:
        items = read_jsonl(data_path)

        # validate
        unknown = set()
        for ex in items:
            for k in ex.get("skills", {}).keys():
                if k not in global_set:
                    unknown.add(k)
        if unknown:
            raise ValueError(f"Unknown skills found (not in skills_v1.txt): {sorted(list(unknown))[:20]}")

        # dedupe
        seen = set()
        deduped = []
        for ex in items:
            fp = fingerprint_example(ex)
            if fp in seen:
                continue
            seen.add(fp)
            deduped.append(ex)

        train, val, test = split_701515(deduped, seed=args.seed)
        write_jsonl(splits_dir / "train.jsonl", train)
        write_jsonl(splits_dir / "val.jsonl", val)
        write_jsonl(splits_dir / "test.jsonl", test)

        print(f"[OK] Splits saved -> {splits_dir} | train={len(train)} val={len(val)} test={len(test)}")

    if args.train:
        train_path = splits_dir / "train.jsonl"
        val_path = splits_dir / "val.jsonl"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Splits not found. Run: python src/model_creation.py --prepare")

        train_ex = read_jsonl(train_path)
        val_ex = read_jsonl(val_path)

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        train_rows = build_pairs(train_ex, global_skills, args.neg_ratio, args.seed, args.min_negs)
        val_rows = build_pairs(val_ex, global_skills, args.neg_ratio, args.seed + 1, args.min_negs)

        print(f"[INFO] Pair rows: train={len(train_rows)} val={len(val_rows)}")

        train_ds = tokenize_pairs(tokenizer, train_rows, args.max_len)
        val_ds = tokenize_pairs(tokenizer, val_rows, args.max_len)

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        use_cuda = torch.cuda.is_available()

        targs = TrainingArguments(
            output_dir=str(out_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.eval_batch,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=use_cuda,
            report_to="none",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        summary = {
            "model": args.model,
            "out_dir": str(out_dir),
            "best_val_macro_f1": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "max_len": args.max_len,
            "cuda": bool(use_cuda),
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        }

        save_json(out_dir / "metrics" / "summary.json", summary)

        results_csv = REPO_ROOT / "outputs" / "model_selection" / "results.csv"
        append_results_csv(
            results_csv,
            {
                "model_name": args.model,
                "output_dir": str(out_dir),
                "best_val_macro_f1": trainer.state.best_metric,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch,
                "max_length": args.max_len,
                "cuda": bool(use_cuda),
                "gpu": torch.cuda.get_device_name(0) if use_cuda else "",
            },
        )

        print(f"[OK] Saved model -> {out_dir}")
        print(f"[OK] Saved metrics -> {out_dir / 'metrics' / 'summary.json'}")
        print(f"[OK] Selection table -> {results_csv}")

    if args.eval:
        test_path = splits_dir / "test.jsonl"
        if not test_path.exists():
            raise FileNotFoundError("Test split not found. Run --prepare first.")

        # load trained model + tokenizer from output_dir
        model = AutoModelForSequenceClassification.from_pretrained(str(out_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(out_dir))  # אפשר להשאיר ככה

        test_ex = read_jsonl(test_path)
        test_rows = build_pairs(test_ex, global_skills, args.neg_ratio, args.seed + 2, args.min_negs)
        test_ds = tokenize_pairs(tokenizer, test_rows, args.max_len)

        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Trainer for prediction (handles padding correctly)
        eval_args = TrainingArguments(
            output_dir=str(out_dir / "_eval_tmp"),
            per_device_eval_batch_size=args.eval_batch,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            data_collator=collator,
        )

        pred_out = trainer.predict(test_ds)
        logits = pred_out.predictions
        labels = pred_out.label_ids
        preds = np.argmax(logits, axis=1)

        report = classification_report(labels, preds, digits=4)
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics" / "test_report.txt").write_text(report, encoding="utf-8")

        print(f"[OK] Test report saved -> {out_dir / 'metrics' / 'test_report.txt'}")



if __name__ == "__main__":
    main()
