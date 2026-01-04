# src/MODELV2/modelv2.py
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------
# Paths / repo root discovery
# -----------------------------

THIS_FILE = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    """
    Walk up until we find a folder that looks like the project root:
    - contains "src" directory
    - and contains "src/skills" directory
    """
    p = start
    for _ in range(12):
        if (p / "src").is_dir() and (p / "src" / "skills").is_dir():
            return p
        p = p.parent
    # fallback: two parents up
    return start.parent.parent


REPO_ROOT = find_repo_root(THIS_FILE.parent)
MODELV2_ROOT = REPO_ROOT / "MODELV2"
SPLITS_DIR = MODELV2_ROOT / "splits_v1"
EXPS_DIR = MODELV2_ROOT / "experiments"
SELECTION_DIR = MODELV2_ROOT / "model_selection"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
EXPS_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading
# -----------------------------

LABEL_MAP_FLOAT = {
    0.0: 0,    # NONE
    0.5: 1,    # IMPLICIT
    1.0: 2,    # EXPLICIT
}


def to_label_id(v: Any) -> int:
    """
    Convert {0,0.5,1} (float/int/str) -> {0,1,2}.
    """
    if v is None:
        return 0
    if isinstance(v, str):
        s = v.strip()
        try:
            v = float(s)
        except Exception:
            return 0
    if isinstance(v, int):
        v = float(v)
    if isinstance(v, float):
        # normalize tiny float issues
        if abs(v - 0.0) < 1e-9:
            return 0
        if abs(v - 0.5) < 1e-9:
            return 1
        if abs(v - 1.0) < 1e-9:
            return 2
    return 0


def load_global_skills() -> List[str]:
    """
    Prefer text list under src/skills/skills_v1.txt (one skill per line).
    """
    candidates = [
        REPO_ROOT / "src" / "skills" / "skills_v1.txt",
        REPO_ROOT / "src" / "skills" / "skills.txt",
    ]
    for p in candidates:
        if p.exists():
            skills = [s.strip() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
            if skills:
                return skills
    raise FileNotFoundError(
        "Global skills list not found. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def find_dataset_jsonl() -> Path:
    """
    Expect your main dataset to be in data/synthetic_dataset.jsonl (as used earlier in your project).
    Each row should look like:
      {"job_description": "...", "skills": {"AWS":1.0, "Docker":0.5, ...}}
    """
    candidates = [
        REPO_ROOT / "data" / "synthetic_dataset.jsonl",
        REPO_ROOT / "data" / "dataset.jsonl",
        REPO_ROOT / "data" / "all.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset JSONL not found. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_examples(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # normalize keys
            text = obj.get("job_description") or obj.get("text") or obj.get("sentence")
            skills = obj.get("skills") or {}
            if not isinstance(skills, dict):
                skills = {}
            if not text:
                continue
            rows.append({"text": text, "skills": skills})
    if not rows:
        raise ValueError(f"No usable rows found in dataset: {dataset_path}")
    return rows


def build_label_vector(
    skills_used: List[str],
    skill_scores: Dict[str, Any],
) -> List[int]:
    """
    Produce a vector length = len(skills_used), values in {0,1,2}.
    Missing => 0.
    """
    out = [0] * len(skills_used)
    for i, s in enumerate(skills_used):
        out[i] = to_label_id(skill_scores.get(s, 0))
    return out


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------
# Dataset for Trainer
# -----------------------------

class OnePassDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        text = r["text"]
        y = r["y"]  # list[int]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # collator will pad
            return_tensors=None,
        )
        enc["labels"] = torch.tensor(y, dtype=torch.long)
        return enc


@dataclass
class PadCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = torch.stack(labels, dim=0)  # [B, num_skills]
        return batch


# -----------------------------
# Model: one forward pass outputs [B, num_skills, 3]
# -----------------------------

class OnePassSkillClassifier(nn.Module):
    def __init__(self, base_name: str, num_skills: int, dropout: float = 0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_name)
        hidden = getattr(self.base.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not infer hidden_size from base model config.")
        self.num_skills = num_skills
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_skills * 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_ignored):
        # IMPORTANT: do NOT forward unknown kwargs to the base model (fixes 'num_items_in_batch' crash)
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token representation
        last_hidden = out.last_hidden_state  # [B, T, H]
        cls = last_hidden[:, 0, :]          # [B, H]
        x = self.dropout(cls)
        logits = self.head(x)               # [B, num_skills*3]
        logits = logits.view(-1, self.num_skills, 3)  # [B, num_skills, 3]

        loss = None
        if labels is not None:
            # labels: [B, num_skills] values in {0,1,2}
            # flatten: treat each (example, skill) as a 3-class classification
            B, S = labels.shape
            flat_logits = logits.reshape(B * S, 3)
            flat_labels = labels.reshape(B * S)
            # guard against any out-of-range labels
            flat_labels = torch.clamp(flat_labels, 0, 2)
            loss = self.loss_fn(flat_logits, flat_labels)

        return {"loss": loss, "logits": logits}


# -----------------------------
# Metrics & reports
# -----------------------------

def compute_metrics(eval_pred):
    """
    eval_pred.predictions: [N, num_skills, 3]
    eval_pred.label_ids:   [N, num_skills]
    We'll flatten over skills for simple global metrics.
    """
    import numpy as np
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    except Exception:
        return {}

    logits = eval_pred.predictions
    y_true = eval_pred.label_ids

    y_pred = logits.argmax(axis=-1)

    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)

    # avoid warnings if a class is missing
    macro_f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    macro_p = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    macro_r = recall_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    acc = accuracy_score(y_true_f, y_pred_f)

    weighted_f1 = f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
    }


def save_classification_report(
    out_path_txt: Path,
    out_path_json: Path,
    y_true_flat: List[int],
    y_pred_flat: List[int],
) -> None:
    try:
        from sklearn.metrics import classification_report
    except Exception:
        out_path_txt.write_text("sklearn not installed, cannot produce classification_report.\n", encoding="utf-8")
        out_path_json.write_text(json.dumps({}, indent=2), encoding="utf-8")
        return

    rep_txt = classification_report(y_true_flat, y_pred_flat, digits=4)
    out_path_txt.parent.mkdir(parents=True, exist_ok=True)
    out_path_txt.write_text(rep_txt, encoding="utf-8")

    rep_dict = classification_report(y_true_flat, y_pred_flat, digits=6, output_dict=True)
    out_path_json.write_text(json.dumps(rep_dict, indent=2), encoding="utf-8")


# -----------------------------
# TrainingArguments compatibility (Transformers v5 vs v4)
# -----------------------------

def make_training_args(**kwargs):
    """
    Transformers v5 uses eval_strategy; older versions used evaluation_strategy.
    We'll try v5 first, fallback to old.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        # fallback: swap eval_strategy <-> evaluation_strategy if needed
        if "evaluation_strategy" in str(e) and "evaluation_strategy" in kwargs:
            fixed = dict(kwargs)
            fixed["eval_strategy"] = fixed.pop("evaluation_strategy")
            return TrainingArguments(**fixed)
        if "eval_strategy" in str(e) and "eval_strategy" in kwargs:
            fixed = dict(kwargs)
            fixed["evaluation_strategy"] = fixed.pop("eval_strategy")
            return TrainingArguments(**fixed)
        raise


# -----------------------------
# Commands
# -----------------------------

def cmd_prepare(args):
    set_seed(args.seed)

    dataset_path = find_dataset_jsonl()
    rows = load_examples(dataset_path)

    global_skills = load_global_skills()

    # choose skills_used
    if args.prefer_extra:
        extra = set()
        for r in rows:
            for s in r["skills"].keys():
                extra.add(s)
        skills_used = sorted(set(global_skills) | extra)
    else:
        skills_used = list(global_skills)

    # build processed rows
    processed = []
    for r in rows:
        y = build_label_vector(skills_used, r["skills"])
        processed.append({"text": r["text"], "y": y})

    rnd = random.Random(args.seed)
    rnd.shuffle(processed)

    n = len(processed)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_rows = processed[:n_train]
    val_rows = processed[n_train:n_train + n_val]
    test_rows = processed[n_train + n_val:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(SPLITS_DIR / "train.jsonl", train_rows)
    save_jsonl(SPLITS_DIR / "val.jsonl", val_rows)
    save_jsonl(SPLITS_DIR / "test.jsonl", test_rows)

    (SPLITS_DIR / "skills_used.txt").write_text("\n".join(skills_used) + "\n", encoding="utf-8")

    print(f"[OK] Splits saved -> {SPLITS_DIR}")
    print(f"[OK] counts: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")


def append_selection_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def cmd_train(args):
    set_seed(args.seed)

    # must exist
    skills_txt = SPLITS_DIR / "skills_used.txt"
    if not skills_txt.exists():
        raise FileNotFoundError(f"Missing skills_used.txt. Run prepare first: {skills_txt}")

    skills_used = [s.strip() for s in skills_txt.read_text(encoding="utf-8").splitlines() if s.strip()]
    num_skills = len(skills_used)

    train_rows = load_jsonl(SPLITS_DIR / "train.jsonl")
    val_rows = load_jsonl(SPLITS_DIR / "val.jsonl")

    exp_name = f"{args.base.replace('/', '__')}__onepass"
    exp_dir = EXPS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    train_ds = OnePassDataset(train_rows, tokenizer, args.max_len)
    val_ds = OnePassDataset(val_rows, tokenizer, args.max_len)

    model = OnePassSkillClassifier(args.base, num_skills=num_skills)

    # IMPORTANT: we force saving and best-model selection
    training_args = make_training_args(
        output_dir=str(exp_dir / "hf_trainer_runs"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",   # wrapper will convert if needed
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    train_out = trainer.train()

    # Save BEST model deterministically to model_best
    best_dir = exp_dir / "model_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save val metrics
    val_metrics = trainer.evaluate()
    rep_dir = exp_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    # Selection table
    row = {
        "exp": exp_name,
        "base": args.base,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "max_len": args.max_len,
        "best_val_macro_f1": float(val_metrics.get("eval_macro_f1", 0.0)),
        "best_val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    append_selection_row(SELECTION_DIR / "results.csv", row)

    print(f"[OK] MODELV2 experiment saved -> {exp_dir}")
    print(f"[OK] Best model -> {best_dir}")
    print(f"[OK] Val metrics -> {rep_dir / 'val_metrics.json'}")
    print(f"[OK] Selection table -> {SELECTION_DIR / 'results.csv'}")


def cmd_eval(args):
    set_seed(42)

    # ensure splits exist
    skills_txt = SPLITS_DIR / "skills_used.txt"
    if not skills_txt.exists():
        raise FileNotFoundError(f"Missing skills_used.txt. Run prepare first: {skills_txt}")

    skills_used = [s.strip() for s in skills_txt.read_text(encoding="utf-8").splitlines() if s.strip()]
    num_skills = len(skills_used)

    exp_dir = EXPS_DIR / args.exp
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    best_dir = exp_dir / "model_best"
    if not best_dir.exists():
        raise FileNotFoundError(
            f"model_best folder not found: {best_dir}\n"
            f"This means training didn't save a model. Run train again."
        )

    # Load tokenizer+model from model_best
    tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=True)

    # IMPORTANT: load base weights from args.base, then load head weights from model_best safetensors/bin
    # Easiest: instantiate the same architecture and load via from_pretrained on AutoModel inside our wrapper.
    # We'll just create model with base=args.base and then load full state_dict via torch.load if present.
    model = OnePassSkillClassifier(args.base, num_skills=num_skills)

    # HuggingFace saved weights in model.safetensors or pytorch_model.bin
    safetensors_path = best_dir / "model.safetensors"
    bin_path = best_dir / "pytorch_model.bin"
    state = None
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as safe_load
            state = safe_load(str(safetensors_path))
        except Exception:
            state = None
    if state is None and bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    if state is None:
        # Trainer.save_model usually writes pytorch_model.bin; but we keep this robust.
        raise FileNotFoundError(f"No model weights found in {best_dir}")

    model.load_state_dict(state, strict=False)

    test_rows = load_jsonl(SPLITS_DIR / "test.jsonl")
    test_ds = OnePassDataset(test_rows, tokenizer, args.max_len)

    eval_args = make_training_args(
        output_dir=str(exp_dir / "eval_runs"),
        per_device_eval_batch_size=args.eval_batch,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=PadCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    pred_out = trainer.predict(test_ds)
    metrics = pred_out.metrics

    # classification report
    import numpy as np
    logits = pred_out.predictions  # [N, S, 3]
    y_pred = np.argmax(logits, axis=-1).reshape(-1).tolist()
    y_true = pred_out.label_ids.reshape(-1).tolist()

    rep_dir = exp_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    save_classification_report(
        rep_dir / "test_report.txt",
        rep_dir / "test_report.json",
        y_true_flat=y_true,
        y_pred_flat=y_pred,
    )

    (rep_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[OK] Test metrics -> {rep_dir / 'test_metrics.json'}")
    print(f"[OK] Test report  -> {rep_dir / 'test_report.txt'}")


def cmd_predict(args):
    """
    Quick toy inference: input text -> predicted vector.
    We will print only skills predicted as IMPLICIT/EXPLICIT (1/2).
    """
    skills_txt = SPLITS_DIR / "skills_used.txt"
    if not skills_txt.exists():
        raise FileNotFoundError(f"Missing skills_used.txt. Run prepare first: {skills_txt}")

    skills_used = [s.strip() for s in skills_txt.read_text(encoding="utf-8").splitlines() if s.strip()]
    num_skills = len(skills_used)

    exp_dir = EXPS_DIR / args.exp
    best_dir = exp_dir / "model_best"
    if not best_dir.exists():
        raise FileNotFoundError(f"model_best folder not found: {best_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=True)

    model = OnePassSkillClassifier(args.base, num_skills=num_skills)
    safetensors_path = best_dir / "model.safetensors"
    bin_path = best_dir / "pytorch_model.bin"
    state = None
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as safe_load
            state = safe_load(str(safetensors_path))
        except Exception:
            state = None
    if state is None and bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    if state is None:
        raise FileNotFoundError(f"No model weights found in {best_dir}")

    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = args.text
    enc = tokenizer(
        text,
        truncation=True,
        max_length=args.max_len,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out["logits"][0]  # [S, 3]
        pred = torch.argmax(logits, dim=-1).tolist()  # [S]

    # decode to your original scores: 0 -> 0, 1 -> 0.5, 2 -> 1
    id_to_score = {0: 0.0, 1: 0.5, 2: 1.0}

    chosen = []
    for s, pid in zip(skills_used, pred):
        if pid in (1, 2):
            chosen.append((s, id_to_score[pid]))

    # sort: explicit first
    chosen.sort(key=lambda x: x[1], reverse=True)

    print("\n=== PREDICTION (only non-zero) ===")
    for s, sc in chosen[: args.topk]:
        print(f"{s}: {sc}")
    print("=== END ===\n")


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--prefer-extra", action="store_true")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser("train")
    st.add_argument("--base", type=str, required=True)
    st.add_argument("--epochs", type=int, default=3)
    st.add_argument("--lr", type=float, default=2e-5)
    st.add_argument("--batch", type=int, default=8)
    st.add_argument("--eval-batch", type=int, default=16)
    st.add_argument("--max-len", type=int, default=384)
    st.add_argument("--seed", type=int, default=42)
    st.set_defaults(func=cmd_train)

    se = sub.add_parser("eval")
    se.add_argument("--exp", type=str, required=True)
    se.add_argument("--base", type=str, required=True)
    se.add_argument("--eval-batch", type=int, default=16)
    se.add_argument("--max-len", type=int, default=384)
    se.set_defaults(func=cmd_eval)

    si = sub.add_parser("predict")
    si.add_argument("--exp", type=str, required=True)
    si.add_argument("--base", type=str, required=True)
    si.add_argument("--text", type=str, required=True)
    si.add_argument("--topk", type=int, default=30)
    si.add_argument("--max-len", type=int, default=384)
    si.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
