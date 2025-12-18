# src/datasets/make_splits.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input JSONL (e.g., data/synthetic_dataset.jsonl)")
    ap.add_argument("--out_dir", required=True, help="Output dir (e.g., data/splits)")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_shuffle", action="store_true", help="Disable shuffling")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    rows = read_jsonl(in_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"No rows found in: {in_path}")

    # Validate ratios (allow tiny float noise)
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Got sum={s}")

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    # Compute exact counts (ensure sum == n)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"Wrote splits to: {out_dir.resolve()}")
    print(f"total: {n}")
    print(f"train: {len(train_rows)}  val: {len(val_rows)}  test: {len(test_rows)}")


if __name__ == "__main__":
    main()
