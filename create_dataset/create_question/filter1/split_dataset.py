import json
import os
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def get_logic_type(sample: Dict) -> str:
    lq = sample.get("logic_query", None)
    if not isinstance(lq, list) or len(lq) < 2 or not isinstance(lq[-1], str):
        raise ValueError(f"Invalid logic_query format: {lq}")
    return lq[-1]

def split_indices(n: int, ratios: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:
    r_train, r_valid, r_test = ratios
    assert abs(r_train + r_valid + r_test - 1.0) < 1e-8, "sum of ratios must be 1"

    n_train = int(n * r_train)
    n_valid = int(n * r_valid)
    n_test  = n - n_train - n_valid

    idxs = list(range(n))
    random.shuffle(idxs)
    return (
        idxs[:n_train],
        idxs[n_train:n_train + n_valid],
        idxs[n_train + n_valid:]
    )

def stratified_split_by_type(
    samples: List[Dict],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)

    buckets = defaultdict(list)
    for s in samples:
        t = get_logic_type(s)
        buckets[t].append(s)

    train_all, valid_all, test_all = [], [], []
    print("Counts by type:")
    for t, group in buckets.items():
        n = len(group)
        train_idx, valid_idx, test_idx = split_indices(n, ratios)

        train_part = [group[i] for i in train_idx]
        valid_part = [group[i] for i in valid_idx]
        test_part  = [group[i] for i in test_idx]

        train_all.extend(train_part)
        valid_all.extend(valid_part)
        test_all.extend(test_part)

        print(f"  - {t:>12s}: total={n:5d} | train={len(train_part):5d}, valid={len(valid_part):5d}, test={len(test_part):5d}")

    print("\nSummary:")
    print(f"  train: {len(train_all)}")
    print(f"  valid: {len(valid_all)}")
    print(f"  test : {len(test_all)}")

    return train_all, valid_all, test_all

def main():
    input_path = r"./merged_queries.json"
    parser = argparse.ArgumentParser(description="Group by logic_query type and split into train/valid/test with ratios 8:1:1")
    parser.add_argument("--input", default=input_path, help="Path to input JSON file (top level is a list; each item is a sample dict)")
    parser.add_argument("--out_dir", default="./all_type_queries/", help="Output directory (default: current directory)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train_file", default="train.json", help="Output filename for training set")
    parser.add_argument("--valid_file", default="valid.json", help="Output filename for validation set")
    parser.add_argument("--test_file",  default="test.json",  help="Output filename for test set")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("Top level of input JSON must be a list.")

    train_all, valid_all, test_all = stratified_split_by_type(samples, (0.8, 0.1, 0.1), args.seed)

    out_train = os.path.join(args.out_dir, args.train_file)
    out_valid = os.path.join(args.out_dir, args.valid_file)
    out_test  = os.path.join(args.out_dir, args.test_file)

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_all, f, ensure_ascii=False, indent=2)
    with open(out_valid, "w", encoding="utf-8") as f:
        json.dump(valid_all, f, ensure_ascii=False, indent=2)
    with open(out_test,  "w", encoding="utf-8") as f:
        json.dump(test_all,  f, ensure_ascii=False, indent=2)

    print("\nWrite complete:")
    print(f"  {out_train}")
    print(f"  {out_valid}")
    print(f"  {out_test}")

if __name__ == "__main__":
    main()
