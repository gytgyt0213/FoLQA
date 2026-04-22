import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_json(fp: Path) -> List[Dict[str, Any]]:
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{fp} does not contain a top-level list")
    return data

def dump_json(obj: Any, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_qtype(rec: Dict[str, Any]) -> str:
    logic_query = rec.get("logic_query", None)
    if isinstance(logic_query, list) and len(logic_query) >= 1:
        last = logic_query[-1]
        if isinstance(last, str):
            return last
    return "__UNKNOWN__"

def process_one_file(
    fp: Path, n_per_type: int, rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = load_json(fp)
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, rec in enumerate(data):
        qtype = get_qtype(rec)
        buckets.setdefault(qtype, []).append((i, rec))
    chosen_indices = set()
    sampled_records: List[Dict[str, Any]] = []
    for qtype, items in buckets.items():
        idxs = [idx for idx, _ in items]
        k = min(n_per_type, len(items))
        if k == 0:
            continue
        chosen_local = set(rng.sample(idxs, k))
        chosen_indices.update(chosen_local)
        for idx, rec in items:
            if idx in chosen_local:
                sampled_records.append(rec)
    remain_records = [rec for i, rec in enumerate(data) if i not in chosen_indices]
    return sampled_records, remain_records

def main(src_dir: Path, dst_dir: Path, n_per_type: int, seed: int):
    rng = random.Random(seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_queries_accum: List[Dict[str, Any]] = []
    file_list = sorted([p for p in src_dir.glob("*.json") if p.is_file()])
    if not file_list:
        print(f"[WARN] No .json files found in {src_dir}")
        return
    for i, fp in enumerate(file_list, 1):
        try:
            sampled, remain = process_one_file(fp, n_per_type, rng)
        except Exception as e:
            print(f"[WARN] Failed to process {fp.name}: {e} (skipped)")
            continue
        all_queries_accum.extend(sampled)
        out_fp = dst_dir / fp.name
        dump_json(remain, out_fp)
        print(
            f"[{i}/{len(file_list)}] {fp.name}: sampled {len(sampled)} items, remaining {len(remain)} -> {out_fp.name}"
        )
    all_queries_fp = dst_dir / "all_queries.json"
    dump_json(all_queries_accum, all_queries_fp)
    print(f"[OK] Wrote summary: {all_queries_fp} (total {len(all_queries_accum)} items)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-file sampling by type into global all_queries, and write each file's remaining records to the destination directory with the same filename"
    )
    parser.add_argument("--src", type=Path, default=Path("../filter"), help="Source directory")
    parser.add_argument("--dst", type=Path, default=Path("../filter1"), help="Destination directory")
    parser.add_argument("--n", type=int, default=1000, help="Samples per type (within each file)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.src, args.dst, args.n, args.seed)
