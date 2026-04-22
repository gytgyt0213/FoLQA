#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path


def normalize_id(s: str) -> str:
    if s.startswith("m."):
        return "/m/" + s[2:]
    if s.startswith("g."):
        return "/g/" + s[2:]
    return s


def load_ent2id(path: Path) -> dict:
    mapping = {}
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            mid = normalize_id(parts[0])
            try:
                num_id = int(parts[1])
            except ValueError:
                continue
            mapping[mid] = num_id
    return mapping


def load_mid2firstname_filtered(path: Path, needed_mids: set) -> dict:
    mapping = {}
    if not needed_mids:
        return mapping

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            mid = normalize_id(parts[0])
            if mid not in needed_mids:
                continue
            if mid in mapping:
                continue

            mapping[mid] = parts[1]

            if len(mapping) == len(needed_mids):
                break
    return mapping


def build_id2name(ent2id_file: Path, mid2name_file: Path, output_pkl: Path) -> dict:
    mid_to_numid = load_ent2id(ent2id_file)
    needed = set(mid_to_numid.keys())
    mid_to_firstname = load_mid2firstname_filtered(mid2name_file, needed)

    id2name = {}
    for mid, numid in mid_to_numid.items():
        name = mid_to_firstname.get(mid)
        if name:
            id2name[numid] = name

    with output_pkl.open("wb") as f:
        pickle.dump(id2name, f, protocol=pickle.HIGHEST_PROTOCOL)

    return id2name


if __name__ == "__main__":
    ent2id_path = Path("")
    mid2name_path = Path("")
    output_pkl_path = Path("")

    id2name = build_id2name(ent2id_path, mid2name_path, output_pkl_path)

    print(f"pkl file has been saved in: {output_pkl_path}")