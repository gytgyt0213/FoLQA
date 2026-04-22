#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Set

import igraph as ig


def load_rel_id2name(rel2id_path: Path) -> Dict[int, str]:
    rel_id2name: Dict[int, str] = {}
    with rel2id_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rel_str = parts[0]
            try:
                rid = int(parts[1])
            except ValueError:
                continue
            rel_id2name[rid] = rel_str
    if not rel_id2name:
        raise ValueError(f"No relations parsed from {rel2id_path}")
    return rel_id2name


def load_ent_id2name_pkl(pkl_path: Path) -> Dict[int, str]:
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{pkl_path} must contain a dict {{int_id: 'Name'}}")
    ent_id2name: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            kid = int(k)
        except Exception:
            continue
        ent_id2name[kid] = str(v)
    if not ent_id2name:
        raise ValueError(f"No names parsed from {pkl_path}")
    return ent_id2name


def load_triples_numeric(
    kg_path: Path,
    limit: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with kg_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if limit is not None and len(triples) >= limit:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                hid = int(parts[0]); rid = int(parts[1]); tid = int(parts[2])
            except ValueError:
                continue
            triples.append((hid, rid, tid))
    if not triples:
        raise ValueError(f"No triples parsed from {kg_path}")
    return triples


def build_graph(
    triples: Sequence[Tuple[int, int, int]],
    *,
    directed: bool = True
) -> Tuple[ig.Graph, Dict[int, int]]:
    entities: Set[int] = set()
    for h, _, t in triples:
        entities.add(h); entities.add(t)

    ent_sorted = sorted(entities)
    id2idx: Dict[int, int] = {eid: i for i, eid in enumerate(ent_sorted)}

    g = ig.Graph(directed=directed)
    g.add_vertices(len(id2idx))
    g.vs["id"] = [None] * len(id2idx)
    for eid, idx in id2idx.items():
        g.vs[idx]["id"] = eid

    g.add_edges([(id2idx[h], id2idx[t]) for h, _, t in triples])
    g.es["relation"] = [rid for _, rid, _ in triples]
    return g, id2idx


class KGResources:

    def __init__(
        self,
        kg_txt: str,
        rel2id_txt: str,
        id2name_pkl: str,
        triple_limit: Optional[int] = None,
    ):
        kg_path = Path(kg_txt)
        rel2id_path = Path(rel2id_txt)
        id2name_path = Path(id2name_pkl)

        print("[KG] Loading relation mapping …")
        self.rel_id2name: Dict[int, str] = load_rel_id2name(rel2id_path)
        print(f"      → {len(self.rel_id2name):,} relations")

        print("[KG] Loading triples …")
        triples = load_triples_numeric(kg_path, limit=triple_limit)
        print(f"      → {len(triples):,} triples")

        print("[KG] Building graph …")
        self.graph, self.id2idx = build_graph(triples)
        self.idx2id = {idx: eid for eid, idx in self.id2idx.items()}
        print(f"      → {self.graph.vcount():,} nodes / {self.graph.ecount():,} edges")

        print("[KG] Loading id→name (pkl) …")
        self.ent_id2name: Dict[int, str] = load_ent_id2name_pkl(id2name_path)
        print(f"      → {len(self.ent_id2name):,} ids with names")

    def ent_name(self, eid: int) -> str:
        return self.ent_id2name.get(eid, str(eid))

    def rel_name(self, rid: int) -> str:
        return self.rel_id2name.get(rid, f"[REL_{rid}]")

    def one_hop(self, eid: int) -> List[Tuple[int, int]]:
        if eid not in self.id2idx:
            return []
        idx = self.id2idx[eid]
        out: List[Tuple[int, int]] = []
        for e_idx in self.graph.incident(idx, mode="OUT"):
            edge = self.graph.es[e_idx]
            rid: int = edge["relation"]
            tgt_idx: int = edge.target
            nbr_eid: int = self.idx2id[tgt_idx]
            out.append((rid, nbr_eid))
        return out

    def logical_not(self, entities: Set[int]) -> Set[int]:
        universe: Set[int] = set(self.ent_id2name.keys())
        return universe - set(entities)


def pick_random_entity_with_outdegree(g: ig.Graph, seed: int = 42) -> Optional[int]:

    rng = random.Random(seed)
    candidates = [v.index for v in g.vs if g.degree(v, mode="OUT") > 0]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    return g.vs[idx]["id"]
