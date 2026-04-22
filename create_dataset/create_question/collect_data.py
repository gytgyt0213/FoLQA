import os
import json
from collections import defaultdict
from typing import Dict, Set, List, Any

KB_FILE_PATH = '../../KG/FB15k/kb.txt'
path = r"./transformed_answers/2p.json"
FILE_LIST = [
    './transformed_answers/2p.json',
    './transformed_answers/3p.json',
    './transformed_answers/pi.json',
    './transformed_answers/ip.json',
    './transformed_answers/2pi.json',
    './transformed_answers/pu.json',
    './transformed_answers/up.json',
    './transformed_answers/2pu.json',
    './transformed_answers/pin.json',
    './transformed_answers/2in.json',
]
OUTPUT_DIR = './filter'
MAX_COMBINED_SIZE = 20


def load_kg(kb_path: str):
    graph: Dict[int, Dict[int, Set[int]]] = {}
    entity_universe: Set[int] = set()

    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            s, r, o = map(int, line.strip().split())
            graph.setdefault(s, {}).setdefault(r, set()).add(o)
            entity_universe.add(s)
            entity_universe.add(o)

    return graph, entity_universe


def traverse_normal(graph: Dict[int, Dict[int, Set[int]]], start: int, path: List[int]) -> Set[int]:
    current = {start}
    for rel in path:
        next_set = set()
        for ent in current:
            next_set |= graph.get(ent, {}).get(rel, set())
        current = next_set
        if not current:
            break
    return current


def traverse(graph: Dict[int, Dict[int, Set[int]]], start: int, path: List[int], entity_universe: Set[int]) -> Set[int]:
    if path and path[-1] == -2:
        actual_path = path[:-1]
        reached = traverse_normal(graph, start, actual_path)
        return entity_universe - reached
    else:
        return traverse_normal(graph, start, path)


def filter_records(data: List[Dict[str, Any]], graph, entity_universe) -> List[Dict[str, Any]]:
    filtered = []
    for rec in data:
        branches = rec.get('relational_path', [])
        op_type = rec.get('type')
        expected = set(rec.get('answer', []))

        branch_sets = []
        ok = True
        for br in branches:
            start = br['start']
            rels = br['relation']
            reached = traverse(graph, start, rels, entity_universe)
            if not reached:
                ok = False
                break
            branch_sets.append(reached)
        if not ok:
            continue

        if op_type == 'inter':
            combined = set.intersection(*branch_sets) if branch_sets else set()
        elif op_type == 'union':
            combined = set.union(*branch_sets) if branch_sets else set()
        else:
            combined = branch_sets[0] if branch_sets else set()

        if expected == combined and len(combined) <= MAX_COMBINED_SIZE and len(expected) > 0:
            filtered.append(rec)
    return filtered


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graph, entity_universe = load_kg(KB_FILE_PATH)

    for in_path in FILE_LIST:
        if not in_path.endswith('.json') or not os.path.isfile(in_path):
            print(f"Skip invalid file: {in_path}")
            continue

        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip file that cannot be read or parsed {in_path}: {e}")
            continue

        records = data if isinstance(data, list) else [data]
        filtered = filter_records(records, graph, entity_universe)

        fname = os.path.basename(in_path)
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"{fname}: original {len(records)} items, after filtering {len(filtered)} items -> {out_path}")


if __name__ == '__main__':
    main()
