import json
from collections import OrderedDict

all_queries_path = "all_queries.json"
generated_questions_path = "generated_questions.jsonl"
output_path = "merged_queries.json"
mismatch_log_path = "mismatch_log.json"

def unique_preserve_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

with open(all_queries_path, "r", encoding="utf-8") as f:
    all_queries_list = json.load(f)

with open(generated_questions_path, "r", encoding="utf-8") as f:
    questions_list = [json.loads(line) for line in f]

len_all = len(all_queries_list)
len_q = len(questions_list)
if len_all != len_q:
    print(f"[WARN] Two files have different lengths: all_queries={len_all}, questions={len_q}")

merged_list = []
mismatches = []

for idx, (aq, q) in enumerate(zip(all_queries_list, questions_list)):
    if aq.get("logic_query") == q.get("logic_query"):
        start_entities = unique_preserve_order(
            [path.get("start") for path in aq.get("relational_path", [])]
        )
        question_text = q.get("question", "")
        if "(Answer type" in question_text:
            question_text = question_text.split("(Answer type")[0].strip()
        new_item = OrderedDict()
        new_item["question"] = question_text
        new_item["StartEntityIDs"] = start_entities
        for k, v in aq.items():
            if k not in ("question", "StartEntityIDs"):
                new_item[k] = v
        merged_list.append(new_item)
    else:
        mismatches.append({
            "index": idx,
            "all_queries_logic_query": aq.get("logic_query"),
            "questions_logic_query": q.get("logic_query")
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_list, f, ensure_ascii=False, indent=2)

with open(mismatch_log_path, "w", encoding="utf-8") as f:
    json.dump(mismatches, f, ensure_ascii=False, indent=2)

print(f"[OK] Merged {len(merged_list)} records; {len(mismatches)} mismatches logged to {mismatch_log_path}")
