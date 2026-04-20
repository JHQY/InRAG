import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/jhqy/IRF/finetuning/qwen3/Qwen3-4B", trust_remote_code=True)

files = [
    "clause_objective_completion.jsonl",
    "clause_subjective_completion.jsonl",
    "db_train_completion.jsonl"
]

def analyze_one(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    lens = []
    for item in data:
        conv = item["conversations"]
        # 拼接所有 user + assistant 文本
        full = ""
        for turn in conv:
            full += turn["value"] + "\n"

        tokens = tokenizer(full, return_tensors="np")["input_ids"]
        lens.append(len(tokens[0]))

    lens.sort()
    print(f"\n📌 File: {path}")
    print(f"Samples: {len(lens)}")
    print(f"Min: {lens[0]}")
    print(f"Median: {lens[len(lens)//2]}")
    print(f"95th percentile: {lens[int(len(lens)*0.95)]}")
    print(f"Max: {lens[-1]}")
    return lens

all_lens = []
for f in files:
    lens = analyze_one(f)
    all_lens.extend(lens)

all_lens.sort()
print("\n==============================")
print("🔥 Overall dataset token length distribution")
print("==============================")
print(f"Total samples: {len(all_lens)}")
print(f"Min: {all_lens[0]}")
print(f"Median: {all_lens[len(all_lens)//2]}")
print(f"95th percentile: {all_lens[int(len(all_lens)*0.95)]}")
print(f"Max: {all_lens[-1]}")
