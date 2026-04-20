import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/jhqy/IRF/finetuning/qwen3/Qwen3-4B", trust_remote_code=True)

def analyze_one(path):
    lens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = ""

            # completion 格式 → 有 text 或 prompt/completion
            if "text" in item:
                text = item["text"] + item.get("completion", "")
            else:
                # 我们自己生成的结构是 prompt + completion
                text = item["prompt"] + item["completion"]

            ids = tokenizer(text, return_tensors="np")["input_ids"]
            lens.append(len(ids[0]))

    lens.sort()
    print(f"\n📌 File: {path}")
    print(f"Samples: {len(lens)}")
    print(f"Min: {lens[0]}")
    print(f"Median: {lens[len(lens)//2]}")
    print(f"95th percentile: {lens[int(len(lens)*0.95)]}")
    print(f"Max: {lens[-1]}")
    return lens


files = [
    "clause_subjective_completion.jsonl",
    "clause_objective_completion.jsonl",
    "db_train_completion.jsonl",
]

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
