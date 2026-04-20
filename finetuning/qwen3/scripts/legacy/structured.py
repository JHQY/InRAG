import json

input_file = "finetuning/qwen3/data/InsQABench/db_train.json"
output_file = "train_sft.jsonl"

with open(input_file, "r", encoding="utf8") as f_in, open(output_file, "w", encoding="utf8") as f_out:
    for line in f_in:
        item = json.loads(line)

        conversations = []

        # 1. 把 system 放入首位
        if "system" in item:
            conversations.append({
                "from": "system",
                "value": item["system"]
            })

        # 2. 原对话直接追加
        for m in item["conversations"]:
            conversations.append({
                "from": m["from"],
                "value": m["value"]
            })

        # 写入新格式
        f_out.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")

print("转换完成 → 写入 train_sft.jsonl")
