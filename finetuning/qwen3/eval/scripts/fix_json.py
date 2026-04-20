import json
import os

def convert(input_file, output_file):
    items = []
    buf = ""

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            buf += line
            # 判断一条 JSON 是否结束（'}'）：
            if line.endswith("}"):
                try:
                    obj = json.loads(buf)
                    items.append(obj)
                except Exception as e:
                    print("解析失败:", buf)
                    raise e
                buf = ""

    # 输出单行 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


DATA_DIR = "/home/jhqy/IRF/finetuning/qwen3/eval/data"
convert(os.path.join(DATA_DIR, "clause.jsonl"), os.path.join(DATA_DIR, "clause_eval.jsonl"))
convert(os.path.join(DATA_DIR, "hallucination.jsonl"), os.path.join(DATA_DIR, "hallucination_eval.jsonl"))
convert(os.path.join(DATA_DIR, "seq2seq.jsonl"), os.path.join(DATA_DIR, "seq2seq_eval.jsonl"))
