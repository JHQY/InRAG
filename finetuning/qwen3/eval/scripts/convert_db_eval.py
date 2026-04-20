#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 db_test.json 路径（list结构）")
    parser.add_argument("--output", required=True, help="输出 db_eval.jsonl 路径")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as wf:
        for item in data:
            out = {
                "id": item.get("ID"),
                "question": item.get("input"),
                "actions": item.get("Actions", []),
                "gold_answer": item.get("Answer", "")
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"已将 {len(data)} 条样本写入 {args.output}")


if __name__ == "__main__":
    main()
