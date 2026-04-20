import json
import re
from typing import Tuple


def parse_objective_input(text: str) -> Tuple[str, str, str]:
    """
    针对 objective.json 中的 input 字段：
    从 input 文本中提取：
    - contract（合同名）
    - clause（条款内容）
    - question（用户提问）
    """
    if not isinstance(text, str):
        return "", "", ""

    # 合同名称：匹配“合同是：XXXX\n”
    contract_match = re.search(r"合同是[:：]\s*(.+?)\n", text)
    contract = contract_match.group(1).strip() if contract_match else ""

    # 条款内容：匹配“片段内容：XXX\n用户提问”
    clause_match = re.search(r"片段内容[:：]\s*(.+?)\n用户提问", text, re.S)
    clause = clause_match.group(1).strip() if clause_match else ""

    # 用户问题：匹配“用户提问：XXX”
    question_match = re.search(r"用户提问[:：]\s*(.+)", text)
    question = question_match.group(1).strip() if question_match else ""

    return contract, clause, question


def convert_objective_io(
    input_path: str,
    output_path: str,
):
    """
    处理 objective.json：
    结构：[
      { "input": "...", "output": "...", "id": ... },
      ...
    ]
    转成 chat 格式，并标记为【任务类型：客观判断】
    """
    data = json.load(open(input_path, "r", encoding="utf-8"))
    out = []

    print(f"[objective] Loaded {len(data)} items from {input_path}")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[objective][WARN] non-dict at idx={idx}, skip")
            continue

        text = item.get("input", "")
        answer = item.get("output", "").strip()

        contract, clause, question = parse_objective_input(text)

        if not clause or not question:
            print(f"[objective][WARN] missing clause/question at idx={idx}, skip")
            continue

        user_prompt = (
            "【任务类型：客观判断】\n"
            f"合同名称：{contract}\n\n"
            f"条款内容：\n{clause}\n\n"
            f"问题：{question}\n"
            "请根据条款内容给出明确判断，不需要额外解释。"
        )

        out.append(
            {
                "conversations": [
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": answer},
                ]
            }
        )

    json.dump(out, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[objective] Saved: {output_path} (valid items: {len(out)})")


def convert_subjective_nested(
    input_path: str,
    output_path: str,
):
    """
    处理 subjective.json：
    结构大致为：
    {
      "onsale 1.鼎诚团体意外伤害保险条款": {
          "问题1": {
              "p": "...条款...",
              "answer": "...带[答案]/[证据]/[解释说明]..."
          },
          "问题2": {...}
      },
      "onsale 2.XXX": {...}
    }
    转成 chat 格式，并标记为【任务类型：主观解释】
    """
    data = json.load(open(input_path, "r", encoding="utf-8"))
    out = []

    if not isinstance(data, dict):
        print(f"[subjective][ERROR] root is not dict, got {type(data)}")
        return

    print(f"[subjective] Loaded {len(data)} top-level entries from {input_path}")

    for contract_key, qa_dict in data.items():
        # contract_key: "onsale 1.鼎诚团体意外伤害保险条款"
        contract_name = contract_key.strip()

        if not isinstance(qa_dict, dict):
            print(f"[subjective][WARN] value of {contract_name} is not dict, skip")
            continue

        for question, qa_item in qa_dict.items():
            if not isinstance(qa_item, dict):
                print(f"[subjective][WARN] qa_item for question={question} not dict, skip")
                continue

            clause = qa_item.get("p", "").strip()
            answer = qa_item.get("answer", "").strip()
            q_text = question.strip()

            if not clause or not q_text or not answer:
                print(f"[subjective][WARN] missing field at question={q_text}, skip")
                continue

            user_prompt = (
                "【任务类型：主观解释】\n"
                f"合同名称：{contract_name}\n\n"
                f"条款内容：\n{clause}\n\n"
                f"问题：{q_text}\n"
                "请结合条款内容解释原因，并给出分析性回答。"
            )

            out.append(
                {
                    "conversations": [
                        {"from": "user", "value": user_prompt},
                        {"from": "assistant", "value": answer},
                    ]
                }
            )

    json.dump(out, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[subjective] Saved: {output_path} (valid items: {len(out)})")


if __name__ == "__main__":
    # 按你的实际路径改一下就行
    base = "/home/jhqy/IRF/finetuning/qwen3/data"

    # 1. objective：input/output 格式
    convert_objective_io(
        f"{base}/clause_objective.json",
        f"{base}/clause_objective_chat.json",
    )

    # 2. subjective：嵌套 dict 格式
    convert_subjective_nested(
        f"{base}/clause_subjective.json",
        f"{base}/clause_subjective_chat.json",
    )
    print("所有条款数据转换完成！")