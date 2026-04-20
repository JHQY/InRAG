import json
import re
from pathlib import Path


def clean_answer(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]:?", "", text)
    return text.strip()


def write_jsonl(path, data):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {path} ({len(data)} samples)")


# ---------------------------
# 1. 主观条款
# ---------------------------
def process_subjective(src, dst):
    raw = json.load(open(src, "r", encoding="utf-8"))
    result = []

    for contract, qa_dict in raw.items():
        for question, item in qa_dict.items():
            if not isinstance(item, dict):
                continue
            p = item.get("p", "").strip()
            ans = clean_answer(item.get("answer", ""))

            if not p or not ans:
                continue

            instr = (
                "你是一名专业保险合同分析助手。\n\n"
                "【任务类型】 主观解释\n\n"
                f"【合同片段】\n{p}\n\n"
                f"【用户提问】\n{question}\n\n"
                "请生成包含结论、证据、解释说明的回答："
            )

            result.append({
                "instruction": instr,
                "output": ans
            })

    write_jsonl(dst, result)


# ---------------------------
# 2. 客观条款
# ---------------------------
def process_objective(src, dst):
    raw = json.load(open(src, "r", encoding="utf-8"))
    result = []

    for entry in raw:
        if not isinstance(entry, dict):
            continue

        text = entry.get("input", "")
        ans = clean_answer(entry.get("output", ""))

        if not text or not ans:
            continue

        if "用户提问：" in text:
            parts = text.split("用户提问：", 1)
            p = parts[0].replace(
                "你是一个保险行业的从业者，需要为保险的消费者解答问题。下面根据保险合同的片段给出回答：",
                ""
            ).strip()
            q = parts[1].strip()
        else:
            p = text
            q = ""

        instr = (
            "你是一名专业保险合同分析助手。\n\n"
            "【任务类型】 客观判断\n\n"
            f"【合同片段】\n{p}\n\n"
            f"【用户提问】\n{q}\n\n"
            "请给出唯一明确的结论："
        )

        result.append({
            "instruction": instr,
            "output": ans
        })

    write_jsonl(dst, result)


# ---------------------------
# 3. DB（SQL+Answer 两种样本）
# ---------------------------
def process_db(src, dst):
    raw = json.load(open(src, "r", encoding="utf-8"))
    result = []

    for sample in raw:
        if not isinstance(sample, dict):
            continue

        system_prompt = sample.get("system", "").strip()
        conv = sample.get("conversations", [])

        if not conv or len(conv) < 2:
            continue

        # 找第一轮
        first_human = None
        first_gpt = None
        for m in conv:
            if m.get("from") == "human" and first_human is None:
                first_human = m
            elif m.get("from") == "gpt" and first_gpt is None:
                first_gpt = m

        # 找第二轮
        second_human = None
        second_gpt = None
        found_first_gpt = False
        for m in conv:
            if m is first_gpt:
                found_first_gpt = True
                continue
            if found_first_gpt:
                if m.get("from") == "human" and second_human is None:
                    second_human = m
                elif second_human and m.get("from") == "gpt" and second_gpt is None:
                    second_gpt = m

        # SQL 样本
        if first_human and first_gpt:
            instr = (
                system_prompt + "\n\n"
                "【任务类型】 数据库问答（SQL生成）\n\n"
                f"【用户提问】\n{first_human['value']}\n\n"
                "请分析问题，先写 <Thought></Thought>，再写 <sql></sql>："
            )
            result.append({
                "instruction": instr,
                "output": first_gpt["value"]
            })

        # 最终 Answer 样本
        if second_gpt:
            instr = (
                system_prompt + "\n\n"
                "【任务类型】 数据库问答（最终回答）\n\n"
                f"用户：{first_human['value']}\n\n"
                f"助手：{first_gpt['value']}\n\n"
                f"用户：{second_human['value']}\n\n"
                "请根据 <exe> 的结果生成最终回答，使用 <Answer></Answer> 包裹："
            )
            result.append({
                "instruction": instr,
                "output": second_gpt["value"]
            })

    write_jsonl(dst, result)


# ---------------------------
# 主入口
# ---------------------------
if __name__ == "__main__":
    process_subjective("clause_subjective.json", "clause_subjective_instruct.jsonl")
    process_objective("clause_objective.json", "clause_objective_instruct.jsonl")
    process_db("db_train.json", "db_train_instruct.jsonl")

    print("\n🔥 ALL DONE — datasets are now in Axolotl instruct format!")
