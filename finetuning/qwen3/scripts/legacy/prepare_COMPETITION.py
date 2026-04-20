import json
import re
from pathlib import Path

# ---------------------------
# 工具：清洗 answer（去掉 [答案]、[证据] 等）
# ---------------------------
def clean_answer(text):
    if not isinstance(text, str):
        return ""
    # 移除 [答案] [证据] [解释说明] 等 tag
    text = re.sub(r"\[.*?\]:?", "", text)
    return text.strip()


def write_jsonl(path, data):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {path}    ({len(data)} samples)")


# ---------------------------
# 1. 主观条款：clause_subjective.json
# ---------------------------
def process_subjective(src_path, dst_path):
    data_raw = json.load(open(src_path, "r", encoding="utf-8"))
    result = []

    # 结构：{ "合同名": { "问题": { "p": "...", "answer": "..." }, ... }, ... }
    for contract_name, qa_dict in data_raw.items():
        for question, item in qa_dict.items():
            if not isinstance(item, dict):
                continue
            p = item.get("p", "").strip()
            answer = clean_answer(item.get("answer", ""))

            if not p or not answer:
                continue

            prompt = (
                "你是一名专业保险合同分析助手。\n\n"
                "【任务类型】\n主观解释\n\n"
                f"【合同名称】\n{contract_name}\n\n"
                f"【合同片段】\n{p}\n\n"
                f"【用户提问】\n{question}\n\n"
                "请生成包含：结论、证据、解释说明 的回答："
            )

            # 这里额外构造 text，把 prompt 和 answer 一起塞进去
            text = (
                "<PROMPT>\n"
                + prompt
                + "\n</PROMPT>\n\n"
                "<COMPLETION>\n"
                + answer
                + "\n</COMPLETION>"
            )

            result.append({
                "task": "subjective_clause",
                "contract_name": contract_name,
                "prompt": prompt,
                "completion": answer,
                "text": text
            })

    write_jsonl(dst_path, result)


# ---------------------------
# 2. 客观条款：clause_objective.json
# ---------------------------
def process_objective(src_path, dst_path):
    data_raw = json.load(open(src_path, "r", encoding="utf-8"))
    result = []

    # 结构：[{ "input": "...用户提问：xxx", "output": "...", "id": 1 }, ...]
    for entry in data_raw:
        if not isinstance(entry, dict):
            continue

        text_in = entry.get("input", "")
        answer = clean_answer(entry.get("output", ""))

        if not text_in or not answer:
            continue

        q = ""
        p = ""

        if "用户提问：" in text_in:
            parts = text_in.split("用户提问：", 1)
            p = parts[0].replace(
                "你是一个保险行业的从业者，需要为保险的消费者解答问题。下面根据保险合同的片段给出回答：",
                ""
            ).strip()
            q = parts[1].strip()
        else:
            p = text_in
            q = ""

        prompt = (
            "你是一名专业保险合同分析助手。\n\n"
            "【任务类型】\n客观判断\n\n"
            f"【合同片段】\n{p}\n\n"
            f"【用户提问】\n{q}\n\n"
            "请给出唯一明确的结论："
        )

        text = (
            "<PROMPT>\n"
            + prompt
            + "\n</PROMPT>\n\n"
            "<COMPLETION>\n"
            + answer
            + "\n</COMPLETION>"
        )

        result.append({
            "task": "objective_clause",
            "id": entry.get("id"),
            "raw_input": text_in,
            "prompt": prompt,
            "completion": answer,
            "text": text
        })

    write_jsonl(dst_path, result)


# ---------------------------
# 3. DB 数据：db_train.json（多轮对话 + system）
# ---------------------------
def process_db(src_path, dst_path):
    data_raw = json.load(open(src_path, "r", encoding="utf-8"))
    result = []

    for sample in data_raw:
        if not isinstance(sample, dict):
            continue

        system_prompt = sample.get("system", "").strip()
        conv = sample.get("conversations", [])
        if not conv or len(conv) < 2:
            continue

        # 找第一轮 human & 第一个 gpt（SQL 那轮）
        first_human = None
        first_gpt = None
        for m in conv:
            if m.get("from") == "human" and first_human is None:
                first_human = m
            elif m.get("from") == "gpt" and first_gpt is None:
                first_gpt = m

        # 第二轮 human & gpt（Answer 那轮）
        second_human = None
        second_gpt = None
        found_first_human = False
        found_first_gpt = False
        for m in conv:
            if m is first_human:
                found_first_human = True
                continue
            if m is first_gpt:
                found_first_gpt = True
                continue

            if found_first_gpt and m.get("from") == "human" and second_human is None:
                second_human = m
            elif second_human is not None and m.get("from") == "gpt" and second_gpt is None:
                second_gpt = m

        # 样本 1：SQL 生成（Thought + <sql>）
        if first_human and first_gpt:
            q1 = first_human.get("value", "").strip()
            a1 = first_gpt.get("value", "").strip()

            if q1 and a1:
                prompt_sql = (
                    system_prompt + "\n\n"
                    "【任务类型】\n数据库问答-SQL生成\n\n"
                    f"【用户提问】\n{q1}\n\n"
                    "请分析用户问题，先用 <Thought></Thought> 给出思考过程，"
                    "再用 <sql></sql> 给出对应的 SQL 语句："
                )

                text_sql = (
                    "<PROMPT>\n"
                    + prompt_sql
                    + "\n</PROMPT>\n\n"
                    "<COMPLETION>\n"
                    + a1
                    + "\n</COMPLETION>"
                )

                result.append({
                    "task": "db_sql",
                    "system": system_prompt,
                    "prompt": prompt_sql,
                    "completion": a1,
                    "text": text_sql
                })

        # 样本 2：最终回答生成（基于 <exe>）
        if first_human and first_gpt and second_human and second_gpt:
            q1 = first_human.get("value", "").strip()
            mid = first_gpt.get("value", "").strip()
            q2 = second_human.get("value", "").strip()
            a2 = second_gpt.get("value", "").strip()

            if q1 and a2:
                prompt_ans = (
                    system_prompt + "\n\n"
                    "【任务类型】\n数据库问答-结果回答\n\n"
                    "下面是之前的对话：\n"
                    f"用户：{q1}\n\n"
                    f"助手：{mid}\n\n"
                    f"用户：{q2}\n\n"
                    "现在请你只根据上述对话和 <exe> 中的查询结果，"
                    "直接生成最终回答，回答内容用 <Answer></Answer> 包裹："
                )

                text_ans = (
                    "<PROMPT>\n"
                    + prompt_ans
                    + "\n</PROMPT>\n\n"
                    "<COMPLETION>\n"
                    + a2
                    + "\n</COMPLETION>"
                )

                result.append({
                    "task": "db_answer",
                    "system": system_prompt,
                    "prompt": prompt_ans,
                    "completion": a2,
                    "text": text_ans
                })

    write_jsonl(dst_path, result)


# ---------------------------
# 入口：一次性处理三个数据集
# ---------------------------
if __name__ == "__main__":
    # 路径按你自己的数据位置改，这里默认在当前目录
    process_subjective(
        "clause_subjective.json",
        "clause_subjective_completion.jsonl"
    )

    process_objective(
        "clause_objective.json",
        "clause_objective_completion.jsonl"
    )

    process_db(
        "db_train.json",
        "db_train_completion.jsonl"
    )

    print("\n🔥 All datasets converted to completion format successfully!")
