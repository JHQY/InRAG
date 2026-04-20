# # ============================================
# # eval_db.py — Stage1：数据库问答能力评估
# # 适配你的 DB 数据集（包含 input / Actions / Answer）
# # ============================================

# import json
# import argparse
# import time
# import requests
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel


# # ---------------------------
# # DeepSeek 作为评审模型
# # ---------------------------
# DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# DEEPSEEK_MODEL = "deepseek-chat"
# DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {DEEPSEEK_KEY}"
# }


# # ---------------------------
# # DeepSeek 评审 Prompt（简化为 DB 能力）
# # ---------------------------
# def build_judge_prompt(question, ansA, ansB):
#     return f"""
#     你现在是一名数据库问答（DB-QA）能力评估专家，你需要比较回答 A 与回答 B 哪个更好。
#     评估维度如下：

#     1. SQL 思路是否合理（sql_reasoning）
#     2. SQL 查询是否正确（sql_correctness）
#     3. 是否正确利用 SQLResult（sqlresult_usage）
#     4. 最终回答是否忠实 SQLResult（fidelity）
#     5. 是否存在幻想/编造（hallucination）
#     6. 最终回答是否结构完整（structure）

#     最终请输出 JSON，注意你必须使用双括号转义：

#     {{
#     "A": {{
#         "sql_reasoning": x,
#         "sql_correctness": x,
#         "sqlresult_usage": x,
#         "fidelity": x,
#         "hallucination": x,
#         "structure": x
#     }},
#     "B": {{
#         "sql_reasoning": x,
#         "sql_correctness": x,
#         "sqlresult_usage": x,
#         "fidelity": x,
#         "hallucination": x,
#         "structure": x
#     }},
#     "winner": "A" 或 "B" 或 "tie"
#     }}

#     【Question】
#     {question}

#     【回答A】
#     {ansA}

#     【回答B】
#     {ansB}


# """


# # ---------------------------
# # DeepSeek 评分
# # ---------------------------
# def judge(question, ansA, ansB):
#     prompt = build_judge_prompt(question, ansA, ansB)

#     for _ in range(3):
#         try:
#             resp = requests.post(
#                 DEEPSEEK_URL,
#                 headers=headers,
#                 json={
#                     "model": DEEPSEEK_MODEL,
#                     "messages": [{"role": "user", "content": prompt}],
#                     "temperature": 0.0,
#                 },
#                 timeout=30,
#             )
#             content = resp.json()["choices"][0]["message"]["content"]

#             # 尝试解析 JSON
#             try:
#                 return json.loads(content)
#             except:
#                 # 截取 {...}
#                 s = content.find("{")
#                 e = content.rfind("}")
#                 return json.loads(content[s:e+1])
#         except:
#             time.sleep(1)

#     return {"winner": "tie", "error": "fail"}


# # ---------------------------
# # 加载 base模型 + LoRA
# # ---------------------------
# def load_models(base_path, lora_path):
#     print("► Loading BASE model …")
#     tokenizer = AutoTokenizer.from_pretrained(base_path)

#     # A: baseline model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_path,
#         torch_dtype=torch.bfloat16,
#         device_map="cuda:0"
#     )
#     base_model.eval()

#     print("► Loading LoRA fine-tuned model …")
#     # B: load base again for lora
#     lora_base = AutoModelForCausalLM.from_pretrained(
#         base_path,
#         torch_dtype=torch.bfloat16,
#         device_map="cuda:0"
#     )

#     ft_model = PeftModel.from_pretrained(
#         lora_base,
#         lora_path,
#         torch_dtype=torch.bfloat16
#     )
#     ft_model.eval()

#     return tokenizer, base_model, ft_model


# # ---------------------------
# # 生成模型回答
# # ---------------------------
# def generate(model, tokenizer, prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
#     return tokenizer.decode(out[0], skip_special_tokens=True)


# # ---------------------------
# # 主评估流程
# # ---------------------------
# def evaluate(args):
#     tokenizer, base_model, ft_model = load_models(args.base, args.lora)

#     data = json.load(open(args.data, "r", encoding="utf-8"))

#     out = open(args.output, "w", encoding="utf-8")

#     for item in tqdm(data):
#         q = item["input"]

#         ansA = generate(base_model, tokenizer, q)
#         ansB = generate(ft_model, tokenizer, q)

#         result = judge(q, ansA, ansB)

#         out.write(json.dumps({
#             "id": item["ID"],
#             "question": q,
#             "base_answer": ansA,
#             "ft_answer": ansB,
#             "judge": result
#         }, ensure_ascii=False) + "\n")

#     out.close()
#     print(f"\n✓ 完成评估，已写入 {args.output}\n")


# # ---------------------------
# # CLI
# # ---------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base", required=True)
#     parser.add_argument("--lora", required=True)
#     parser.add_argument("--data", required=True)
#     parser.add_argument("--output", required=True)
#     args = parser.parse_args()

#     evaluate(args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# 防止 awq 乱入
sys.modules["awq"] = None
sys.modules["awq.quantize"] = None
sys.modules["awq.quantize.quantizer"] = None
sys.modules["awq.quantize.scale"] = None
sys.modules["awq.modules"] = None
sys.modules["awq.modules.linear"] = None

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import time
import requests

# ============================================
#   1. FINAL_DB_SYSTEM_PROMPT
# ============================================
FINAL_DB_SYSTEM_PROMPT = """
你是一名 SQL-CoT 推理助手，必须根据用户 Query 产生“用于理解 Query 的思维链”，而不是执行 SQL 或直接给答案。

以下是数据库结构（你只能使用这些字段）：

【保险公司】
- 公司编号 INTEGER
- 法定名称 TEXT
- 成立时间 TEXT
- 法定代表人 TEXT
- 官方网址 TEXT
- 公司住所 TEXT
- 注册资本(亿元人民币) REAL
- 经营范围 TEXT
- 公司缩写 TEXT
- 公司类型 TEXT
- 所属公司编号 INTEGER
- 公司总机 TEXT
- 经营区域 TEXT
- 客服热线 TEXT
- 传真 TEXT
- 邮编 TEXT
- 营业场所 TEXT

【保险产品】
- 产品编号 INTEGER
- 产品名称 TEXT
- 产品类型 TEXT
- 特色 TEXT
- 适宜人群 TEXT
- 产品网址 TEXT
- 责任免除 TEXT
- 免赔金额 TEXT
- 保险期间 TEXT
- 等待期 TEXT
- 犹豫期 TEXT
- 保险责任 TEXT
- 交费方式/投保年龄 TEXT
- 公司编号 INTEGER
- 销售状态 TEXT
- 红利 TEXT
- 保单贷款 TEXT

任务：
你的目标是通过 SQL-CoT 推理链展示你如何理解用户问题。
你不能给最终答案，也不能执行 SQL，也不能假装看到 SQLResult。

你必须输出以下格式（严格）：

<FieldSelection>
需要使用的字段：...
理由：...
</FieldSelection>

<SQLCoT>
#Step1
Thought: ...
SQL: ...

#Step2
Thought: ...
SQL: ...

#Step3
Thought: ...
SQL: ...
</SQLCoT>

规则：
- “字段选择（FieldSelection）”必须明确指出你将查询哪些字段并解释原因
- SQL 必须只使用上述字段名
- Thought 必须解释该 SQL 的目的
- SQL 不执行、不推断结果、不给 Answer
- 禁止幻觉字段、禁止虚构表、禁止额外知识
- 可多步推理，但必须结构化清晰
"""


# ============================================
#   2. DeepSeek-R1 评审器
# ============================================

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def deepseek_judge(question, gold_actions, base_out, ft_out):
    """
    使用 DeepSeek-R1 评分，返回 0~5 的客观评分。
    gold_actions 用于判断 similarity_to_gold。
    """

    judge_prompt = f"""
你是一个严谨的 SQL-CoT 推理链评估器，你需要对两个模型的 SQL-CoT 输出进行评分。他们将基于同一个用户问题进行回答，至多生成三步思考。且他们所输入的可参考数据库结构如下：
【保险公司】
- 公司编号 INTEGER
- 法定名称 TEXT
- 成立时间 TEXT
- 法定代表人 TEXT
- 官方网址 TEXT
- 公司住所 TEXT
- 注册资本(亿元人民币) REAL
- 经营范围 TEXT
- 公司缩写 TEXT
- 公司类型 TEXT
- 所属公司编号 INTEGER
- 公司总机 TEXT
- 经营区域 TEXT
- 客服热线 TEXT
- 传真 TEXT
- 邮编 TEXT
- 营业场所 TEXT

【保险产品】
- 产品编号 INTEGER
- 产品名称 TEXT
- 产品类型 TEXT
- 特色 TEXT
- 适宜人群 TEXT
- 产品网址 TEXT
- 责任免除 TEXT
- 免赔金额 TEXT
- 保险期间 TEXT
- 等待期 TEXT
- 犹豫期 TEXT
- 保险责任 TEXT
- 交费方式/投保年龄 TEXT
- 公司编号 INTEGER
- 销售状态 TEXT
- 红利 TEXT
- 保单贷款 TEXT


====== 用户问题 ======
{question}

====== Gold Actions（评测集中的标准 CoT） ======
{json.dumps(gold_actions, ensure_ascii=False, indent=2)}

====== 模型 A（baseline） 输出 ======
{base_out}

====== 模型 B（finetune） 输出 ======
{ft_out}

请根据以下七个维度对两个模型的输出进行评分，评分范围为0到5分，分数越高表示表现越好:
1. field_selection：是否选择了正确字段，是否遗漏或使用不存在的字段，若选择合理且完整得高分，若选择不合理或遗漏，一个扣除0.5分，最低0分。
2. sql_reasoning：推理链是否清晰、多步逻辑是否合理，若推理链清晰且合理得高分，若推理链混乱或不合理，一个扣除0.5分，最低0分。
3. sql_correctness：SQL 语句是否合法、字段是否真实存在。若 SQL 语句正确得高分，若存在语法错误或使用不存在字段，一个语法或字段扣除0，5分，最低0分。
4. structure：是否遵循 <FieldSelection> + <SQLCoT> 格式。 若格式正确得高分，若格式混乱或缺失，一个扣除1分，若全错得0分，最低0分。
5. hallucination：是否产生幻觉（不存在字段/表/额外知识）【越低越好 → 高分=少幻觉】，若出现任何一个不存在字段/表/额外知识，扣1分，最低0分。
6. fidelity：是否遵守系统规则（不回答最终问题、不执行 SQL）此条较为宽松，以你的感受为准。
7. similarity_to_gold：与 gold action 的思路相近程度，这道题是附加题，不算在总分之内。如果哪一个更符合思路，请在该项中写标注“更符合”，否则写“相似”。

请按下述格式给出结果：

{{
 "A": {{
    "field_selection": x,
    "sql_reasoning": x,
    "sql_correctness": x,
    "structure": x,
    "hallucination": x,
    "fidelity": x,
    "similarity_to_gold": x
 }},
 "B": {{
    "field_selection": x,
    "sql_reasoning": x,
    "sql_correctness": x,
    "structure": x,
    "hallucination": x,
    "fidelity": x,
    "similarity_to_gold": x
 }},
 "winner": "A" 或 "B" 或 "tie"
}}
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "user", "content": judge_prompt}
        ]
    }

    try:
        resp = requests.post(DEEPSEEK_API, headers=headers, json=payload)
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        return json.loads(text)

    except Exception as e:
        return {"error": str(e)}


# ============================================
#   3. 加载模型（baseline + LoRA）
# ============================================

def load_baseline(base_path):
    print("► Loading baseline model...")
    tok = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    return tok, model


def load_finetuned(base_path, lora_path):
    print("► Loading LoRA model...")
    tok = AutoTokenizer.from_pretrained(base_path)

    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return tok, model
# ============================================
#   4. 生成 SQL-CoT 输出
# ============================================
def clean_output(text):
    # 只截取从 <FieldSelection> 开始的内容
    m = re.search(r"<FieldSelection>.*", text, flags=re.S)
    if m:
        return m.group(0).strip()
    return text.strip()

def generate(model, tokenizer, user_query):
    messages = [
        {"role": "system", "content": FINAL_DB_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9,
        )

    return clean_output(tokenizer.decode(out[0], skip_special_tokens=True))



# ============================================
#   5. 主流程：读取数据 → 生成 → deepseek 评审 → 保存
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("► Loading evaluation data…")
    eval_data = [json.loads(l.strip()) for l in open(args.data, "r", encoding="utf-8")]

    results = []

    # ========== Step 1：baseline ==========
    tokA, modelA = load_baseline(args.base)
    base_outputs = {}

    for item in tqdm(eval_data, desc="Baseline 推理"):
        q = item["question"]
        base_outputs[item["id"]] = generate(modelA, tokA, q)
        print(base_outputs[item["id"]])

    del modelA
    torch.cuda.empty_cache()
    time.sleep(1)

    # ========== Step 2：finetune ==========
    tokB, modelB = load_finetuned(args.base, args.lora)
    ft_outputs = {}

    for item in tqdm(eval_data, desc="Finetune 推理"):
        q = item["question"]
        ft_outputs[item["id"]] = generate(modelB, tokB, q)
        print(ft_outputs[item["id"]])

    del modelB
    torch.cuda.empty_cache()
    time.sleep(1)

    # ========== Step 3：DeepSeek judge ==========
    print("► DeepSeek 评分中…")

    for item in tqdm(eval_data, desc="Scoring"):
        q = item["question"]
        gold_actions = item["actions"]

        base_out = base_outputs[item["id"]]
        ft_out = ft_outputs[item["id"]]

        judge = deepseek_judge(q, gold_actions, base_out, ft_out)

        results.append({
            "id": item["id"],
            "question": q,
            "base_answer": base_out,
            "ft_answer": ft_out,
            "judge": judge
        })

    # ========== 保存结果 ==========
    print("► Saving:", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✓ Done!")


if __name__ == "__main__":
    main()
