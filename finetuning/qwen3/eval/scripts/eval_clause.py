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
#   1. 统一的系统提示（Clause 模型用）
# ============================================

CLAUSE_SYSTEM_PROMPT = """
你是一名严谨的保险条款问答助手，需要基于 contract_fragment 回答用户问题。

规则：
1. 不得使用 contract_fragment 之外的知识。
2. 不得幻想不存在的条款。
3. 对于 objective 任务，请给出明确的是非判断。
4. 对于 subjective 任务，需要给出：
   - conclusion（结论）
   - evidence（引用条款）
   - explanation（解释）
5. 请保持表达简洁、基于条款本身。
"""


# ============================================
#   2. DeepSeek-R1 评审器
# ============================================

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def build_judge_prompt(task_type, prompt, gold_answer_json, base_out, ft_out):
    judge_prompt = f"""
你是一名严谨的保险条款问答评估器，负责比较两个模型（A=baseline, B=finetune）在“保险合同条款理解任务”中的表现。

该任务分为两类：

【objective 任务】
- 回答必须包含较为明确的是非判断。
- 不允许补充其他外源知识
- 必须只基于 contract_fragment 中的内容进行推理
- 评估重点：是否直接依据条款、是否逻辑自洽、是否出现幻觉、回答是否准确

【subjective 任务】
包含三部分：
- conclusion：对于问题的答复，也可以称为“答案”
- evidence：必须引用（准确引用）contract_fragment 原文或核心信息
- explanation：必须给出基于条款的合理解释
- 若缺失任一部分，则要扣 structure 分
评估重点：三段结构、是否直接依据条款、是否逻辑自洽、是否出现幻觉、回答是否准确

================================================
评测数据：
- 任务类型: {task_type}
- 保险合同名称（contract_name）、条款片段（contract_fragment）、用户问题（question）均在输入的prompt中，需要你自行解析：
----------------
{prompt}
----------------

标准答案（gold_answer）如下：
{gold_answer_json}

================================================
模型 A（baseline）输出：
{base_out}

模型 B（finetune）输出：
{ft_out}

================================================
请按以下 7 个维度对两个模型分别评分（每项 0~5 分，越高越好）：

1. correctness（正确性）
   - objective：给出明确的是非判断且与 gold_answer 的判断方向一致，若不一致，记0分
   - subjective：结论是否与 gold_answer方向一致，若不一致，记0分。同时检查推理是否正确，合理即可。

2. evidence_use（依据条款程度）
   - 是否明确使用 contract_fragment 的内容推理，若没有依据条款内容推理，出现一次扣一分，最低0分。
   - 特别的，subjective 任务必须检查 evidence 是否引用片段内容，若未引用，记0分。

3. no_hallucination（无幻觉）（对于两个问题判断标准一致）
   - 不得虚构条款、不得加入 contract_fragment 外内容。出现一次扣 1 分，最低0分。

4. structure（结构符合任务类型）
   - objective：有明确的是非判断即可。若无，记0分。
   - subjective：必须包含 conclusion / evidence / explanation 三部分，缺一部分扣1.5分，全部缺失，记0分。

5. clarity（表达清晰度）
   - 是否逻辑清楚易理解，较为宽松，能通顺表达即可。若完全无法理解，记0分。

6. fidelity（是否遵守说明）
   - 不跳题，不给无关内容

7. similarity_to_gold（附加项，不计入胜负）
   - 哪个与 gold_answer 的 reasoning 思路更接近（写 “更接近” / “相似”）

================================================
请按如下 JSON 结构给出最终结果：

{{
 "A": {{
    "correctness": x,
    "evidence_use": x,
    "no_hallucination": x,
    "structure": x,
    "clarity": x,
    "fidelity": x,
    "similarity_to_gold": "更接近" / "相似"
 }},
 "B": {{
    "correctness": x,
    "evidence_use": x,
    "no_hallucination": x,
    "structure": x,
    "clarity": x,
    "fidelity": x,
    "similarity_to_gold": "更接近" / "相似"
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
#   3. 加载 baseline & LoRA finetuned
# ============================================

def load_baseline(base_path):
    print("► Loading baseline model...")
    tok = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()
    return tok, model


def load_finetuned(base_path, lora_path):
    print("► Loading LoRA model...")
    tok = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return tok, model


# ============================================
#   4. generation function
# ============================================

def clean_output(text):
    return text.strip()


def generate(model, tokenizer, prompt):
    """
    prompt 是完整拼好的包含合同片段的字符串
    """
    messages = [
        {"role": "system", "content": CLAUSE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=380,
            temperature=0.2,
            top_p=0.9,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_output(text)


# ============================================
#   5. main: read → generate → deepseek → save
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("► Loading evaluation data…")
    eval_data = [json.loads(l) for l in open(args.data, "r", encoding="utf-8")]

    # ================= baseline =================
    tokA, modelA = load_baseline(args.base)
    base_outputs = {}

    for item in tqdm(eval_data, desc="Baseline 推理"):
        base_outputs[item["id"]] = generate(
            modelA, tokA, item["prompt"]
        )
        print(base_outputs[item["id"]])

    del modelA
    torch.cuda.empty_cache()
    time.sleep(1)

    # ================= finetuned =================
    tokB, modelB = load_finetuned(args.base, args.lora)
    ft_outputs = {}

    for item in tqdm(eval_data, desc="Finetuned 推理"):
        ft_outputs[item["id"]] = generate(
            modelB, tokB, item["prompt"]
        )
        print(ft_outputs[item["id"]])

    del modelB
    torch.cuda.empty_cache()
    time.sleep(1)

    # ================= scoring =================
    print("► DeepSeek 评分中…")
    results = []


    for item in tqdm(eval_data, desc="Scoring"):
        judge = build_judge_prompt(
            task_type=item["task_type"],
            # contract_name=item["contract_name"],
            #contract_fragment=item["contract_fragment"],
            prompt=item["prompt"],
            gold_answer_json=item["gold_answer_text"],
            base_out=base_outputs[item["id"]],
            ft_out=ft_outputs[item["id"]]
        )

        results.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "gold_answer": item["gold_answer_text"],
            "base_output": base_outputs[item["id"]],
            "ft_output": ft_outputs[item["id"]],
            "judge": judge
        })

    # save
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✓ Done! →", args.output)


if __name__ == "__main__":
    main()
