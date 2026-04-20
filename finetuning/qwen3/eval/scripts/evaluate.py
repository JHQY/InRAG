# ============================================
# evaluate.py — Insurance-Finetune 专用评估器
# ============================================

import sys
sys.modules["awq"] = None
sys.modules["awq.quantize"] = None
sys.modules["awq.quantize.quantizer"] = None
sys.modules["awq.quantize.scale"] = None
sys.modules["awq.modules"] = None
sys.modules["awq.modules.linear"] = None

import argparse
import json
import time
import requests
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =======================
# DeepSeek API 配置
# =======================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_KEY}"
}


# =======================
# 新版评审 Prompt
# =======================
def build_judge_prompt(context, ansA, ansB):
    return f"""
你是一名**保险行业微调模型**的质量评审专家。你不比较“答案是否正确”，而是比较：

1. **格式执行度（format）**  
   - 是否严格遵守 prompt 的结构，例如 <Answer>、<sql>、"结论/证据/解释" 三段式等。

2. **保险行业风格专业度（style_professional）**  
   - 是否呈现类似合同条款解读类的专业语言，而非口语、聊天式表达。

3. **术语准确性（term_accuracy）**  
   - 是否正确使用“等待期”“给付”“责任免除”“如实告知”等专业词汇，避免误用。

4. **幻觉控制（hallucination）**  
   - 是否避免捏造合同条款、扩展不存在的事实、生成未提供的定义。

5. **条款材料忠实度（fidelity_to_context）**  
   - 回答是否仅基于 context，而没有违背 context 内容。

6. **结构完整性（structure_completeness）**  
   - 回答是否缺段、丢标签、内容残缺。

请为回答 A 和回答 B 在以上六个维度分别给出 0–10 的评分。  
并给出 winner："A"、"B" 或 "tie"。

请严格输出如下 JSON 格式：

{{
  "A": {{
      "format": x,
      "style_professional": x,
      "term_accuracy": x,
      "hallucination": x,
      "fidelity_to_context": x,
      "structure_completeness": x
  }},
  "B": {{
      "format": x,
      "style_professional": x,
      "term_accuracy": x,
      "hallucination": x,
      "fidelity_to_context": x,
      "structure_completeness": x
  }},
  "winner": "A" 或 "B" 或 "tie"
}}

【Context】
{context}

【回答 A】
{ansA}

【回答 B】
{ansB}
"""


# =======================
# 调用 DeepSeek (LLM-as-a-Judge)
# =======================
def judge_with_deepseek(context, ansA, ansB):
    prompt = build_judge_prompt(context, ansA, ansB)

    for _ in range(3):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=30,
            )
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # 尝试直接解析 JSON
            try:
                return json.loads(content)
            except:
                # 尝试截取 { ... }
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    block = content[start : end + 1]
                    return json.loads(block)

        except Exception as e:
            print(f"[DeepSeek 重试] {e}")
            time.sleep(2)

    return {
        "A": {}, "B": {}, "winner": "tie", "error": "failed_to_parse"
    }


# =======================
# 模型加载
# =======================
def load_models(base_path, lora_path):
    print("► Loading BASE model …")
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("► Loading LoRA adapter …")
    ft_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16
    )
    ft_model.eval()

    return tokenizer, base_model, ft_model


# =======================
# 推理
# =======================
def generate_answer(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# =======================
# 单文件评估
# =======================
def evaluate_file(path, tokenizer, base_model, ft_model, outfile):

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)

        # 自动选择 context
        context = (
            item.get("raw_input")
            or item.get("prompt")
            or item.get("text")
            or item.get("contract_name")
        )
        if context is None:
            print(f"[WARNING] ID={item.get('id')} 缺少 context，跳过")
            continue

        # baseline
        base_ans = generate_answer(tokenizer, base_model, context)

        # finetune
        ft_ans = generate_answer(tokenizer, ft_model, context)

        # judge
        judge = judge_with_deepseek(context, base_ans, ft_ans)

        result = {
            "id": item.get("id"),
            "task": item.get("task"),
            "context": context,
            "base_answer": base_ans,
            "ft_answer": ft_ans,
            "judge": judge,
        }

        with open(outfile, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(result, ensure_ascii=False) + "\n")


# =======================
# 主函数
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--lora")
    parser.add_argument("--eval_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    tokenizer, base_model, ft_model = load_models(args.base, args.lora)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    files = [
        "clause_eval.jsonl",
        "hallucination_eval.jsonl",
        "seq2seq_eval.jsonl",
    ]

    for fname in files:
        path = os.path.join(args.eval_dir, fname)
        outfile = os.path.join(args.output_dir, f"result_{fname}")
        print(f"\n========== Evaluating {fname} ==========")
        evaluate_file(path, tokenizer, base_model, ft_model, outfile)


if __name__ == "__main__":
    main()
