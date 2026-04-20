"""
RAFT 三模型对比评估脚本

对比 baseline / stage2 / stage3_raft 在三类任务上的表现：
  1. clause_quality  - 标准保险条款QA（各自使用原生格式）
  2. hallucination   - 幻觉抑制（evidence不足时是否正确拒答）
  3. distractor      - RAG干扰样本（evidence不匹配时是否拒绝作答）

用法：
  python eval_raft.py \
    --base   /path/to/Qwen3-4B \
    --lora2  /path/to/qwen3-4b-lora-stage2 \
    --lora3  /path/to/qwen3-4b-lora-raft \
    --clause_data  /path/to/clause.jsonl \
    --hall_data    /path/to/hallucination_eval.jsonl \
    --output_dir   /path/to/eval/results
"""

import sys
sys.modules["awq"] = None
sys.modules["awq.quantize"] = None

import argparse
import json
import os
import random
import re
import time
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

random.seed(42)

# ─── API 配置 ────────────────────────────────────────────────────────
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
with open(os.path.join(os.path.dirname(__file__), "../apikey")) as f:
    DEEPSEEK_KEY = f.read().strip()

# ─── System prompts ──────────────────────────────────────────────────
CLAUSE_SYSTEM = (
    "你是一名严谨的保险条款问答助手，需要基于提供的合同片段回答用户问题。"
    "不得使用合同片段之外的知识，不得幻想不存在的条款。"
)

RAFT_SYSTEM = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识。"
    "如果检索结果不包含回答所需信息，请明确告知用户无法作答。"
)

# ─── 模型加载 ─────────────────────────────────────────────────────────
def load_model(base_path, lora_path=None, label=""):
    print(f"► 加载 {label}...")
    tok = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tok, model

def unload_model(model):
    del model
    torch.cuda.empty_cache()
    time.sleep(1)

# ─── 生成 ────────────────────────────────────────────────────────────
def generate(model, tok, system_prompt, user_content, max_new=400):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # 只保留助手输出部分
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return text.strip()

# ─── 构造输入 ─────────────────────────────────────────────────────────
def build_clause_input(item):
    """原生格式：单chunk，用于 baseline / stage2"""
    return item["prompt"]

def _extract_field(item, prompt_patterns, fallback_keys):
    """先从 prompt 正则提取，失败则用 item 字段"""
    for pat, flags in prompt_patterns:
        m = re.search(pat, item.get("prompt", ""), flags)
        if m:
            return m.group(1).strip()
    for key in fallback_keys:
        if item.get(key):
            return str(item[key]).strip()
    return ""

def build_raft_input_positive(item, noise_pool, n_noise=1):
    """RAFT格式正例：正确evidence作为结果1，加n_noise个干扰chunk"""
    contract = _extract_field(item,
        [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0),
         (r'来自合同[：:]\s*《?(.+?)》?', 0)],
        ["contract_name", "contract"])
    fragment = _extract_field(item,
        [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL),
         (r'【合同片段】\s*\n(.+?)(?:\n\n|\n【)', re.DOTALL)],
        ["clause", "fragment", "p"])
    question = _extract_field(item,
        [(r'用户提问[：:]\s*(.+?)$', re.DOTALL),
         (r'【用户提问】\s*\n(.+?)$', re.DOTALL)],
        ["question"])

    if not contract:
        contract = "未知合同"
    if not question:
        question = item.get("prompt", "")

    chunks = [f"检索结果1:\n合同名称：{contract}\n条款内容：{fragment}"]
    noises = random.sample([p for p in noise_pool if p[0] != contract], min(n_noise, len(noise_pool)))
    for i, (c_name, c_text) in enumerate(noises, 2):
        chunks.append(f"检索结果{i}:\n合同名称：{c_name}\n条款内容：{c_text}")

    return "\n\n".join(chunks) + f"\n\n问题：{question}"

def build_raft_input_distractor(item, noise_pool):
    """RAFT格式干扰样本：全部chunk来自不同合同"""
    contract = _extract_field(item,
        [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0),
         (r'来自合同[：:]\s*《?(.+?)》?', 0)],
        ["contract_name", "contract"])
    question = _extract_field(item,
        [(r'用户提问[：:]\s*(.+?)$', re.DOTALL),
         (r'【用户提问】\s*\n(.+?)$', re.DOTALL)],
        ["question"])

    if not contract:
        contract = "未知合同"
    if not question:
        question = item.get("prompt", "")

    noises = random.sample([p for p in noise_pool if p[0] != contract], min(2, len(noise_pool)))
    chunks = []
    for i, (c_name, c_text) in enumerate(noises, 1):
        chunks.append(f"检索结果{i}:\n合同名称：{c_name}\n条款内容：{c_text}")

    if not chunks:
        return None
    return "\n\n".join(chunks) + f"\n\n问题：{question}"

# ─── DeepSeek 评审 ────────────────────────────────────────────────────
def judge_clause(prompt, gold, outputs: dict):
    """评审 clause 质量，outputs = {model_name: output_text}"""
    model_names = list(outputs.keys())
    outputs_text = "\n\n".join([f"模型 {k}：\n{v}" for k, v in outputs.items()])
    judge_prompt = f"""
你是一名保险条款问答评估专家。请评估以下各模型回答的质量。

【用户问题及合同片段】
{prompt}

【标准答案】
{gold}

【各模型输出】
{outputs_text}

请对每个模型按以下5个维度打分（0-5分）：
1. correctness（正确性）：与标准答案方向是否一致
2. evidence_use（依据条款）：是否基于给定的合同片段作答
3. no_hallucination（无幻觉）：是否引入了合同片段外的内容
4. structure（格式结构）：输出格式是否规范
5. fidelity（遵守指令）：是否严格回答了问题本身

请按如下 JSON 格式输出，其中 scores 包含每个模型的各项分数：
{{
  "scores": {{
    "{model_names[0]}": {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}},
    {', '.join([f'"{n}": {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}}' for n in model_names[1:]])}
  }},
  "best": "最佳模型名称或tie",
  "comment": "一句话总结"
}}
"""
    return _call_deepseek(judge_prompt)

def judge_distractor(question_context, outputs: dict):
    """评审干扰样本下的拒答能力，outputs = {model_name: output_text}"""
    model_names = list(outputs.keys())
    outputs_text = "\n\n".join([f"模型 {k}：\n{v}" for k, v in outputs.items()])
    judge_prompt = f"""
你是一名保险问答系统评估专家。以下测试场景中，提供给模型的检索结果来自错误的保险合同（与用户询问的内容不相关）。

【测试输入（含干扰检索结果）】
{question_context}

【各模型输出】
{outputs_text}

正确的行为：模型应当识别到检索结果不匹配，明确告知用户无法作答，而不是编造答案。

请对每个模型按以下3个维度打分（0-5分）：
1. rejection_correctness（拒答正确性）：是否正确识别证据不足并拒绝作答（5=完全正确拒答，0=直接给出幻觉答案）
2. no_hallucination（无幻觉）：是否避免了用参数化知识编造答案（5=无幻觉）
3. explanation_quality（解释质量）：拒答时的说明是否清晰合理

请按如下 JSON 格式输出：
{{
  "scores": {{
    "{model_names[0]}": {{"rejection_correctness": x, "no_hallucination": x, "explanation_quality": x}},
    {', '.join([f'"{n}": {{"rejection_correctness": x, "no_hallucination": x, "explanation_quality": x}}' for n in model_names[1:]])}
  }},
  "best": "最佳模型名称或tie",
  "comment": "一句话总结"
}}
"""
    return _call_deepseek(judge_prompt)

def _call_deepseek(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_API, headers=headers, json=payload, timeout=60)
            text = resp.json()["choices"][0]["message"]["content"]
            # 提取JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"  API调用失败 (attempt {attempt+1}): {e}")
            time.sleep(2)
    return {"error": "API failed"}

# ─── 主评估流程 ───────────────────────────────────────────────────────
def run_inference_all_models(args, clause_data, hall_data, noise_pool, distractor_data, preloaded=None):
    """逐个加载模型，批量推理，逐个卸载"""
    empty = {"clause": {}, "hallucination": {}, "distractor": {}}
    all_outputs = {
        "baseline": dict(preloaded.get("baseline", empty)) if preloaded else {"clause": {}, "hallucination": {}, "distractor": {}},
        "stage2":   dict(preloaded.get("stage2",   empty)) if preloaded else {"clause": {}, "hallucination": {}, "distractor": {}},
        "stage3":   {"clause": {}, "hallucination": {}, "distractor": {}},  # stage3 总是重新推理
    }

    model_configs = [
        ("baseline", args.base,  None,        CLAUSE_SYSTEM, "clause", 400),
        ("stage2",   args.base,  args.lora2,  CLAUSE_SYSTEM, "clause", 400),
        ("stage3",   args.base,  args.lora3,  RAFT_SYSTEM,   "raft",   1024),
    ]

    for model_name, base, lora, sys_prompt, fmt, max_new in model_configs:
        # 如果已有输出则跳过
        if all(all_outputs[model_name][t] for t in ["clause", "hallucination", "distractor"]):
            print(f"► 跳过 {model_name}（已有缓存输出）")
            continue

        tok, model = load_model(base, lora, model_name)

        # === Clause 任务 ===
        print(f"\n  {model_name} → clause ({len(clause_data)} samples)...")
        for item in tqdm(clause_data, desc=f"{model_name}/clause"):
            if fmt == "raft":
                user = build_raft_input_positive(item, noise_pool, n_noise=1)
            else:
                user = build_clause_input(item)
            all_outputs[model_name]["clause"][str(item["id"])] = generate(model, tok, sys_prompt, user, max_new)

        # === Hallucination 任务（均用clause格式，evidence本身就不足）===
        print(f"\n  {model_name} → hallucination ({len(hall_data)} samples)...")
        for item in tqdm(hall_data, desc=f"{model_name}/hallucination"):
            all_outputs[model_name]["hallucination"][str(item["id"])] = generate(
                model, tok, sys_prompt, item["prompt"], max_new
            )

        # === Distractor 任务 ===
        print(f"\n  {model_name} → distractor ({len(distractor_data)} samples)...")
        for item in tqdm(distractor_data, desc=f"{model_name}/distractor"):
            user = build_raft_input_distractor(item, noise_pool)
            if user:
                all_outputs[model_name]["distractor"][str(item["id"])] = generate(
                    model, tok, RAFT_SYSTEM, user, max_new
                )

        unload_model(model)

    return all_outputs

def score_all(clause_data, hall_data, distractor_data, noise_pool, all_outputs):
    """用DeepSeek对所有样本评分"""
    results = {"clause": [], "hallucination": [], "distractor": []}

    # Clause
    print("\n► 评分 clause...")
    for item in tqdm(clause_data, desc="judge/clause"):
        iid = str(item["id"])
        outputs = {m: all_outputs[m]["clause"].get(iid, "") for m in all_outputs}
        judge = judge_clause(item["prompt"], item.get("completion", ""), outputs)
        results["clause"].append({"id": iid, "outputs": outputs, "judge": judge})

    # Hallucination
    print("\n► 评分 hallucination...")
    for item in tqdm(hall_data, desc="judge/hallucination"):
        iid = str(item["id"])
        outputs = {m: all_outputs[m]["hallucination"].get(iid, "") for m in all_outputs}
        judge = judge_clause(item["prompt"], item.get("completion", ""), outputs)
        results["hallucination"].append({"id": iid, "outputs": outputs, "judge": judge})

    # Distractor
    print("\n► 评分 distractor...")
    for item in tqdm(distractor_data, desc="judge/distractor"):
        iid = str(item["id"])
        context = build_raft_input_distractor(item, noise_pool)
        if not context:
            continue
        outputs = {m: all_outputs[m]["distractor"].get(iid, "") for m in all_outputs}
        judge = judge_distractor(context, outputs)
        results["distractor"].append({"id": iid, "outputs": outputs, "judge": judge})

    return results

def print_summary(results):
    for task, items in results.items():
        print(f"\n{'='*50}")
        print(f"任务: {task}  (n={len(items)})")
        print(f"{'='*50}")
        model_scores = {}
        for item in items:
            if "error" in item["judge"]:
                continue
            scores = item["judge"].get("scores", {})
            for model, dims in scores.items():
                if model not in model_scores:
                    model_scores[model] = {}
                for dim, val in dims.items():
                    model_scores[model].setdefault(dim, []).append(val)

        for model, dims in model_scores.items():
            avgs = {d: sum(v)/len(v) for d, v in dims.items()}
            total = sum(avgs.values()) / len(avgs)
            print(f"  {model:10s}  总分={total:.2f}  " +
                  "  ".join([f"{d}={v:.2f}" for d, v in avgs.items()]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",        required=True)
    parser.add_argument("--lora2",       required=True, help="stage2 LoRA路径")
    parser.add_argument("--lora3",       required=True, help="stage3 RAFT LoRA路径")
    parser.add_argument("--clause_data", required=True)
    parser.add_argument("--hall_data",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_clause",    type=int, default=30, help="clause评估样本数")
    parser.add_argument("--n_distractor",type=int, default=15, help="干扰样本数")
    parser.add_argument("--load_outputs", default=None, help="加载已有raw_outputs.json，仅重跑缺失模型")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据（支持单行JSONL和多行JSON对象拼接两种格式）
    def load_jsonl(path):
        text = open(path, encoding="utf-8").read().strip()
        decoder = json.JSONDecoder()
        items, idx = [], 0
        while idx < len(text):
            while idx < len(text) and text[idx] in " \n\r\t":
                idx += 1
            if idx >= len(text):
                break
            obj, end = decoder.raw_decode(text, idx)
            items.append(obj)
            idx = end
        return items

    clause_all = load_jsonl(args.clause_data)
    hall_data  = load_jsonl(args.hall_data)

    # 抽样
    clause_data     = random.sample(clause_all, min(args.n_clause, len(clause_all)))
    distractor_data = random.sample(clause_all, min(args.n_distractor, len(clause_all)))

    # 构建noise pool（从clause数据里提取合同名+片段，多模式兼容）
    noise_pool = []
    for item in clause_all:
        contract = _extract_field(item,
            [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0),
             (r'来自合同[：:]\s*《?(.+?)》?', 0)],
            ["contract_name", "contract"])
        fragment = _extract_field(item,
            [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL),
             (r'【合同片段】\s*\n(.+?)(?:\n\n|\n【)', re.DOTALL)],
            ["clause", "fragment", "p"])
        if contract and fragment:
            noise_pool.append((contract, fragment[:300]))

    print(f"评估数据: clause={len(clause_data)}, hallucination={len(hall_data)}, distractor={len(distractor_data)}")
    print(f"Noise pool: {len(noise_pool)} chunks\n")

    # 推理（可加载已有输出跳过完成的模型）
    preloaded = {}
    if args.load_outputs and os.path.exists(args.load_outputs):
        with open(args.load_outputs, encoding="utf-8") as f:
            preloaded = json.load(f)
        print(f"已加载缓存输出: {args.load_outputs}")
    all_outputs = run_inference_all_models(args, clause_data, hall_data, noise_pool, distractor_data,
                                           preloaded=preloaded)

    # 保存原始输出
    raw_path = os.path.join(args.output_dir, "raft_raw_outputs.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"\n原始输出已保存: {raw_path}")

    # 评分
    results = score_all(clause_data, hall_data, distractor_data, noise_pool, all_outputs)

    # 保存评分结果
    result_path = os.path.join(args.output_dir, "raft_eval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评分结果已保存: {result_path}")

    # 打印汇总
    print_summary(results)


if __name__ == "__main__":
    main()
