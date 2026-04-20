"""
Abstention Re-test: v1 / v2 / v3 prompt variants

v1: 严格拒答（原版）
v2: 分级响应（含"部分相关→尽量作答"，已验证导致distractor幻觉飙升）
v3: 严格拒答 + 标准建议语（去掉第2条，拒答时输出固定指引）

复用 raft_eval_v4 中的 baseline 输出，只重跑 Stage-3。

运行方式：
  cd /home/jhqy/IRF/finetuning/qwen3
  source .venv/bin/activate
  python3 eval/scripts/eval_abstention_v2.py \
    --base        Qwen3-4B \
    --lora3       outputs/qwen3-4b-lora-raft-v2 \
    --hall_data   eval/data/hallucination_eval.jsonl \
    --clause_data eval/data/clause.jsonl \
    --old_outputs eval/results/raft_eval_v4/raft_raw_outputs.json \
    --output_dir  eval/results/abstention_v2 \
    --prompt_version v3
"""

import sys
sys.modules["awq"] = None
sys.modules["awq.quantize"] = None

import argparse, gc, json, os, random, re, time
import requests, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

random.seed(42)

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
with open(os.path.join(os.path.dirname(__file__), "../apikey")) as f:
    DEEPSEEK_KEY = f.read().strip()

CLAUSE_SYSTEM = (
    "你是一名严谨的保险条款问答助手，需要基于提供的合同片段回答用户问题。"
    "不得使用合同片段之外的知识，不得幻想不存在的条款。"
)

# 旧版（对照组）
RAFT_SYSTEM_V1 = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识。"
    "如果检索结果不包含回答所需信息，请明确告知用户无法作答。"
)

# v2（分级响应，已验证导致distractor幻觉飙升，保留供对照）
RAFT_SYSTEM_V2 = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识。\n"
    "请按以下优先级处理：\n"
    "1. 检索结果包含相关信息 → 直接基于该信息作答，引用具体条款内容\n"
    "2. 检索结果包含部分相关信息 -> 基于现有内容尽量作答，并注明'根据现有资料，如需完整信息请查阅合同原文'\n"
    "3. 检索结果与问题完全无关 → 告知用户无法作答，并说明原因"
)

# v3：严格拒答 + 标准建议语（去掉"部分作答"中间层）
RAFT_SYSTEM_V3 = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识，不得根据不相关的检索结果推断答案。\n"
    "请按以下规则处理：\n"
    "1. 检索结果包含相关信息 → 直接基于该信息作答，引用具体条款内容\n"
    "2. 检索结果不包含相关信息或与问题无关 → 输出：\n"
    "   '根据现有检索资料，暂无法提供关于该问题的准确信息。"
    "建议您直接联系保险公司客服或查阅完整合同原文，以获取准确的条款解释。'"
)


# ── 模型工具 ──────────────────────────────────────────────────────────────────
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


def unload_model(model, tok=None):
    del model
    if tok is not None:
        del tok
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


def generate(model, tok, system_prompt, user_content, max_new=512):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=0.2, top_p=0.9, do_sample=True,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return text.strip()


# ── 输入构造 ──────────────────────────────────────────────────────────────────
def _extract_field(item, prompt_patterns, fallback_keys):
    for pat, flags in prompt_patterns:
        m = re.search(pat, item.get("prompt", ""), flags)
        if m:
            return m.group(1).strip()
    for key in fallback_keys:
        if item.get(key):
            return str(item[key]).strip()
    return ""


def build_raft_distractor_input(item, noise_pool):
    """全错chunk输入（两个来自不同合同的chunk，无正确答案）"""
    contract = _extract_field(item,
        [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0)], ["contract_name"])
    question = _extract_field(item,
        [(r'用户提问[：:]\s*(.+?)$', re.DOTALL)], ["question"])
    if not contract:
        contract = "未知合同"
    if not question:
        question = item.get("prompt", "")

    noises = random.sample([p for p in noise_pool if p[0] != contract], min(2, len(noise_pool)))
    if not noises:
        return None
    chunks = [f"检索结果{i}:\n合同名称：{c}\n条款内容：{t}" for i, (c, t) in enumerate(noises, 1)]
    return "\n\n".join(chunks) + f"\n\n问题：{question}"


def build_raft_hall_input(item):
    """幻觉任务：将单chunk prompt 包装成RAFT格式（evidence本身就不足）"""
    contract = _extract_field(item,
        [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0)], ["contract_name"])
    fragment = _extract_field(item,
        [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL)], ["clause", "fragment"])
    question = _extract_field(item,
        [(r'用户提问[：:]\s*(.+?)$', re.DOTALL)], ["question"])
    if not contract:
        contract = "未知合同"
    if not question:
        question = item.get("prompt", "")
    chunk = f"检索结果1:\n合同名称：{contract}\n条款内容：{fragment}"
    return chunk + f"\n\n问题：{question}"


# ── Judge ─────────────────────────────────────────────────────────────────────
def judge_hallucination(prompt_ctx, gold, outputs):
    outputs_text = "\n\n".join(f"模型 {k}：\n{v}" for k, v in outputs.items())
    judge_prompt = f"""
你是一名保险问答系统评估专家。以下场景中，提供给模型的证据不足以完整回答问题。

【用户问题及条款片段】
{prompt_ctx}

【标准答案（参考）】
{gold}

【各模型输出】
{outputs_text}

请对每个模型按以下5个维度打分（0-5分）：
1. correctness（正确性）：与标准答案方向是否一致
2. evidence_use（依据条款）：是否基于给定片段作答，而非参数化知识
3. no_hallucination（无幻觉）：是否避免了编造条款以外的内容
4. structure（格式结构）：输出格式是否规范
5. fidelity（遵守指令）：是否严格回答了问题本身

请按如下 JSON 格式输出：
{{
  "scores": {{
    {', '.join(f'"{k}": {{"correctness": 0, "evidence_use": 0, "no_hallucination": 0, "structure": 0, "fidelity": 0}}' for k in outputs)}
  }},
  "best": "最佳模型名称或tie",
  "comment": "一句话总结"
}}"""
    return _call_deepseek(judge_prompt)


def judge_distractor(question_ctx, outputs):
    outputs_text = "\n\n".join(f"模型 {k}：\n{v}" for k, v in outputs.items())
    judge_prompt = f"""
你是一名保险问答系统评估专家。以下场景中，提供给模型的检索结果来自错误的保险合同（与问题无关）。

【测试输入（含干扰检索结果）】
{question_ctx}

【各模型输出】
{outputs_text}

正确行为：模型应识别检索结果不匹配，拒绝作答而非编造内容。

请对每个模型按以下3个维度打分（0-5分）：
1. rejection_correctness（拒答正确性）：是否正确识别证据不足并拒绝（5=完全正确拒答，0=直接给出幻觉答案）
2. no_hallucination（无幻觉）：是否避免用参数化知识编造答案
3. explanation_quality（解释质量）：拒答时说明是否清晰合理

请按如下 JSON 格式输出：
{{
  "scores": {{
    {', '.join(f'"{k}": {{"rejection_correctness": 0, "no_hallucination": 0, "explanation_quality": 0}}' for k in outputs)}
  }},
  "best": "最佳模型名称或tie",
  "comment": "一句话总结"
}}"""
    return _call_deepseek(judge_prompt)


def _call_deepseek(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_API, headers=headers, json=payload, timeout=60)
            text = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"  API失败 (attempt {attempt+1}): {e}")
            time.sleep(2)
    return {"error": "API failed"}


# ── 主流程 ────────────────────────────────────────────────────────────────────
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",         required=True)
    parser.add_argument("--lora3",        required=True)
    parser.add_argument("--hall_data",    required=True)
    parser.add_argument("--clause_data",  required=True)
    parser.add_argument("--old_outputs",  required=True, help="raft_eval_v4 raw outputs for baseline reuse")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--n_distractor",   type=int, default=15)
    parser.add_argument("--prompt_version", default="v3", choices=["v1", "v2", "v3"])
    args = parser.parse_args()

    prompt_map = {"v1": RAFT_SYSTEM_V1, "v2": RAFT_SYSTEM_V2, "v3": RAFT_SYSTEM_V3}
    active_prompt = prompt_map[args.prompt_version]
    print(f"Using prompt: {args.prompt_version}")

    os.makedirs(args.output_dir, exist_ok=True)

    hall_data    = load_jsonl(args.hall_data)
    clause_all   = load_jsonl(args.clause_data)
    distractor_data = random.sample(clause_all, min(args.n_distractor, len(clause_all)))

    # Build noise pool
    noise_pool = []
    for item in clause_all:
        contract = _extract_field(item, [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0)], ["contract_name"])
        fragment = _extract_field(item, [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL)], ["clause", "fragment"])
        if contract and fragment:
            noise_pool.append((contract, fragment[:300]))

    print(f"Hall: {len(hall_data)}, Distractor: {len(distractor_data)}, Noise pool: {len(noise_pool)}")

    # Load baseline outputs from raft_eval_v4 (no need to re-run)
    with open(args.old_outputs) as f:
        old = json.load(f)
    baseline_hall = old["baseline"]["hallucination"]
    baseline_dist = old["baseline"]["distractor"]
    print(f"Loaded baseline outputs: hall={len(baseline_hall)}, distractor={len(baseline_dist)}")

    # Build inputs for Stage-3 v2
    hall_inputs = {str(item["id"]): build_raft_hall_input(item) for item in hall_data}
    dist_inputs = {}
    for item in distractor_data:
        inp = build_raft_distractor_input(item, noise_pool)
        if inp:
            dist_inputs[str(item["id"])] = inp

    # Inference: Stage-3 with selected prompt version
    raw_path = os.path.join(args.output_dir, f"abstention_{args.prompt_version}_raw.json")
    stage3_v2 = {"hallucination": {}, "distractor": {}}

    tok, model = load_model(args.base, args.lora3, "stage3-v2")

    print("\n► Stage-3 v2: hallucination...")
    for item in tqdm(hall_data, desc="hall/stage3-v2"):
        iid = str(item["id"])
        stage3_v2["hallucination"][iid] = generate(
            model, tok, active_prompt, hall_inputs[iid], 512
        )

    print(f"\n► Stage-3 {args.prompt_version}: distractor...")
    for item in tqdm(distractor_data, desc=f"dist/stage3-{args.prompt_version}"):
        iid = str(item["id"])
        if iid in dist_inputs:
            stage3_v2["distractor"][iid] = generate(
                model, tok, active_prompt, dist_inputs[iid], 512
            )

    unload_model(model, tok)
    del model, tok

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(stage3_v2, f, ensure_ascii=False, indent=2)
    print(f"推理输出已保存: {raw_path}")

    # Check refusal rates
    refusal_pats = ['无法确认', '无法回答', '暂无法', '建议查阅', '无法作答', '不包含', '无相关']
    for task in ["hallucination", "distractor"]:
        n = len(stage3_v2[task])
        ref = sum(1 for v in stage3_v2[task].values() if any(p in v for p in refusal_pats))
        print(f"Stage-3-v2 {task}: refusal rate = {ref}/{n} ({ref/n*100:.0f}%)")

    # Judge
    print("\n► 评分 hallucination...")
    hall_results = []
    for item in tqdm(hall_data, desc="judge/hall"):
        iid = str(item["id"])
        outputs = {
            "baseline":    baseline_hall.get(iid, ""),
            "stage3_v2":   stage3_v2["hallucination"].get(iid, ""),
        }
        judge = judge_hallucination(item["prompt"], item.get("completion", ""), outputs)
        hall_results.append({"id": iid, "judge": judge})

    print("\n► 评分 distractor...")
    dist_results = []
    for item in tqdm(distractor_data, desc="judge/dist"):
        iid = str(item["id"])
        if iid not in dist_inputs:
            continue
        outputs = {
            "baseline":    baseline_dist.get(iid, ""),
            "stage3_v2":   stage3_v2["distractor"].get(iid, ""),
        }
        judge = judge_distractor(dist_inputs[iid], outputs)
        dist_results.append({"id": iid, "judge": judge})

    # Save
    results = {"hallucination": hall_results, "distractor": dist_results}
    out_path = os.path.join(args.output_dir, f"abstention_{args.prompt_version}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    # Summary
    print("\n=== Abstention v2 Results ===")
    task_dims = {
        "hallucination": ["correctness", "evidence_use", "no_hallucination", "structure", "fidelity"],
        "distractor":    ["rejection_correctness", "no_hallucination", "explanation_quality"],
    }
    task_max = {"hallucination": 25, "distractor": 15}

    for task, items in [("hallucination", hall_results), ("distractor", dist_results)]:
        dims = task_dims[task]
        scores = {"baseline": {d: [] for d in dims}, "stage3_v2": {d: [] for d in dims}}
        for r in items:
            if "error" in r.get("judge", {}):
                continue
            for model in ["baseline", "stage3_v2"]:
                for d in dims:
                    v = r["judge"].get("scores", {}).get(model, {}).get(d)
                    if isinstance(v, (int, float)):
                        scores[model][d].append(v)

        print(f"\n  [{task}] max={task_max[task]}")
        for model in ["baseline", "stage3_v2"]:
            avgs  = {d: (sum(v)/len(v) if v else 0) for d, v in scores[model].items()}
            total = sum(avgs.values())
            pct   = total / task_max[task] * 100
            print(f"    {model:15s}: {total:.2f}/{task_max[task]} ({pct:.1f}%)  "
                  + "  ".join(f"{d[:6]}={v:.2f}" for d, v in avgs.items()))

    # Compare with old Stage-3 v1 scores (from plan memory)
    print("\n  [对比参考] raft_eval_v4 中 Stage-3-v1 旧分数：")
    print("    hallucination: 72.0%  distractor: 58.2%")
    print("    baseline:      85.2%  baseline:   68.9%")


if __name__ == "__main__":
    main()
