"""
Noise Filtering Evaluation (Exp-E)

测试 Stage-3 在混合输入（1个正确chunk + 2个干扰chunk）下能否：
  1. 正确识别含答案的chunk
  2. 不混入干扰chunk的内容
  3. 正确回答问题
  4. 明确引用来源合同

对比：Stage-3（RAFT_SYSTEM）vs Baseline（收到相同的多chunk输入）

运行方式：
  cd /home/jhqy/IRF/finetuning/qwen3
  source .venv/bin/activate
  python3 eval/scripts/eval_noise_filter.py \
    --base        Qwen3-4B \
    --lora3       outputs/qwen3-4b-lora-raft-v2 \
    --clause_data eval/data/clause.jsonl \
    --output_dir  eval/results/noise_filter \
    --n_clause    30
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

# Stage-3 uses RAFT system prompt; Baseline receives same multi-chunk input
# but with CLAUSE_SYSTEM (simulates a naive model seeing retrieval results)
RAFT_SYSTEM = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识，不得根据不相关的检索结果推断答案。\n"
    "请按以下规则处理：\n"
    "1. 检索结果包含相关信息 → 直接基于该信息作答，引用具体条款内容\n"
    "2. 检索结果不包含相关信息或与问题无关 → 输出：\n"
    "   '根据现有检索资料，暂无法提供关于该问题的准确信息。"
    "建议您直接联系保险公司客服或查阅完整合同原文，以获取准确的条款解释。'"
)  # v3: strict refusal + standard advisory message
CLAUSE_SYSTEM = (
    "你是一名严谨的保险条款问答助手，需要基于提供的合同片段回答用户问题。"
    "不得使用合同片段之外的知识，不得幻想不存在的条款。"
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


def build_multi_chunk_input(item, noise_pool, n_noise=2):
    """
    构造多chunk输入：1个正确chunk + n_noise个干扰chunk，共 n_noise+1 个chunk。
    chunk顺序随机打乱，使模型无法依赖位置偏见。
    返回 (input_text, correct_contract_name) 供 judge 使用。
    """
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

    correct_chunk = (contract, fragment)
    noises = random.sample(
        [p for p in noise_pool if p[0] != contract],
        min(n_noise, len(noise_pool))
    )

    # Shuffle: correct chunk may appear at any position
    all_chunks = [correct_chunk] + noises
    random.shuffle(all_chunks)

    parts = []
    for i, (c, t) in enumerate(all_chunks, 1):
        parts.append(f"检索结果{i}:\n合同名称：{c}\n条款内容：{t}")

    input_text = "\n\n".join(parts) + f"\n\n问题：{question}"
    return input_text, contract


# ── DeepSeek judge ────────────────────────────────────────────────────────────
def judge_noise_filter(model_outputs, gold, correct_contract):
    """
    评估两个模型在相同多chunk输入下的噪声过滤能力。
    4个维度（各5分，满分20）：
      1. evidence_selection:  是否选择了含答案的那个chunk（而非干扰chunk）
      2. no_contamination:    答案是否混入干扰chunk的内容
      3. correctness:         最终答案正确性
      4. source_attribution:  是否明确引用了正确合同名称
    """
    sections = []
    for name, (ctx, out) in model_outputs.items():
        sections.append(f"【模型 {name} 的输入（包含正确chunk和干扰chunk）】\n{ctx}\n\n【模型 {name} 的输出】\n{out}")

    prompt = f"""你是一名保险问答评估专家，专门测试模型在多个检索结果中识别正确证据的能力。

本次测试中，每个模型收到了相同的多chunk输入，其中只有来自「{correct_contract}」的chunk包含正确答案，其余是来自不同合同的干扰chunk。

【标准答案（供参考）】
{gold}

{chr(10).join(sections)}

请对每个模型按以下4个维度打分（0-5分）：
1. evidence_selection（证据选择）：是否引用了含答案的chunk（来自「{correct_contract}」），而非干扰chunk
2. no_contamination（无污染）：答案内容是否仅来自正确chunk，没有混入干扰chunk的内容
3. correctness（正确性）：最终答案与标准答案方向是否一致
4. source_attribution（来源归因）：是否在答案或推理中明确提到了正确合同名称「{correct_contract}」

请按如下 JSON 格式输出：
{{
  "scores": {{
    "baseline": {{"evidence_selection": x, "no_contamination": x, "correctness": x, "source_attribution": x}},
    "stage3":   {{"evidence_selection": x, "no_contamination": x, "correctness": x, "source_attribution": x}}
  }},
  "best": "baseline 或 stage3 或 tie",
  "comment": "一句话总结两个模型在噪声过滤上的差异"
}}"""

    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",         required=True)
    parser.add_argument("--lora3",        required=True)
    parser.add_argument("--clause_data",  required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--n_clause",     type=int, default=30)
    parser.add_argument("--load_outputs", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    clause_all  = load_jsonl(args.clause_data)
    clause_data = random.sample(clause_all, min(args.n_clause, len(clause_all)))

    # Build noise pool
    noise_pool = []
    for item in clause_all:
        contract = _extract_field(item,
            [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0)], ["contract_name"])
        fragment = _extract_field(item,
            [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL)], ["clause", "fragment"])
        if contract and fragment:
            noise_pool.append((contract, fragment[:300]))

    print(f"Clause samples: {len(clause_data)}, Noise pool: {len(noise_pool)}")

    # Build inputs (fixed seed, same for both models)
    sample_inputs = {}
    for item in clause_data:
        iid = str(item["id"])
        input_text, correct_contract = build_multi_chunk_input(item, noise_pool, n_noise=2)
        sample_inputs[iid] = {
            "input":            input_text,
            "correct_contract": correct_contract,
            "gold":             item.get("completion", ""),
        }

    # Inference
    raw_outputs_path = os.path.join(args.output_dir, "noise_filter_raw_outputs.json")
    if args.load_outputs and os.path.exists(args.load_outputs):
        with open(args.load_outputs) as f:
            all_outputs = json.load(f)
        print(f"已加载缓存: {args.load_outputs}")
    else:
        all_outputs = {"baseline": {}, "stage3": {}}

        # Baseline receives the same multi-chunk input
        tok, model = load_model(args.base, None, "baseline")
        for item in tqdm(clause_data, desc="baseline"):
            iid = str(item["id"])
            all_outputs["baseline"][iid] = generate(
                model, tok, CLAUSE_SYSTEM, sample_inputs[iid]["input"], 400
            )
        unload_model(model, tok)
        del model, tok

        # Save after baseline
        with open(raw_outputs_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"► baseline 输出已保存: {raw_outputs_path}")

        # Stage-3
        tok, model = load_model(args.base, args.lora3, "stage3")
        for item in tqdm(clause_data, desc="stage3"):
            iid = str(item["id"])
            all_outputs["stage3"][iid] = generate(
                model, tok, RAFT_SYSTEM, sample_inputs[iid]["input"], 512
            )
        unload_model(model, tok)
        del model, tok

        with open(raw_outputs_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"已保存全部推理输出: {raw_outputs_path}")

    # Judge
    print("\n► 噪声过滤评分中...")
    results = []
    for item in tqdm(clause_data, desc="judge"):
        iid = str(item["id"])
        si  = sample_inputs[iid]
        model_outputs = {
            "baseline": (si["input"], all_outputs["baseline"].get(iid, "")),
            "stage3":   (si["input"], all_outputs["stage3"].get(iid, "")),
        }
        judge = judge_noise_filter(model_outputs, si["gold"], si["correct_contract"])
        results.append({"id": iid, "correct_contract": si["correct_contract"], "judge": judge})

    # Save
    out_path = os.path.join(args.output_dir, "noise_filter_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存评分结果: {out_path}")

    # Aggregate
    dims = ["evidence_selection", "no_contamination", "correctness", "source_attribution"]
    scores = {"baseline": {d: [] for d in dims}, "stage3": {d: [] for d in dims}}

    for r in results:
        if "error" in r["judge"]:
            continue
        for model, dim_scores in r["judge"].get("scores", {}).items():
            for d, v in dim_scores.items():
                if isinstance(v, (int, float)):
                    scores[model].setdefault(d, []).append(v)

    print("\n=== Noise Filtering Results ===")
    for model in ["baseline", "stage3"]:
        avgs = {d: (sum(v)/len(v) if v else 0) for d, v in scores[model].items()}
        total = sum(avgs.values())
        pct   = total / (len(avgs) * 5) * 100
        print(f"  {model:10s}: total={total:.2f}/20 ({pct:.1f}%)  "
              + "  ".join(f"{d}={v:.2f}" for d, v in avgs.items()))

    best_counts = {"baseline": 0, "stage3": 0, "tie": 0}
    for r in results:
        b = r["judge"].get("best", "")
        if b in best_counts:
            best_counts[b] += 1
    print(f"\n  Best counts: {best_counts}")


if __name__ == "__main__":
    main()
