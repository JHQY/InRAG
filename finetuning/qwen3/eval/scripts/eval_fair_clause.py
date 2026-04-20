"""
Fair Clause Evaluation: Baseline vs Stage-3 (RAFT format)

解决 eval_raft.py 中的 judge 偏差问题：
- 原始eval：Stage-3 用多chunk输入推理，但 judge 只看单chunk context → evidence_use 维度有偏
- 本脚本：judge 收到的 context 与各模型实际推理的 input 一致

运行方式：
  cd /home/jhqy/IRF/finetuning/qwen3
  source .venv/bin/activate
  python3 eval/scripts/eval_fair_clause.py \
    --base     Qwen3-4B \
    --lora3    outputs/qwen3-4b-lora-raft-v2 \
    --clause_data eval/data/clause.jsonl \
    --output_dir  eval/results/raft_eval_fair \
    --n_clause 30
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
RAFT_SYSTEM = (
    "你是一个保险助手，请严格根据提供的检索结果回答用户问题，"
    "不得使用检索结果以外的任何知识，不得根据不相关的检索结果推断答案。\n"
    "请按以下规则处理：\n"
    "1. 检索结果包含相关信息 → 直接基于该信息作答，引用具体条款内容\n"
    "2. 检索结果不包含相关信息或与问题无关 → 输出：\n"
    "   '根据现有检索资料，暂无法提供关于该问题的准确信息。"
    "建议您直接联系保险公司客服或查阅完整合同原文，以获取准确的条款解释。'"
)  # v3: strict refusal + standard advisory message

# ── 模型工具 ─────────────────────────────────────────────────────────────────
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
INPUT_PATTERN = re.compile(
    r'合同是：(.+?)\n片段内容：(.+?)\n用户提问：(.+?)$', re.DOTALL
)

def _extract_field(item, prompt_patterns, fallback_keys):
    for pat, flags in prompt_patterns:
        m = re.search(pat, item.get("prompt", ""), flags)
        if m:
            return m.group(1).strip()
    for key in fallback_keys:
        if item.get(key):
            return str(item[key]).strip()
    return ""

def build_single_chunk_input(item):
    """Baseline格式：原始单chunk输入"""
    return item["prompt"]

def build_raft_multi_chunk_input(item, noise_pool, n_noise=1):
    """Stage-3格式：正确chunk + n_noise个干扰chunk"""
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
    for i, (c, t) in enumerate(noises, 2):
        chunks.append(f"检索结果{i}:\n合同名称：{c}\n条款内容：{t}")
    return "\n\n".join(chunks) + f"\n\n问题：{question}"

# ── DeepSeek judge ────────────────────────────────────────────────────────────
def judge_fair_clause(actual_input, gold, model_outputs):
    """
    公平judge：传入各模型实际看到的输入（而非统一的原始单chunk）
    model_outputs: {'baseline': (input_context, output_text), 'stage3': (input_context, output_text)}
    """
    sections = []
    for name, (ctx, out) in model_outputs.items():
        sections.append(f"【模型 {name} 的实际输入】\n{ctx}\n\n【模型 {name} 的输出】\n{out}")

    judge_prompt = f"""
你是一名保险条款问答评估专家。请分别评估以下模型在各自收到的输入下的回答质量。
注意：不同模型收到的上下文（检索结果）可能不同，请基于各自实际收到的输入进行评分。

【标准答案（供参考）】
{gold}

{chr(10).join(sections)}

请对每个模型按以下5个维度打分（0-5分）：
1. correctness（正确性）：与标准答案方向是否一致
2. evidence_use（依据条款）：是否基于实际收到的输入作答，而非参数化知识
3. no_hallucination（无幻觉）：是否引入了输入以外的内容
4. structure（格式结构）：输出格式是否规范
5. fidelity（遵守指令）：是否严格回答了问题本身

请按如下 JSON 格式输出：
{{
  "scores": {{
    "baseline": {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}},
    "stage3":   {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}}
  }},
  "best": "baseline 或 stage3 或 tie",
  "comment": "一句话总结"
}}
"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": judge_prompt}], "temperature": 0.1}
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

    # 加载数据
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
    clause_data = random.sample(clause_all, min(args.n_clause, len(clause_all)))

    # 构建noise pool
    noise_pool = []
    for item in clause_all:
        contract = _extract_field(item,
            [(r'合同[：:]\s*《?(.+?)》?\s*[\n】]', 0)], ["contract_name"])
        fragment = _extract_field(item,
            [(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', re.DOTALL)], ["clause", "fragment"])
        if contract and fragment:
            noise_pool.append((contract, fragment[:300]))

    print(f"Clause samples: {len(clause_data)}, Noise pool: {len(noise_pool)}")

    # 构造各模型的输入（保存供judge使用）
    sample_inputs = {}
    for item in clause_data:
        iid = str(item["id"])
        sample_inputs[iid] = {
            "baseline_input": build_single_chunk_input(item),
            "stage3_input":   build_raft_multi_chunk_input(item, noise_pool, n_noise=1),
            "gold": item.get("completion", ""),
        }

    # 推理
    raw_outputs_path = os.path.join(args.output_dir, "fair_raw_outputs.json")
    if args.load_outputs and os.path.exists(args.load_outputs):
        with open(args.load_outputs) as f:
            all_outputs = json.load(f)
        print(f"已加载缓存: {args.load_outputs}")
    else:
        all_outputs = {"baseline": {}, "stage3": {}}

        # Baseline
        if not all_outputs["baseline"]:
            tok, model = load_model(args.base, None, "baseline")
            for item in tqdm(clause_data, desc="baseline/clause"):
                iid = str(item["id"])
                all_outputs["baseline"][iid] = generate(
                    model, tok, CLAUSE_SYSTEM,
                    sample_inputs[iid]["baseline_input"], 400
                )
            unload_model(model, tok)
            del model, tok
            # 立即保存 baseline 输出，防止 OOM 崩溃后丢失
            with open(raw_outputs_path, "w", encoding="utf-8") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            print(f"► baseline 输出已保存: {raw_outputs_path}")

        # Stage-3
        tok, model = load_model(args.base, args.lora3, "stage3")
        for item in tqdm(clause_data, desc="stage3/clause"):
            iid = str(item["id"])
            all_outputs["stage3"][iid] = generate(
                model, tok, RAFT_SYSTEM,
                sample_inputs[iid]["stage3_input"], 1024
            )
        unload_model(model, tok)
        del model, tok

        with open(raw_outputs_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"已保存全部推理输出: {raw_outputs_path}")

    # Judge（公平：每个模型传入其实际input）
    print("\n► 公平评分中...")
    results = []
    for item in tqdm(clause_data, desc="judge/fair_clause"):
        iid = str(item["id"])
        si = sample_inputs[iid]
        model_outputs = {
            "baseline": (si["baseline_input"], all_outputs["baseline"].get(iid, "")),
            "stage3":   (si["stage3_input"],   all_outputs["stage3"].get(iid, "")),
        }
        judge = judge_fair_clause(None, si["gold"], model_outputs)
        results.append({"id": iid, "judge": judge})

    # 保存结果
    out_path = os.path.join(args.output_dir, "fair_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存评分结果: {out_path}")

    # 汇总
    scores = {"baseline": {}, "stage3": {}}
    for r in results:
        if "error" in r["judge"]:
            continue
        for model, dims in r["judge"].get("scores", {}).items():
            for dim, val in dims.items():
                if isinstance(val, (int, float)):
                    scores[model].setdefault(dim, []).append(val)

    print("\n=== Fair Clause Evaluation Results ===")
    for model in ["baseline", "stage3"]:
        avgs = {d: sum(v)/len(v) for d, v in scores[model].items()}
        total = sum(avgs.values())
        max_total = len(avgs) * 5
        pct = total / max_total * 100
        print(f"  {model:10s}: total={total:.2f}/{max_total} ({pct:.1f}%)  "
              + "  ".join(f"{d}={v:.2f}" for d, v in avgs.items()))

if __name__ == "__main__":
    main()
