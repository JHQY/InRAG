"""
Insurance Answer Compliance Evaluation (Exp-F, revised)

目标：验证 SQL-CoT 训练（Stage-2→3）是否使模型输出更"合规"——
  更精确使用保险术语、更忠实于条款原文、措辞更专业。

数据来源：复用 fair_raw_outputs.json（无需新推理）
  - baseline: 收到单chunk格式输入，纯基础模型输出
  - stage3:   收到多chunk RAFT格式输入，微调后模型输出

评分维度（各5分，满分15）：
  1. answer_compliance    保险术语准确性：是否使用条款中的精确专业用语，而非口语化概括
  2. clause_fidelity      条款忠实度：关键表述是否来自检索内容，而非模型自行解释/改写
  3. response_formality   回答规范性：措辞是否符合保险问答正式规范（不夸大、不做承诺、说明局限性）

运行方式：
  cd /home/jhqy/IRF/finetuning/qwen3
  source .venv/bin/activate
  python3 eval/scripts/eval_cot_quality.py \
    --raw_outputs  eval/results/raft_eval_fair/fair_raw_outputs.json \
    --clause_data  eval/data/clause.jsonl \
    --output_dir   eval/results/cot_quality
"""

import argparse, json, os, random, re, time
import requests

random.seed(42)

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
with open(os.path.join(os.path.dirname(__file__), "../apikey")) as f:
    DEEPSEEK_KEY = f.read().strip()

DIMS = ["answer_compliance", "clause_fidelity", "response_formality"]


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


def extract_question(prompt):
    m = re.search(r'用户提问[：:]\s*(.+?)$', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def extract_clause_fragment(prompt):
    m = re.search(r'片段内容[：:]\s*(.+?)(?:\n\n|\n【)', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def clean_output(text):
    """去除 <think> 和 <Thought> 块，只保留实际答案部分"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<Thought>.*?</Thought>', '', text, flags=re.DOTALL).strip()
    return text.strip()


def judge_compliance(question, clause_fragment, baseline_out, stage3_out):
    prompt = f"""你是一名保险条款问答质量评估专家，专门评估模型回答是否符合保险行业的专业表达规范。

【用户提问】
{question}

【相关条款原文（供参考）】
{clause_fragment if clause_fragment else "（未能提取）"}

【模型 baseline 的回答】
{baseline_out}

【模型 stage3 的回答】
{stage3_out}

请对每个模型的回答按以下3个维度打分（0-5分）：

1. answer_compliance（术语合规性）：
   - 5分：使用了条款中的精确保险专业用语（如"保险责任期间"、"被保险人"等原文术语）
   - 3分：用语基本准确，偶有口语化
   - 1分：大量口语化概括，未使用条款术语
   - 0分：回答内容空洞或无实质内容

2. clause_fidelity（条款忠实度）：
   - 5分：关键结论直接引用或紧贴条款原文表述，而非模型自行总结/解释
   - 3分：部分引用原文，部分自行概括
   - 1分：完全是模型自己的解释，与原文措辞差异大
   - 0分：无法判断（回答过于简短或无关）

3. response_formality（表述规范性）：
   - 5分：措辞严谨，说明了依据，必要时说明局限性，不做超出条款的承诺
   - 3分：基本规范，略有不严谨之处
   - 1分：随意、口语化或做出超出条款的承诺/结论
   - 0分：不相关或无实质内容

请按如下 JSON 格式输出：
{{
  "scores": {{
    "baseline": {{"answer_compliance": x, "clause_fidelity": x, "response_formality": x}},
    "stage3":   {{"answer_compliance": x, "clause_fidelity": x, "response_formality": x}}
  }},
  "best": "baseline 或 stage3 或 tie",
  "comment": "一句话总结两个模型在合规表达上的差异"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_outputs",  default="eval/results/raft_eval_fair/fair_raw_outputs.json")
    parser.add_argument("--clause_data",  default="eval/data/clause.jsonl")
    parser.add_argument("--output_dir",   default="eval/results/cot_quality")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.raw_outputs, encoding="utf-8") as f:
        raw = json.load(f)

    clause_all  = load_jsonl(args.clause_data)
    clause_dict = {str(item["id"]): item for item in clause_all}

    baseline_outputs = raw.get("baseline", {})
    stage3_outputs   = raw.get("stage3", {})
    ids = list(stage3_outputs.keys())

    print(f"Loaded: {len(ids)} pairs")

    results = []
    for iid in ids:
        item            = clause_dict.get(iid, {})
        question        = extract_question(item.get("prompt", ""))
        clause_fragment = extract_clause_fragment(item.get("prompt", ""))

        baseline_ans = clean_output(baseline_outputs.get(iid, ""))
        stage3_ans   = clean_output(stage3_outputs.get(iid, ""))

        if not baseline_ans and not stage3_ans:
            print(f"  [SKIP] ID {iid}: both outputs empty")
            continue

        print(f"  Judging ID {iid}...")
        judge = judge_compliance(question, clause_fragment, baseline_ans, stage3_ans)
        results.append({
            "id":           iid,
            "question":     question[:100],
            "baseline_ans": baseline_ans[:200],
            "stage3_ans":   stage3_ans[:200],
            "judge":        judge,
        })
        time.sleep(0.5)

    out_path = os.path.join(args.output_dir, "compliance_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    # Aggregate
    scores = {"baseline": {d: [] for d in DIMS}, "stage3": {d: [] for d in DIMS}}
    best_counts = {"baseline": 0, "stage3": 0, "tie": 0}

    for r in results:
        if "error" in r.get("judge", {}):
            continue
        for model in ["baseline", "stage3"]:
            for d in DIMS:
                v = r["judge"].get("scores", {}).get(model, {}).get(d)
                if isinstance(v, (int, float)):
                    scores[model][d].append(v)
        b = r["judge"].get("best", "")
        if b in best_counts:
            best_counts[b] += 1

    print("\n=== Insurance Answer Compliance Results ===")
    for model in ["baseline", "stage3"]:
        avgs  = {d: (sum(v)/len(v) if v else 0) for d, v in scores[model].items()}
        total = sum(avgs.values())
        pct   = total / (len(DIMS) * 5) * 100
        print(f"  {model:10s}: total={total:.2f}/15 ({pct:.1f}%)  "
              + "  ".join(f"{d.split('_')[0]}={v:.2f}" for d, v in avgs.items()))

    print(f"\n  Best counts: {best_counts}")

    print("\n  Per-dimension delta (stage3 - baseline):")
    for d in DIMS:
        b_avg = sum(scores["baseline"][d])/len(scores["baseline"][d]) if scores["baseline"][d] else 0
        s_avg = sum(scores["stage3"][d])/len(scores["stage3"][d]) if scores["stage3"][d] else 0
        arrow = "▲" if s_avg > b_avg else ("▼" if s_avg < b_avg else "=")
        print(f"    {d:30s}: {b_avg:.2f} → {s_avg:.2f}  {arrow}{abs(s_avg-b_avg):.2f}")


if __name__ == "__main__":
    main()
