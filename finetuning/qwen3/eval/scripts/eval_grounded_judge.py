"""
Grounded Judge Re-evaluation

修复 baseline 假高问题：在常规评分前加一个「溯源检查」阶段：
  1. 列出模型输出中的具体声明（数字、条件、比例、程序）
  2. 逐条核查是否能在提供的证据原文中找到依据
  3. 存在无法溯源的声明 → no_hallucination=0, evidence_use=0（强制）
  4. 在溯源结果基础上完成常规评分

复用 raft_eval_v4/raft_raw_outputs.json，无需重新推理。

运行方式：
  cd /home/jhqy/IRF/finetuning/qwen3
  source .venv/bin/activate
  python3 eval/scripts/eval_grounded_judge.py \
    --raw_outputs eval/results/raft_eval_v4/raft_raw_outputs.json \
    --hall_data   eval/data/hallucination_eval.jsonl \
    --output_dir  eval/results/grounded_judge
"""

import argparse, json, os, re, time
import requests

DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
with open(os.path.join(os.path.dirname(__file__), "../apikey")) as f:
    DEEPSEEK_KEY = f.read().strip()


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


def _call_deepseek(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1}
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


def grounded_judge_hallucination(evidence_text, gold, model_outputs):
    """
    两阶段 judge for hallucination suppression task.
    evidence_text: 提供给模型的实际证据（不足以完整回答）
    model_outputs: {'baseline': text, 'stage3': text}
    """
    outputs_text = "\n\n".join(f"【模型 {k} 的输出】\n{v}" for k, v in model_outputs.items())

    prompt = f"""你是一名严格的保险问答评估专家，专门识别模型是否使用了证据以外的知识。

本次测试的证据片段（故意不足以完整回答问题）：
【证据原文】
{evidence_text}

【标准答案（供参考）】
{gold}

{outputs_text}

请执行两阶段评估：

=== 第一阶段：溯源检查 ===
对每个模型的输出，列出其中所有具体声明（如具体数字、条件描述、程序步骤、资格要求等）。
然后判断：每条声明是否可以在上方「证据原文」中找到直接依据？

如果模型输出包含「证据原文」中没有的具体信息 → grounded=false
如果模型输出仅基于证据内容或表示无法作答 → grounded=true

=== 第二阶段：评分 ===
在溯源结果基础上，对每个模型按5个维度打分（0-5）：
1. correctness：与标准答案方向是否一致
2. evidence_use：是否基于证据作答（若 grounded=false 则此项强制为0）
3. no_hallucination：是否避免了证据外内容（若 grounded=false 则此项强制为0）
4. structure：输出格式是否规范
5. fidelity：是否回答了问题本身（拒答并给出合理说明也算）

请按如下 JSON 格式输出：
{{
  "grounding": {{
    "baseline": {{
      "grounded": true或false,
      "unverifiable_claims": ["声明1", "声明2"]
    }},
    "stage3": {{
      "grounded": true或false,
      "unverifiable_claims": []
    }}
  }},
  "scores": {{
    "baseline": {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}},
    "stage3":   {{"correctness": x, "evidence_use": x, "no_hallucination": x, "structure": x, "fidelity": x}}
  }},
  "comment": "一句话总结两模型在证据使用上的差异"
}}"""
    return _call_deepseek(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_outputs", default="eval/results/raft_eval_v4/raft_raw_outputs.json")
    parser.add_argument("--stage3_raw",  default=None,
                        help="可选：单独指定 Stage-3 的 raw outputs（dict {id: output_str}），"
                             "用于评测不同 prompt 版本的 Stage-3 输出。"
                             "格式：{\"hallucination\": {\"1\": \"...\", ...}}")
    parser.add_argument("--hall_data",   default="eval/data/hallucination_eval.jsonl")
    parser.add_argument("--output_dir",  default="eval/results/grounded_judge")
    parser.add_argument("--output_file", default="grounded_hall_results.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.raw_outputs) as f:
        raw = json.load(f)

    hall_data = load_jsonl(args.hall_data)
    hall_dict = {str(item["id"]): item for item in hall_data}

    baseline_hall = raw["baseline"]["hallucination"]

    if args.stage3_raw:
        with open(args.stage3_raw) as f:
            s3_raw = json.load(f)
        stage3_hall = s3_raw["hallucination"]
    else:
        stage3_hall = raw["stage3"]["hallucination"]

    print(f"Hallucination samples: {len(hall_data)}")
    print("Running grounded judge...\n")

    results = []
    for iid in baseline_hall:
        item = hall_dict.get(iid, {})
        evidence = item.get("prompt", "")
        gold     = item.get("completion", "")

        model_outputs = {
            "baseline": baseline_hall.get(iid, ""),
            "stage3":   stage3_hall.get(iid, ""),
        }

        print(f"  Judging ID {iid}...")
        judge = grounded_judge_hallucination(evidence, gold, model_outputs)
        results.append({
            "id":    iid,
            "judge": judge,
            "outputs": {k: v[:200] for k, v in model_outputs.items()},
        })
        time.sleep(0.5)

    # Save
    out_path = os.path.join(args.output_dir, args.output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")

    # Aggregate
    dims = ["correctness", "evidence_use", "no_hallucination", "structure", "fidelity"]
    scores    = {"baseline": {d: [] for d in dims}, "stage3": {d: [] for d in dims}}
    grounded  = {"baseline": 0, "stage3": 0}

    for r in results:
        if "error" in r.get("judge", {}):
            continue
        for model in ["baseline", "stage3"]:
            if r["judge"].get("grounding", {}).get(model, {}).get("grounded", True):
                grounded[model] += 1
            for d in dims:
                v = r["judge"].get("scores", {}).get(model, {}).get(d)
                if isinstance(v, (int, float)):
                    scores[model][d].append(v)

    n = len([r for r in results if "error" not in r.get("judge", {})])
    print(f"\n=== Grounded Hallucination Judge (n={n}) ===")
    print(f"\n  溯源通过率（grounded=true）:")
    for model in ["baseline", "stage3"]:
        print(f"    {model:10s}: {grounded[model]}/{n} ({grounded[model]/n*100:.0f}%)")

    print(f"\n  评分结果（满分25）:")
    for model in ["baseline", "stage3"]:
        avgs  = {d: (sum(v)/len(v) if v else 0) for d, v in scores[model].items()}
        total = sum(avgs.values())
        pct   = total / 25 * 100
        print(f"    {model:10s}: {total:.2f}/25 ({pct:.1f}%)  "
              + "  ".join(f"{d[:6]}={v:.2f}" for d, v in avgs.items()))

    src_note = f"(stage3_raw={args.stage3_raw})" if args.stage3_raw else "(来自 raft_eval_v4)"
    print(f"\n  [对比] 原始 judge 结果 {src_note}:")
    print(f"    baseline : 85.2%  (无溯源检查)")
    print(f"    stage3   : 72.0%  (无溯源检查, v1) / 83.6% (v3)")

    # Show unverifiable claims
    print(f"\n=== Baseline 无法溯源的声明样例 ===")
    for r in results[:3]:
        claims = r["judge"].get("grounding", {}).get("baseline", {}).get("unverifiable_claims", [])
        if claims:
            print(f"  ID={r['id']}: {claims}")


if __name__ == "__main__":
    main()
