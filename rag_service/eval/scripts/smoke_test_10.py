#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10-prompt end-to-end smoke test for IRAG + RAFT pipeline.

Output: eval/results/smoke_test_10.json
"""

import json
import os
import re
import time
import http.client
from datetime import datetime
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────
IRAG_HOST = "127.0.0.1"
IRAG_PORT = 8000
RAFT_HOST = "127.0.0.1"
RAFT_PORT = 8001
TOP_K = 8
MODE = "raft"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "smoke_test_15_v5.json"  # judge=DeepSeek

# ── 15 條測試 Prompt（香港保險用語）────────────────────────────────
TEST_PROMPTS = [
    {
        "id": 1,
        "prompt": "保誠保險計劃的冷靜期（猶豫期）是多少天？如何在冷靜期內取消保單？",
        "category": "定義類-冷靜期",
        "expected": "有答案",
        "note": "KB中有保誠冷靜期21天條款",
    },
    {
        "id": 2,
        "prompt": "如果受保人在保單生效後1年內自殺，保誠人壽會否賠付身故賠償？",
        "category": "邊界條款-自殺條款",
        "expected": "有答案",
        "note": "KB中有保誠1年自殺不賠條款",
    },
    {
        "id": 3,
        "prompt": "永明危疾保的等候期是多少天？哪些疾病有較長的等候期？",
        "category": "定義類-等候期",
        "expected": "有答案",
        "note": "KB中有永明危疾90日/180日等候期條款",
    },
    {
        "id": 4,
        "prompt": "友邦特級健康之寶住院計劃是否保證終身續保？保費如何釐定？",
        "category": "續保條款",
        "expected": "有答案",
        "note": "KB中有AIA SGHR2保證終身續保條款",
    },
    {
        "id": 5,
        "prompt": "宏利自願醫保計劃屬於香港自願醫保計劃（VHIS）認可產品嗎？有哪些主要保障範圍？",
        "category": "產品資格-VHIS",
        "expected": "有答案",
        "note": "改為問VHIS認可資格及保障範圍，避免兩款計劃對比",
    },
    {
        "id": 6,
        "prompt": "保誠隽逸人生延期年金計劃是否屬於合資格延期年金保單（QDAP）？有何稅務優惠？",
        "category": "稅務/年金",
        "expected": "有答案",
        "note": "KB中有保誠QDAP年金計劃條款",
    },
    {
        "id": 7,
        "prompt": "富衛自主保定期壽險的保障年期有哪些選擇？受保人身故後賠償金額如何計算？",
        "category": "身故保障-定期壽險",
        "expected": "有答案",
        "note": "改為問保障年期選擇，更易在KB中命中FWD條款",
    },
    {
        "id": 8,
        "prompt": "中銀家務助理保障計劃的意外死亡保障金額是多少？受益人如何申請賠償？",
        "category": "家傭保險-意外保障",
        "expected": "有答案",
        "note": "改為問意外死亡保障，更易命中BOC家傭計劃條款",
    },
    {
        "id": 9,
        "prompt": "滙豐靈活醫保鑽級計劃設有哪些自付費選項？自付費對保費有何影響？",
        "category": "數值類-自付費",
        "expected": "有答案",
        "note": "KB中有HSBC保費表及自付費選項",
    },
    {
        "id": 10,
        "prompt": "永明人壽危疾保障計劃的身故還原保障是什麼？受保人身故後保單會如何處理？",
        "category": "身故還原保障",
        "expected": "有答案",
        "note": "KB中有永明危疾身故還原保障條款",
    },
    {
        "id": 11,
        "prompt": "安盛（AXA）意外保計劃有哪些不保事項？從事哪些危險活動不在承保範圍內？",
        "category": "不保事項-意外險",
        "expected": "有答案",
        "note": "KB中有AXA意外保不保事項條款",
    },
    {
        "id": 12,
        "prompt": "恒生人壽保障計劃的身故賠償金額最低要求是多少？保障期限是否為終身？",
        "category": "身故保障-壽險",
        "expected": "有答案",
        "note": "KB中有恒生life身故保障及最低保額條款",
    },
    {
        "id": 13,
        "prompt": "友邦特級健康之寶的住院病房級別有何限制？私家醫院住院是否有額外扣減？",
        "category": "住院保障-病房限制",
        "expected": "有答案",
        "note": "KB中有AIA SGHR2住院賠償及病房條款",
    },
    {
        "id": 14,
        "prompt": "保誠危疾保障計劃對原位癌（非侵入性癌症）是否有保障？賠償比例是多少？",
        "category": "危疾定義-原位癌",
        "expected": "有答案",
        "note": "KB中有保誠危疾計劃條款，預期含原位癌相關定義",
    },
    {
        "id": 15,
        "prompt": "香港保險合約中的「不可爭議條款」是什麼意思？保單生效多久後保險公司不可以不誠實陳述為由拒絕理賠？",
        "category": "邊界-拒答測試",
        "expected": "拒答（證據不足）",
        "note": "一般性法律條款，KB未必有具體條文，測試模型是否懂得拒答",
    },
]

# ── 工具函数 ──────────────────────────────────────────────────────

def call_irag(prompt: str, top_k: int = TOP_K, mode: str = MODE) -> dict:
    """调用 IRAG /api/ask，返回完整 JSON 响应。"""
    conn = http.client.HTTPConnection(IRAG_HOST, IRAG_PORT, timeout=300)
    payload = json.dumps({"question": prompt, "top_k": top_k, "mode": mode}, ensure_ascii=False).encode("utf-8")
    conn.request("POST", "/api/ask", payload, {"Content-Type": "application/json; charset=utf-8"})
    res = conn.getresponse()
    return json.loads(res.read().decode("utf-8"))


DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def call_deepseek(messages: list, max_tokens: int = 512) -> str:
    """调用 DeepSeek API 作为 judge。"""
    conn = http.client.HTTPSConnection(DEEPSEEK_HOST, timeout=60)
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
    }
    conn.request("POST", "/chat/completions", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_raft_output(raw: str) -> dict:
    """从 RAFT 模型输出中解析 <Thought> CoT 和最终答案。"""
    # 去除 <think>...</think>（空的内部思考块）
    raw_clean = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    # 提取 <Thought>...</Thought>
    cot_match = re.search(r"<Thought>([\s\S]*?)</Thought>", raw_clean)
    cot = cot_match.group(1).strip() if cot_match else ""

    # 最终答案 = Thought 块之后的所有内容
    if cot_match:
        final = raw_clean[cot_match.end():].strip()
    else:
        final = raw_clean

    return {"cot": cot, "final_answer": final, "raw": raw_clean}


EVAL_SYSTEM = """你是一名保险RAG系统评估专家。只输出JSON，不要输出任何其他内容。

请根据以下信息评估模型回答的质量：
1. 检索到的参考资料（知识库片段）
2. 模型的推理过程（CoT）
3. 模型的最终答案

评估维度（每项0-3分）：
- retrieval_relevance：检索结果与问题是否相关？（0=完全无关，3=高度相关）
- faithfulness：答案是否严格来自检索证据，没有幻觉？宽松标准：答案中的信息只要能在任一参考片段中找到依据即算忠实。（0=大量幻觉，3=完全忠实）
- rejection_quality：若模型拒答，是否合理？若有证据却拒答扣分。（0=不合理，3=合理）
- completeness：答案是否充分回答了问题？（0=完全未答，3=完整）

只输出如下JSON，不要有任何前缀、解释或markdown：
{"retrieval_relevance": 0, "faithfulness": 0, "rejection_quality": 0, "completeness": 0, "total": 0, "verdict": "PASS", "reason": ""}"""


def evaluate(prompt: str, refs: list, cot: str, final_answer: str) -> dict:
    """调用 DeepSeek API 作为 judge 评估结果。"""
    ref_text = "\n\n".join(
        f"[参考{i+1}] {r['metadata'].get('source','?').split('/')[-1].rsplit('.',1)[0]}\n{r['text'][:500]}"
        for i, r in enumerate(refs)
    )
    user_msg = f"""【问题】
{prompt}

【检索到的参考资料】
{ref_text}

【模型推理过程（CoT）】
{cot if cot else "（无）"}

【模型最终答案】
{final_answer}"""

    messages = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = call_deepseek(messages, max_tokens=300)

    # 提取最后一个完整 JSON 对象（避免模型输出前缀文字）
    json_matches = list(re.finditer(r"\{[\s\S]*?\}", raw))
    for m in reversed(json_matches):
        try:
            result = json.loads(m.group())
            if "verdict" in result and "total" in result:
                return result
        except Exception:
            pass
    return {
        "retrieval_relevance": -1,
        "faithfulness": -1,
        "rejection_quality": -1,
        "completeness": -1,
        "total": -1,
        "verdict": "ERROR",
        "reason": f"评估解析失败: {raw[:200]}",
    }


# ── 主流程 ────────────────────────────────────────────────────────

def run_smoke_test():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"{'='*70}")
    print(f"  IRAG + RAFT 端到端 Smoke Test v5 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*70}")

    for item in TEST_PROMPTS:
        tid = item["id"]
        prompt = item["prompt"]
        print(f"\n[{tid}/15] {item['category']} — {prompt[:50]}...")

        # 1. RAG 检索 + 模型推理
        t0 = time.time()
        try:
            irag_resp = call_irag(prompt)
        except Exception as e:
            print(f"  ✗ IRAG 调用失败: {e}")
            results.append({"id": tid, "error": str(e)})
            continue
        latency_total = round(time.time() - t0, 2)

        refs = irag_resp.get("refs", [])
        raw_answer = irag_resp.get("answer", "")

        # 2. 解析 CoT / 最终答案
        parsed = parse_raft_output(raw_answer)

        # 3. 检索结果元数据
        def _parse_source(source: str) -> dict:
            """从 source 路径提取 company / category / product_name。
            路径格式: sourcepdf/{company}/{category}/{filename}.pdf
            """
            parts = source.replace("\\", "/").split("/")
            company = parts[1] if len(parts) > 1 else ""
            category = parts[2] if len(parts) > 2 else ""
            filename = parts[-1] if parts else source
            product_name = filename.rsplit(".", 1)[0]  # 去掉 .pdf
            return {"company": company, "category": category, "product_name": product_name}

        retrieval_info = []
        for i, r in enumerate(refs):
            meta = r.get("metadata", {})
            source = meta.get("source", "?")
            parsed_src = _parse_source(source)
            retrieval_info.append({
                "rank": i + 1,
                "company": meta.get("company") or parsed_src["company"],
                "category": meta.get("category") or parsed_src["category"],
                "product_name": parsed_src["product_name"],
                "source": source,
                "modality": r.get("modality", "text"),
                "page_number": meta.get("page_number", "?"),
                "score": r["score"],
                "text_preview": r["text"][:200] if r.get("text") else "",
            })

        # 4. 评估
        print(f"  → 检索到 {len(refs)} 个 chunk，正在评估...")
        t1 = time.time()
        try:
            eval_result = evaluate(prompt, refs, parsed["cot"], parsed["final_answer"])
        except Exception as e:
            eval_result = {"verdict": "ERROR", "reason": str(e), "total": -1}
        latency_eval = round(time.time() - t1, 2)

        verdict = eval_result.get("verdict", "?")
        total = eval_result.get("total", -1)
        print(f"  → 评估: {verdict}  总分:{total}/12  ({eval_result.get('reason','')[:60]})")

        results.append({
            "id": tid,
            "category": item["category"],
            "expected": item["expected"],
            "note": item["note"],
            "prompt": prompt,
            "mode": MODE,
            "top_k": TOP_K,
            "latency_total_s": latency_total,
            "latency_eval_s": latency_eval,
            "retrieval": retrieval_info,
            "model_cot": parsed["cot"],
            "final_answer": parsed["final_answer"],
            "raw_model_output": parsed["raw"],
            "evaluation": eval_result,
        })

    # 汇总
    verdicts = [r.get("evaluation", {}).get("verdict", "ERROR") for r in results]
    pass_n = verdicts.count("PASS")
    partial_n = verdicts.count("PARTIAL")
    fail_n = verdicts.count("FAIL")
    error_n = verdicts.count("ERROR")
    scores = [r.get("evaluation", {}).get("total", 0) for r in results if r.get("evaluation", {}).get("total", -1) >= 0]
    avg_score = round(sum(scores) / len(scores), 2) if scores else -1

    summary = {
        "run_at": datetime.now().isoformat(),
        "total_tests": len(TEST_PROMPTS),
        "pass": pass_n,
        "partial": partial_n,
        "fail": fail_n,
        "error": error_n,
        "avg_score_12": avg_score,
    }

    output = {"summary": summary, "tests": results}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  完成！PASS={pass_n}  PARTIAL={partial_n}  FAIL={fail_n}  ERROR={error_n}")
    print(f"  平均分: {avg_score}/12")
    print(f"  结果已保存至: {OUTPUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_smoke_test()
