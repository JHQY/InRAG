import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/home/jhqy/IRF/finetuning/qwen3/eval/results"

FILES = {
    "clause": "result_clause_eval.jsonl",
    "hallucination": "result_hallucination_eval.jsonl",
    "seq2seq": "result_seq2seq_eval.jsonl"
}

# 新评估维度（和 evaluate.py 完全一致）
NEW_METRICS = [
    "format",
    "style_professional",
    "term_accuracy",
    "hallucination",
    "fidelity_to_context",
    "structure_completeness"
]

# --------------------------
# 工具：读取 JSONL
# --------------------------
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                continue
    return items


# ------------------------------------------------------
# A：统计胜率 + 平均分（新版六维度）
# ------------------------------------------------------
def summary_statistics(data):
    winner_count = Counter()
    score_sum = {side: defaultdict(float) for side in ["A", "B"]}
    sample_count = len(data)

    for item in data:
        judge = item.get("judge", {})
        winner = judge.get("winner", None)
        if winner:
            winner_count[winner] += 1

        for side in ["A", "B"]:
            scores = judge.get(side, {})
            for m in NEW_METRICS:
                if m in scores:
                    score_sum[side][m] += scores[m]

    # 平均值
    avg_scores = {side: {} for side in ["A", "B"]}
    for side in ["A", "B"]:
        for m in NEW_METRICS:
            avg_scores[side][m] = (
                score_sum[side][m] / sample_count if sample_count > 0 else 0
            )

    return winner_count, avg_scores


# ------------------------------------------------------
# B：自动错误分析（新版 Error Taxonomy）
# ------------------------------------------------------
def error_taxonomy(data):
    errors = defaultdict(list)

    for item in data:
        judge = item.get("judge", {})
        B = judge.get("B", {})

        if not B:
            continue

        id_ = item.get("id")

        # 1. 格式错误（format < 5）
        if B.get("format", 10) < 5:
            errors["格式不规范"].append(id_)

        # 2. 风格专业度差（style < 5）
        if B.get("style_professional", 10) < 5:
            errors["语言不够专业"].append(id_)

        # 3. 术语不准确（term < 5）
        if B.get("term_accuracy", 10) < 5:
            errors["术语使用不准确"].append(id_)

        # 4. 幻觉过多（hallucination < 5 → 注意幻觉越高越好）
        if B.get("hallucination", 10) < 5:
            errors["存在幻觉或扩写"].append(id_)

        # 5. 忠实度差（fidelity < 5）
        if B.get("fidelity_to_context", 10) < 5:
            errors["不忠实材料"].append(id_)

        # 6. 结构不完整（structure < 5）
        if B.get("structure_completeness", 10) < 5:
            errors["结构不完整"].append(id_)

    return errors


# ------------------------------------------------------
# C：绘图
# ------------------------------------------------------
def plot_winrate(winner_count, out_path):
    labels = ["A", "B", "tie"]
    values = [winner_count.get(k, 0) for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Model Win Rate (A = Base, B = FineTune)")
    plt.ylabel("Count")
    plt.savefig(out_path)
    plt.close()


def plot_avgscore(avg_scores, out_path):
    categories = NEW_METRICS
    A_vals = [avg_scores["A"][k] for k in categories]
    B_vals = [avg_scores["B"][k] for k in categories]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    A_plot = A_vals + A_vals[:1]
    B_plot = B_vals + B_vals[:1]
    ang_plot = angles + angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(ang_plot, A_plot, label="Base")
    ax.fill(ang_plot, A_plot, alpha=0.1)

    ax.plot(ang_plot, B_plot, label="FineTune")
    ax.fill(ang_plot, B_plot, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles), categories)
    ax.set_title("Average Score Radar Chart")
    ax.legend()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------
# 主流程
# ------------------------------------------------------
def main():
    for key, fname in FILES.items():
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            print("缺少文件：", path)
            continue

        print(f"\n========== 分析 {key} ==========")
        data = load_jsonl(path)

        # A：胜率 + 平均分
        winner_count, avg_scores = summary_statistics(data)
        print("胜率统计：", winner_count)
        print("平均分：", avg_scores)

        # B：错误分析
        error_clusters = error_taxonomy(data)
        print("错误分类：", error_clusters)

        # C：绘图
        plot_winrate(
            winner_count,
            os.path.join(BASE_DIR, f"{key}_winrate.png")
        )
        plot_avgscore(
            avg_scores,
            os.path.join(BASE_DIR, f"{key}_avgscore.png")
        )

        # 保存 JSON
        out_json = {
            "winner_count": dict(winner_count),
            "avg_scores": avg_scores,
            "error_clusters": error_clusters,
        }

        with open(
            os.path.join(BASE_DIR, f"summary_{key}.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        print(f"输出完成：summary_{key}.json\n")


if __name__ == "__main__":
    main()
