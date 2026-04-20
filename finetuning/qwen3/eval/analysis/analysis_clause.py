import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# ===========================================
#   读取评测结果
# ===========================================

def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ===========================================
#   统计 win / tie / loss
# ===========================================

def compute_win_rate(results):
    total = len(results)
    win = sum(1 for r in results if r["judge"].get("winner") == "B")
    loss = sum(1 for r in results if r["judge"].get("winner") == "A")
    tie = sum(1 for r in results if r["judge"].get("winner") == "tie")

    return {
        "total": total,
        "win": win,
        "loss": loss,
        "tie": tie,
        "win_rate": win / total if total > 0 else 0
    }


# ===========================================
#   按 task_type 分类统计
# ===========================================

def split_by_task_type(results):
    obj = []
    sub = []
    for r in results:
        if r.get("task_type") == "objective":
            obj.append(r)
        else:
            sub.append(r)
    return obj, sub


# ===========================================
#   统计 DeepSeek 各维度平均分
# ===========================================

def compute_dimension_scores(results):
    # 每条的 judge 是：
    # {
    #   "A": {dim: score, ...},
    #   "B": {dim: score, ...},
    #   "winner": ...
    # }

    dims = ["correctness", "evidence_use", "no_hallucination", "structure", "clarity", "fidelity"]

    avg_A = defaultdict(list)
    avg_B = defaultdict(list)

    for r in results:
        jr = r["judge"]
        if "A" not in jr or "B" not in jr:
            continue

        for d in dims:
            if d in jr["A"]:
                avg_A[d].append(jr["A"][d])
            if d in jr["B"]:
                avg_B[d].append(jr["B"][d])

    # 计算均值
    meanA = {d: np.mean(avg_A[d]) if len(avg_A[d]) else 0 for d in dims}
    meanB = {d: np.mean(avg_B[d]) if len(avg_B[d]) else 0 for d in dims}

    return dims, meanA, meanB


# ===========================================
#   绘制雷达图
# ===========================================

def plot_radar(dims, meanA, meanB, out_png="radar_clause.png"):
    valuesA = [meanA[d] for d in dims]
    valuesB = [meanB[d] for d in dims]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    valuesA += valuesA[:1]
    valuesB += valuesB[:1]
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, valuesA, "o-", label="Baseline")
    ax.fill(angles, valuesA, alpha=0.3)

    ax.plot(angles, valuesB, "o-", label="Finetune")
    ax.fill(angles, valuesB, alpha=0.3)

    ax.set_thetagrids(np.degrees(angles[:-1]), dims)
    ax.set_title("Clause Evaluation Radar Chart", fontsize=16)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"✓ 雷达图已保存：{out_png}")


# ===========================================
#   绘制 win/loss/tie 柱状图
# ===========================================

def plot_bar(win_stats, out_png="winrate_clause.png"):
    labels = ["Win", "Tie", "Loss"]
    values = [win_stats["win"], win_stats["tie"], win_stats["loss"]]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, color=["green", "gray", "red"])
    plt.title("Finetune vs Baseline – Win/Tie/Loss")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"✓ 柱状图已保存：{out_png}")


# ===========================================
#   主流程
# ===========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="result_clause.jsonl")
    parser.add_argument("--outdir", default=".", help="output directory")
    args = parser.parse_args()

    print("► Loading result…")
    results = load_results(args.data)

    print(f"共 {len(results)} 条评测样本")

    # 全部结果
    full_stats = compute_win_rate(results)
    print("=== Overall ===")
    print(full_stats)

    # objective & subjective 分开
    obj, sub = split_by_task_type(results)

    print("\n=== Objective ===")
    print(compute_win_rate(obj))

    print("\n=== Subjective ===")
    print(compute_win_rate(sub))

    # 各维度分数
    dims, meanA, meanB = compute_dimension_scores(results)
    print("\n=== Baseline Avg Scores ===")
    print(meanA)
    print("\n=== Finetune Avg Scores ===")
    print(meanB)

    # 绘图
    plot_radar(dims, meanA, meanB, out_png=f"{args.outdir}/radar_clause.png")
    plot_bar(full_stats, out_png=f"{args.outdir}/winrate_clause.png")

    print("\n✓ Analysis Done!")


if __name__ == "__main__":
    main()
