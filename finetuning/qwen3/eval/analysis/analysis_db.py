#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# 将 deepseek 字段转换为 float
# ----------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return None   # 转不出来就当作无效


# ----------------------------
# 读取 result_db.jsonl
# ----------------------------
def load_results(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# ----------------------------
# 聚合统计
# ----------------------------
def aggregate_scores(data):
    metrics = [
        "field_selection",
        "sql_reasoning",
        "sql_correctness",
        "structure",
        "hallucination",
        "fidelity",
        "similarity_to_gold",
    ]

    baseline_sum = defaultdict(float)
    ft_sum = defaultdict(float)
    baseline_count = defaultdict(int)
    ft_count = defaultdict(int)

    wins = {"A": 0, "B": 0, "tie": 0}

    for item in data:
        judge = item["judge"]
        wins[judge["winner"]] += 1

        for m in metrics:
            a = safe_float(judge["A"][m])
            b = safe_float(judge["B"][m])

            if a is not None:   # 只有有效数据才累计
                baseline_sum[m] += a
                baseline_count[m] += 1

            if b is not None:
                ft_sum[m] += b
                ft_count[m] += 1

    # 求平均（无效数据不会参与）
    baseline_avg = {
        m: (baseline_sum[m] / baseline_count[m]) if baseline_count[m] else 0.0
        for m in metrics
    }
    ft_avg = {
        m: (ft_sum[m] / ft_count[m]) if ft_count[m] else 0.0
        for m in metrics
    }

    total_items = len(data)

    return metrics, baseline_avg, ft_avg, wins, total_items


# ----------------------------
# 雷达图绘制
# ----------------------------
def plot_radar(metrics, base_avg, ft_avg, output_png):
    labels = metrics
    base_scores = [base_avg[m] for m in metrics]
    ft_scores = [ft_avg[m] for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    base_scores += base_scores[:1]
    ft_scores += ft_scores[:1]
    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, base_scores, "o-", label="Baseline")
    ax.fill(angles, base_scores, alpha=0.1)

    ax.plot(angles, ft_scores, "o-", label="Finetuned")
    ax.fill(angles, ft_scores, alpha=0.1)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=10)
    ax.set_title("DB-CoT Evaluation Radar Chart", fontsize=14)
    ax.legend(loc="upper right")

    plt.savefig(output_png, dpi=200)
    plt.close()


# ----------------------------
# 柱状图绘制
# ----------------------------
def plot_bar(metrics, base_avg, ft_avg, output_png):
    x = np.arange(len(metrics))
    base_scores = [base_avg[m] for m in metrics]
    ft_scores = [ft_avg[m] for m in metrics]

    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, base_scores, width, label="Baseline")
    plt.bar(x + width/2, ft_scores, width, label="Finetuned")

    plt.xticks(x, metrics, rotation=30)
    plt.ylabel("Avg Score (0–5)")
    plt.title("Metric Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


# ----------------------------
# summary
# ----------------------------
def write_summary(path, metrics, base_avg, ft_avg, wins, count):
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== DB-CoT Evaluation Summary =====\n\n")
        f.write(f"Total samples: {count}\n\n")

        f.write("Win Rate:\n")
        f.write(f"  Baseline wins: {wins['A']}\n")
        f.write(f"  Finetuned wins: {wins['B']}\n")
        f.write(f"  Ties: {wins['tie']}\n\n")

        f.write("Average Scores (0–5):\n")
        for m in metrics:
            f.write(f"- {m:20s} | base={base_avg[m]:.3f} | ft={ft_avg[m]:.3f}\n")


# ----------------------------
# 主流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    print("► Loading result file…")
    data = load_results(args.input)

    print("► Aggregating scores…")
    metrics, base_avg, ft_avg, wins, count = aggregate_scores(data)

    print("► Plot radar chart…")
    plot_radar(metrics, base_avg, ft_avg, f"{args.outdir}/radar.png")

    print("► Plot bar chart…")
    plot_bar(metrics, base_avg, ft_avg, f"{args.outdir}/bar.png")

    print("► Writing summary…")
    write_summary(f"{args.outdir}/summary.txt", metrics, base_avg, ft_avg, wins, count)

    print(f"\n✓ Done! Results saved to {args.outdir}\n")


if __name__ == "__main__":
    main()
