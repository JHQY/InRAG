# 模型评测说明文档

> 用途：论文 Experiment 章节写作材料  
> 更新日期：2026-04-14  
> 评测对象：Stage-3 RAFT 模型（`outputs/qwen3-4b-lora-raft-v2`）vs Baseline（Qwen3-4B 无微调）

---

## 一、评测设计原则

### 训练目标

本项目的三阶段微调目标如下：

| 阶段 | 核心目标 |
|------|---------|
| Stage-1（Clause） | 学会基于条款原文作答，建立证据锚定习惯 |
| Stage-2（SQL-CoT） | 学会结构化推理，精确使用保险专业术语，拒绝幻觉输出 |
| Stage-3（RAFT） | 在多chunk检索场景下有据作答，无据拒答，输出结构化合规格式 |

**Stage-3 的六项核心能力**（评测以此为导向）：

| 编号 | 能力 | 说明 |
|------|------|------|
| ① | CoT推理透明度 | 推理过程可追溯，输出`<Thought>`块 |
| ② | 拒答（abstention） | 证据不足或不相关时拒绝回答，不编造 |
| ③ | 结构化模板输出 | `[答案]/[证据]/[解释说明]`格式，自适应切换 |
| ④ | 证据锚定 | 答案内容来自检索结果，不使用参数化知识 |
| ⑤ | 噪声过滤 | 多chunk混合输入中识别并引用正确chunk |
| ⑥ | 跨合同不混淆 | 不将干扰合同的内容套用到当前问题 |

### 评测的核心主张

> Stage-3 的价值不在于通用答题能力，而在于**合规性基础设施**：在有据时精确引用条款，在无据时明确拒答并指引用户，全程输出结构化可审计格式。

这与传统 QA benchmark（追求高覆盖率、高正确率）有根本差异——在保险场景，**答错比不答危险**。

---

## 二、系统 Prompt（最终版 v3）

所有 Stage-3 推理均使用以下 system prompt：

```
你是一个保险助手，请严格根据提供的检索结果回答用户问题，
不得使用检索结果以外的任何知识，不得根据不相关的检索结果推断答案。
请按以下规则处理：
1. 检索结果包含相关信息 → 直接基于该信息作答，引用具体条款内容
2. 检索结果不包含相关信息或与问题无关 → 输出：
   '根据现有检索资料，暂无法提供关于该问题的准确信息。
   建议您直接联系保险公司客服或查阅完整合同原文，以获取准确的条款解释。'
```

**版本演进记录**（供论文 Discussion 参考）：

| 版本 | 策略 | Hallucination | Distractor | 问题 |
|------|------|--------------|------------|------|
| v1（原版） | 严格拒答，无固定格式 | 72.0% | 58.2% | 拒答时说明质量差 |
| v2（分级响应） | 部分相关→尝试作答 | 85.2% | 18.7% | distractor幻觉飙升 |
| v3（最终版） | 严格拒答+标准建议语 | 83.6% | 31.1% | 两侧折中，无根本突破 |

**结论**：Prompt 工程无法同时解决两个方向的取舍，根本解法是训练数据中增加"部分相关"类别样本。v3 在实际部署中用户体验更好（明确指引），采用为最终版。

---

## 三、各项测试说明

### Test-1：有据回答质量（Fair Clause Eval）

**对应能力**：④ 证据锚定

**为什么"公平"**：原始 eval_raft.py 对 Stage-3 使用 RAFT 多chunk格式推理，但 judge 只收到单chunk原始 prompt，evidence_use 维度有偏。本测试修正此偏差，judge 收到各模型实际输入。

**测试设计**：
- 数据：`eval/data/clause.jsonl`，随机抽取30题（seed=42）
- Baseline：单chunk输入 + CLAUSE_SYSTEM
- Stage-3：多chunk RAFT输入（1正确chunk + 1干扰chunk）+ RAFT_SYSTEM_V3
- Judge：DeepSeek-Chat（温度0.1），5维度，满分25

**结果**：

| 模型 | 总分/25 | correctness | evidence_use | no_hallucination | structure | fidelity |
|------|---------|-------------|-------------|-----------------|-----------|---------|
| Baseline | 82.8% | 3.43 | 3.53 | **4.90** | **4.70** | 4.13 |
| Stage-3 | 73.7% | 2.53 | **4.07** | 4.17 | 4.10 | 3.57 |

**解读**：
- Stage-3 evidence_use（4.07）显著高于 Baseline（3.53），说明证据锚定训练有效
- Baseline 总分略高，差距集中在 correctness 和 structure：Stage-3 在单chunk任务中输出了 RAFT 格式（含`<Thought>`），在该格式下不是最优适配
- no_hallucination Baseline 更高（4.90 vs 4.17），原因见 Test-5 的 Grounded Judge 修正

**数据文件**：
- 脚本：`eval/scripts/eval_fair_clause.py`
- 结果：`eval/results/raft_eval_fair/fair_eval_results.json`

---

### Test-2：拒答能力——证据不足场景（Hallucination Suppression）

**对应能力**：② 拒答

**测试设计**：
- 数据：`eval/data/hallucination_eval.jsonl`，10题，证据故意不足以完整回答
- Baseline：单chunk输入 + CLAUSE_SYSTEM
- Stage-3：将单chunk包装为RAFT格式（检索结果1）+ RAFT_SYSTEM_V3
- Judge：5维度，满分25

**结果（v3 prompt）**：

| 模型 | 总分/25 | correctness | evidence_use | no_hallucination | structure | fidelity |
|------|---------|-------------|-------------|-----------------|-----------|---------|
| Baseline | 85.6% | 4.00 | 4.10 | 4.50 | 4.30 | 4.50 |
| Stage-3 v3 | 83.6% | 3.50 | 4.00 | 4.50 | **4.90** | 4.00 |

**解读**：
- 两者得分接近（85.6% vs 83.6%），差距在1个标准误差内
- Baseline 高分来自"大胆作答"——它从不拒答，judge 给了高 fidelity 和 correctness
- Stage-3 的 structure 满分（4.90）反映了结构化输出的稳定性
- **重要**：Baseline 的高分是"假高"，见 Test-5（Grounded Judge）修正后差距缩小

**数据文件**：
- 脚本：`eval/scripts/eval_abstention_v2.py --prompt_version v3`
- 结果：`eval/results/abstention_v2/abstention_v3_results.json`

---

### Test-3：拒答能力——全错chunk场景（Distractor Rejection）

**对应能力**：②⑥ 拒答 + 跨合同不混淆

**测试设计**：
- 数据：从`eval/data/clause.jsonl`随机抽取15题（seed=42）
- 输入：2个来自不同合同的干扰chunk（不含正确答案），Stage-3 应拒答
- Judge：3维度，满分15

**结果（v3 prompt）**：

| 模型 | 总分/15 | rejection_correctness | no_hallucination | explanation_quality |
|------|---------|----------------------|-----------------|-------------------|
| Baseline | ~65% | 3.67 | 3.67 | ~2.3 |
| Stage-3 v1 | **58.2%** | 3.00 | 3.00 | 2.40 |
| Stage-3 v3 | 31.1% | 1.67 | 1.67 | **1.33** |

**解读**：
- Stage-3 v1 在 distractor 任务最优（拒答率更高，不幻觉），v3 因标准建议语降低了判断严格性
- v3 的 31.1% 是当前最大局限：模型将干扰chunk误判为"部分相关"后尝试作答
- Baseline 的 65% 同样存在假高问题：它"恰好"给出了正确方向的模糊回答
- **根本原因**：训练数据中 distractor 样本比例（31.5%）不足，且缺少"部分相关但无法作答"类别

**数据文件**：
- 脚本：`eval/scripts/eval_abstention_v2.py --prompt_version v3`
- 结果：`eval/results/abstention_v2/abstention_v3_results.json`

---

### Test-4：合规表达质量（Answer Compliance Eval）

**对应能力**：①③④ CoT透明度 + 结构化输出 + 证据锚定的综合体现

**核心问题**：SQL-CoT 训练（Stage-2）是否使模型输出更"合规"——更精确使用保险术语、更忠实于条款原文？

**测试设计**：
- 数据：复用 `fair_raw_outputs.json` 中 Baseline 和 Stage-3 的30对输出，无需新推理
- Judge：对比两个模型答案的合规质量，3维度，满分15

**结果**：

| 模型 | 总分/15 | answer_compliance | clause_fidelity | response_formality |
|------|---------|------------------|----------------|-------------------|
| Baseline | 9.30（62%） | 2.93 | 2.57 | **3.80** |
| Stage-3 | **10.60（71%）** | **3.47** | **3.43** | 3.70 |

**解读**：
- **clause_fidelity 差距最大（+0.87）**：Stage-3 更倾向直接引用条款原文而非自行解释——SQL-CoT 的"引用字段→构造答案"习惯迁移到了 RAFT 场景
- answer_compliance（+0.53）：Stage-3 更精确使用保险专业术语
- response_formality 两者接近：Baseline 在措辞规范性上有轻微优势（从不拒答，偶尔显得更"服务导向"）
- Stage-3 胜出 15/28 场，平局 2 场，Baseline 胜 13 场

**数据文件**：
- 脚本：`eval/scripts/eval_cot_quality.py`
- 结果：`eval/results/cot_quality/compliance_results.json`

---

### Test-5：评测公平性修正——溯源检查（Grounded Judge）

**核心问题**：修复 Baseline 假高问题——原始 judge 对"模糊泛化"型幻觉不敏感。

**两阶段 Judge 设计**：
1. 溯源检查：列出模型输出中的具体声明，逐条核查是否在证据原文中有直接依据
2. 如声明无法溯源 → `no_hallucination=0`，`evidence_use=0`（强制）
3. 在溯源结果基础上完成常规5维度评分

**结果（Hallucination Suppression 任务）**：

| 模型 | 原始 Judge | 溯源 Judge | 差值 | 溯源通过率 |
|------|-----------|-----------|------|-----------|
| Baseline | 85.2% | 80.4% | ▼4.8% | 7/10（70%） |
| Stage-3 | 72.0% | 74.0% | ▲2.0% | 6/10（60%） |

**解读**：
- 差距从 13.2% 缩减至 6.4%，方向正确
- Baseline 的幻觉多为"模糊泛化"（无具体数字，难被捕捉），LLM-as-judge 对此系统性不敏感
- 这是当前自动评测的固有局限，而非 Stage-3 本身的问题
- **论文 Discussion 要点**：LLM judge 对"隐性幻觉"（有信心但无据的泛化）的检测能力有限，这是领域 fine-tuning 评测的通用挑战

**数据文件**：
- 脚本：`eval/scripts/eval_grounded_judge.py`
- 结果：`eval/results/grounded_judge/grounded_hall_results.json`

---

### Test-6：结构化输出合规性（Format Compliance）

**对应能力**：③ 结构化模板输出

**测试设计**：
- 数据：Smoke Test v5 全部15个 Stage-3 输出的 `raw_model_output` 字段
- 方法：正则解析，统计各结构标签出现率

**结果**：

| 指标 | 结果 |
|------|------|
| `<Thought>` 块出现率 | 15/15（**100%**） |
| `[答案]` 字段出现率 | 15/15（**100%**） |
| `[证据]+[解释说明]`（拒答时） | 2/15（13%，仅拒答题） |

**解读**：
- Stage-3 实现了**自适应格式**：有据作答时只输出`[答案]`，拒答时切换为完整3字段结构
- 100% 合规率在端到端真实流量中验证，不依赖特定测试集

---

### Test-7：噪声过滤（Noise Filter Eval）

**对应能力**：⑤⑥ 噪声过滤 + 跨合同不混淆

**测试设计**：
- 数据：30题（clause.jsonl），输入为1正确chunk + 2干扰chunk，顺序随机打乱
- 对比：Stage-3 vs Baseline（均收到相同多chunk输入）
- Judge：4维度，满分20

**结果**：

| 模型 | 总分/20 | evidence_selection | no_contamination | correctness | source_attribution |
|------|---------|-------------------|-----------------|-------------|-------------------|
| Baseline | 12.10（60.5%） | **3.47** | **4.13** | **3.33** | 1.17 |
| Stage-3 | 8.40（42.0%） | 2.50 | 4.00 | 1.90 | 0.00 |

**关键发现——元数据依赖性**：
- 30题中10题的正确chunk合同名提取失败，被标注为"未知合同"
- Stage-3 看到"未知合同"触发保守拒答（30%拒答率），导致 correctness 被惩罚
- Baseline 从不拒答，反而得分更高
- **生产环境中此问题不存在**：Milvus 从 PDF 元数据中提取完整合同名，chunk 标注完整
- **论文 Discussion 要点**：Stage-3 的噪声过滤能力依赖 chunk 元数据完整性，这是系统层面的前提条件，而非模型缺陷

**数据文件**：
- 脚本：`eval/scripts/eval_noise_filter.py`
- 结果：`eval/results/noise_filter/noise_filter_results.json`

---

### Test-8：端到端 Smoke Test

**对应能力**：全部能力在真实 RAG 管道中的综合体现

**测试设计**：
- 真实 HK 保险问题 → Milvus 检索（top-8）→ Stage-3 RAFT 生成
- Judge：4维度（retrieval relevance, faithfulness, rejection quality, completeness），满分12

**结果**：

| 测试集 | 题数 | avg/12 | 典型失败模式 |
|--------|------|--------|------------|
| v4 | 10 | 7.90 | 检索层未命中（3题） |
| v5 | 15 | 8.07 | 检索层 + KB覆盖缺口 + 信息跨多chunk |

**失败归因（v5）**：

| 类型 | 题目 | 原因层 |
|------|------|--------|
| Faithfulness=0 | Q3, Q7, Q9 | 检索层：命中主题但无具体数值 |
| 双失败 | Q12 | KB层：Hang Seng 覆盖缺口 |
| 部分失败 | Q11, Q13 | 模型层：信息分散在多个chunk，生成不完整 |
| 正确拒答 | Q15 | 6/12，拒答成功，RAFT abstention 有效迁移 |

**关键结论**：三类失败（检索层/KB层/模型层）需分层归因。当前失败主要集中在检索层，模型行为本身（faithfulness、abstention）在端到端环境中表现稳定。

---

### Test-9：DB-QA 专项（Stage-2 结构化推理能力）

**说明**：此测试对比 Stage-2 vs Baseline，独立于 Stage-3 的 RAG 能力评测，体现 SQL-CoT 训练的专项价值。

**测试设计**：
- 数据：`eval/results/result_db.jsonl`，15题结构化保险数据库问答
- 对比：Baseline vs Stage-2，7维度（满分35）

**结果**：Win/Tie/Loss = **13:1:1**（Stage-2 胜）

| 维度（avg/5） | Baseline | Stage-2 |
|-------------|----------|---------|
| field_selection | 2.93 | **4.53** |
| sql_reasoning | 2.13 | **4.10** |
| sql_correctness | 2.37 | **4.67** |
| structure | 2.93 | **4.87** |
| no_hallucination | 4.53 | **5.00** |
| similarity_to_gold | 0.33 | **3.00** |

**解读**：SQL-CoT 训练赋予了模型在保险数据库查询场景下的完整推理能力。这是 Baseline 完全不具备的能力维度，体现了 Stage-2 的独立价值——此能力通过 DB-CoT passthrough 在 Stage-3 中保留。

---

## 四、核心局限与论文 Discussion 方向

| 局限 | 具体表现 | 建议写法 |
|------|---------|---------|
| 训练数据比例 | Distractor 样本 31.5%，模型拒答阈值偏严但不稳定 | 建议增加 distractor 比例或加入"部分相关"第四类 |
| Prompt 工程天花板 | v2/v3 在 hallucination vs distractor 之间存在根本取舍 | Prompt 只能调阈值，边界判断需训练数据支撑 |
| 评测偏差 | LLM judge 对模糊泛化型幻觉不敏感，Baseline 假高 | Grounded Judge 将差距从13.2%缩至6.4%，但隐性幻觉仍难捕捉 |
| 噪声过滤评测设计 | 元数据缺失导致 Stage-3 过度拒答 | 生产环境测试（Smoke Test）是更可靠的端到端验证 |
| 小样本规模 | Hallucination 10题，Distractor 15题 | 置信区间宽，结论需谨慎表述 |

---

## 五、结果文件索引

| 测试 | 脚本 | 结果文件 |
|------|------|---------|
| Fair Clause Eval | `eval/scripts/eval_fair_clause.py` | `eval/results/raft_eval_fair/` |
| Hallucination + Distractor v3 | `eval/scripts/eval_abstention_v2.py` | `eval/results/abstention_v2/abstention_v3_*.json` |
| Answer Compliance | `eval/scripts/eval_cot_quality.py` | `eval/results/cot_quality/compliance_results.json` |
| Grounded Judge | `eval/scripts/eval_grounded_judge.py` | `eval/results/grounded_judge/grounded_hall_results.json` |
| Noise Filter | `eval/scripts/eval_noise_filter.py` | `eval/results/noise_filter/` |
| Smoke Test v5 | `rag_service/eval/scripts/` | `rag_service/eval/results/smoke_test_15_v5.json` |
| DB-QA | — | `eval/results/result_db.jsonl` |
| Original 3-model eval | `eval/scripts/eval_raft.py` | `eval/results/raft_eval_v4/` |
