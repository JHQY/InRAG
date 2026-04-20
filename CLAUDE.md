# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Insurance RAG Finetuning (IRF) — a three-stage LoRA fine-tuning pipeline for **Qwen3-4B** targeting insurance contract QA. The model is trained to answer questions strictly from retrieved evidence (RAG), reject when evidence is insufficient, and handle SQL-based database QA.

**Base model:** `finetuning/qwen3/Qwen3-4B/` (local copy)
**Virtual env:** `finetuning/qwen3/.venv/`
**Training framework:** Axolotl

## Training Pipeline (Three Stages)

| Stage | Config | Data | Purpose |
|-------|--------|------|---------|
| Stage 1 (Clause) | `configs/stage1_clause/qwen3-4b-clause.yaml` | `data/clause_{objective,subjective}_completion.jsonl` + `db_train_completion.jsonl` | Completion-format clause reading comprehension |
| Stage 2 (DB mix) | `configs/stage2_db/qwen3-4b.yaml` | `data/db_train.json` | SQL-CoT database QA; loads stage1 LoRA at `outputs/qwen3-4b-lora-stage2` |
| Stage 3 (RAFT) | `configs/stage3_raft/qwen3-4b-raft.yaml` | `data/raft/raft_train.jsonl` | RAG-augmented faithful answering; loads stage2 LoRA at `outputs/qwen3-4b-lora-stage2` |

All outputs go under `finetuning/qwen3/outputs/`.

## Common Commands

All commands run from `finetuning/qwen3/` with the venv activated:

```bash
cd /home/jhqy/IRF/finetuning/qwen3
source .venv/bin/activate
```

**Train a stage:**
```bash
axolotl train configs/stage3_raft/qwen3-4b-raft.yaml
```

**Regenerate RAFT training data:**
```bash
python3 data/raft/generate_raft.py
```

**Evaluate RAFT (3-model comparison):**
```bash
python3 eval/scripts/eval_raft.py \
  --base   Qwen3-4B \
  --lora2  outputs/qwen3-4b-lora-stage2 \
  --lora3  outputs/qwen3-4b-lora-raft-v2 \
  --clause_data  eval/data/clause.jsonl \
  --hall_data    eval/data/hallucination_eval.jsonl \
  --output_dir   eval/results/raft_eval_v4 \
  --n_clause 30 --n_distractor 15
```

**Evaluate clause quality (baseline vs fine-tuned):**
```bash
python3 eval/scripts/eval_clause.py  # see script for args
```

**Test GPU max sequence length:**
```bash
python3 test_max.py
```

**Verify RAFT data quality after regeneration:**
```bash
python3 -c "
import json
samples = [json.loads(l) for l in open('data/raft/raft_train.jsonl') if l.strip()]
total = len(samples)
default_entity = sum(1 for s in samples for m in s.get('conversations',[]) if m['from']=='gpt' and '（见问题）' in m['value'])
with_zhuanye = sum(1 for s in samples for m in s.get('conversations',[]) if m['from']=='gpt' and '专有名词' in m['value'])
print(f'总样本: {total}')
print(f'（见问题）占比: {default_entity/total*100:.1f}% （目标 < 20%）')
print(f'含专有名词占比: {with_zhuanye/total*100:.1f}% （目标 = 0%）')
"
```

## Architecture: `generate_raft.py`

`data/raft/generate_raft.py` generates three sample types (target ratio 52.5% : 31.5% : 15%):

1. **RAG positive** — question + 1–3 chunks including the correct evidence → `<Thought>` CoT + structured answer
2. **RAG distractor** — question + mismatched chunks only → `<Thought>` + fixed rejection answer
3. **DB-CoT** — sampled directly from `db_train.json` to preserve SQL reasoning

Key functions:
- `extract_entities(question)` — extracts key entities (quoted text > clause numbers > insurance terms > amounts) for `<Thought>` generation; falls back to `"（见问题）"` only when nothing matches
- `strip_zhuanye(answer)` — removes `专有名词解释` paragraphs (parameterized knowledge leak) from training answers
- `_has_keyword_overlap(question, clause)` — filters positive samples where question entities don't appear in the clause (prevents false positives)
- `build_positive_samples` / `build_distractor_samples` — construct chat-format samples with `<Thought>` reasoning chains

Chat format uses `conversations: [{from: human/gpt, value: ...}]` with a system prompt enforcing strict evidence-only answering.

## Evaluation

`eval/scripts/eval_raft.py` runs three-way comparison (baseline / stage2 / stage3):
- Loads models sequentially, unloads after inference to fit 16GB VRAM
- Uses **DeepSeek API** as judge (key stored in `eval/apikey`)
- Three task types: `clause` (evidence QA), `hallucination` (evidence-insufficient), `distractor` (mismatched evidence → should reject)
- Supports `--load_outputs` to resume from cached `raft_raw_outputs.json`

Results saved to `eval/results/raft_eval_v*/`.

## Hardware Notes

- GPU: 16GB VRAM (Blackwell architecture — `flash_attention: false`, `flex_attention: false`)
- Stage 3: `micro_batch_size: 1`, `gradient_accumulation_steps: 4`, `sequence_len: 1536`, `sample_packing: false`
- Stage 1/2: `sequence_len: 1024`, stage2 uses `sliding_window: true`
- All training uses `bf16: true`, `gradient_checkpointing: true`
- WandB project: `qwen3_ift`
