# InRAG — Insurance RAG Finetuning

A production-ready RAG QA system for Hong Kong insurance contracts, combining a **three-stage LoRA fine-tuned Qwen3-4B** model with a **Milvus-backed retrieval pipeline**.

The model is trained to answer strictly from retrieved evidence, and explicitly reject when evidence is insufficient — targeting faithfulness over coverage.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  api_server.py  (port 8000)     │
│  FastAPI + Vue3 frontend        │
│                                 │
│  1. Embed query (bge-large-zh)  │
│  2. Milvus search + rerank      │
│  3. Build RAFT prompt           │
│  4. Call RAFT inference server  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  serve/server.py  (port 8001)   │
│  Qwen3-4B + RAFT LoRA (merged)  │
│  OpenAI-compatible API          │
└─────────────────────────────────┘
               │
               ▼
         Answer + Sources
```

**Knowledge base:** Milvus `rag_collection` — 37,571 entities from 247 HK insurance PDFs  
**Embedding model:** `BAAI/bge-large-zh-v1.5` (1024-dim, CPU)  
**Reranker:** `BAAI/bge-reranker-base` (CPU)  
**LLM:** Qwen3-4B with 3-stage LoRA (merged to `outputs/qwen3-4b-raft-v2-merged`)

---

## Repository Structure

```
InRAG/
├── finetuning/qwen3/           # Fine-tuning pipeline
│   ├── configs/                # Axolotl training configs (3 stages)
│   ├── serve/server.py         # RAFT inference server (port 8001)
│   ├── eval/                   # Fine-tuning evaluation scripts
│   └── outputs/                # Model checkpoints (not in repo)
│
└── rag_service/                # RAG service
    ├── api_server.py           # Main FastAPI app (port 8000)
    ├── frontend/index.html     # Vue3 chat UI
    ├── embedding/embedder.py   # bge-large-zh-v1.5 wrapper
    ├── storage/milvus_store.py # Milvus vector store
    ├── retrieval/retriever.py  # Query-type-aware retrieval + reranker
    ├── ingestion/indexer.py    # PDF indexing pipeline
    ├── prompt_template.py      # RAFT prompt builder
    └── eval/                   # End-to-end smoke tests
```

---

## Requirements

- GPU: 16GB VRAM (tested on RTX 5080)
- Python 3.11 / 3.12
- Docker (for Milvus standalone)
- Two separate virtual environments (see below)

---

## Setup

### 1. Milvus

```bash
cd rag_service
bash standalone_embed.sh start
```

### 2. RAG service environment

```bash
cd rag_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Fine-tuning / inference environment

```bash
cd finetuning/qwen3
python3 -m venv .venv
source .venv/bin/activate
pip install axolotl torch transformers
```

### 4. Environment variables

Create `rag_service/.env`:

```env
LLM_BASE_URL=http://127.0.0.1:8001
LLM_MODEL=qwen3-raft
```

For evaluation scripts, set:

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
```

---

## Building the Knowledge Base

Place PDF files under `rag_service/sourcepdf/{company}/{category}/`.

```bash
cd rag_service
source .venv/bin/activate
python3 -c "from ingestion.indexer import build_index; build_index('sourcepdf')"
```

Indexing 247 PDFs takes ~1 hour on CPU embedding. Progress is printed per file.

---

## Running the System

Start all three components (separate terminals):

**Terminal 1 — Milvus**
```bash
cd rag_service && bash standalone_embed.sh start
```

**Terminal 2 — RAFT inference server (port 8001)**
```bash
cd finetuning/qwen3
source .venv/bin/activate
uvicorn serve.server:app --host 0.0.0.0 --port 8001
```

Model loads from `outputs/qwen3-4b-raft-v2-merged`. Takes ~30s.

**Terminal 3 — API + frontend (port 8000)**
```bash
cd rag_service
source .venv/bin/activate
LLM_BASE_URL=http://127.0.0.1:8001 uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Fine-tuning Pipeline

Three sequential LoRA stages, all run from `finetuning/qwen3/` with venv activated:

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| Stage 1 (Clause) | `axolotl train configs/stage1_clause/qwen3-4b-clause.yaml` | clause JSONL | `outputs/qwen3-4b-lora-clause` |
| Stage 2 (DB-CoT) | `axolotl train configs/stage2_db/qwen3-4b.yaml` | db_train.json | `outputs/qwen3-4b-lora-stage2` |
| Stage 3 (RAFT) | `axolotl train configs/stage3_raft/qwen3-4b-raft.yaml` | raft_train.jsonl | `outputs/qwen3-4b-lora-raft-v2` |

Merge LoRA weights before serving:

```bash
python3 serve/merge_lora.py
# outputs to outputs/qwen3-4b-raft-v2-merged
```

Regenerate RAFT training data:

```bash
python3 data/raft/generate_raft.py
```

---

## Evaluation

**End-to-end smoke test (15 HK insurance questions):**

```bash
cd rag_service
source .venv/bin/activate
export DEEPSEEK_API_KEY=sk-your-key
python3 eval/scripts/smoke_test_10.py
# Results saved to eval/results/smoke_test_15_v5.json
```

Latest results (v5, top_k=8, DeepSeek judge):

| Metric | Score |
|--------|-------|
| PASS | 9 / 15 |
| avg score | 8.07 / 12 |
| ERROR | 0 |

**RAFT fine-tuning evaluation (3-model comparison):**

```bash
cd finetuning/qwen3
source .venv/bin/activate
python3 eval/scripts/eval_raft.py \
  --base   Qwen3-4B \
  --lora2  outputs/qwen3-4b-lora-stage2 \
  --lora3  outputs/qwen3-4b-lora-raft-v2 \
  --clause_data  eval/data/clause.jsonl \
  --hall_data    eval/data/hallucination_eval.jsonl \
  --output_dir   eval/results/raft_eval_v4 \
  --n_clause 30 --n_distractor 15
```

---

## Hardware Notes

- Embedding + reranker run on **CPU** to avoid VRAM contention with the LLM
- RAFT model uses ~9.5GB VRAM; peak during inference ~10.5GB
- `top_k=8` is the recommended setting for 16GB GPUs (tested stable)
- `top_k=15` causes OOM at inference time on 16GB

---

## Known Limitations

- Model may hallucinate when evidence chunks are too short or partial
- Product-specific queries (e.g. exact premium tables) require the relevant PDF to be indexed
- SQL/database QA capability requires Stage 2 LoRA (included in merged model)
