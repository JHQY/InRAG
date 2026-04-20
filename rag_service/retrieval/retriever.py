import numpy as np
import json
import zlib
import base64
from typing import List, Dict, Any
from embedding.embedder import Embedder
from retrieval.reranker import Reranker
from storage.milvus_store import MilvusVectorStore


class RAGInterface:
    def __init__(
            self,
            w_text: float = 1.0,
            w_table: float = 1.5,  # 【核心1】表格权重提高到1.5
            gamma: float = 0.6,  # 【核心2】降低重排权重，让检索权重更重要
            candidate_multiplier: int = 5,
            gamma_map: Dict[str, float] = None,
    ):
        print("Initializing RAG interface (fixed table ranking)...")
        self.embedder = Embedder()
        self.store = MilvusVectorStore()
        # 【核心3】大幅提高重排表格加成
        self.reranker = Reranker(debug=True, table_boost=0.5)

        self.w_text = w_text
        self.w_table = w_table
        self.gamma = gamma
        self.candidate_multiplier = candidate_multiplier

        self.gamma_map = gamma_map or {
            "text": 0.7,  # 文本查询：平衡
            "column": 0.5,  # 数值查询：更看重检索权重
            "row": 0.5,
            "table": 0.5
        }

    @staticmethod
    def classify_query(query: str) -> str:
        q = query.lower()
        column_keywords = ["金额", "赔付", "保额", "保费", "免赔额", "报销比例", "多少", "多少钱"]
        row_keywords = ["哪一个", "哪个", "哪种", "哪类", "哪项"]
        table_keywords = ["表格", "表中", "表里", "数据表", "费率表"]

        if any(k in q for k in table_keywords):
            return "table"
        if any(k in q for k in column_keywords):
            return "column"
        if any(k in q for k in row_keywords):
            return "row"
        return "text"

    def _decompress_table(self, blob: str) -> Dict:
        if not blob:
            return {}
        try:
            data = base64.b64decode(blob)
            raw = zlib.decompress(data)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[DEBUG] 表格解压失败: {e}")
            return {}

    def _build_hit_id(self, item: Dict) -> str:
        import hashlib
        meta = item.get("metadata") or {}
        payload = item.get("text", "") + json.dumps(item.get("table", {}), ensure_ascii=False)
        payload_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return "|".join([
            str(meta.get("source", "x")),
            f"p{meta.get('page_number', '0')}",
            str(item.get("modality", "")),
            payload_hash,
        ])

    def _build_context_text(self, item: Dict) -> str:
        parts = []
        text = item.get("text") or ""
        if text:
            parts.append(text)

        table = item.get("table") or {}
        if table and "header" in table and "rows" in table:
            header = [str(cell).strip() for cell in table["header"]]
            rows = [[str(cell).strip() for cell in row] for row in table["rows"]]

            parts.append("\n\n=== 完整表格数据 ===")
            parts.append("| " + " | ".join(header) + " |")
            parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in rows[:20]:
                parts.append("| " + " | ".join(row) + " |")
            if len(rows) > 20:
                parts.append(f"| ... 共{len(rows)}行，省略{len(rows) - 20}行 |")

        return "\n".join(parts)

    def retrieve(self, query: str, top_k: int = 5):
        if not query:
            return []

        qtype = self.classify_query(query)
        print(f"\n[DEBUG] 查询类型: {qtype}")
        k_each = max(top_k * self.candidate_multiplier, top_k)
        q_vec = self.embedder.embed_text([query])[0]

        hits = []
        # text 始终搜足量（保险条款主要在 text modality）
        hits += self.store.search(q_vec, modality="text", top_k=k_each)
        if qtype == "column":
            hits += self.store.search(q_vec, modality="column", top_k=k_each)
            hits += self.store.search(q_vec, modality="table", top_k=k_each // 2)
        elif qtype == "row":
            hits += self.store.search(q_vec, modality="row", top_k=k_each)
            hits += self.store.search(q_vec, modality="table", top_k=k_each // 2)
        elif qtype == "table":
            hits += self.store.search(q_vec, modality="table", top_k=k_each)
            hits += self.store.search(q_vec, modality="column", top_k=k_each // 2)
        else:
            # 文本查询：text 已搜，补充 table
            hits += self.store.search(q_vec, modality="table", top_k=k_each // 2)

        # 兜底
        if not hits:
            hits = self.store.search(q_vec, modality="text", top_k=k_each * 2)

        # 【核心6】双重过滤：空Blob + 低相关性
        filtered_hits = []
        for hit in hits:
            modality = hit.entity.get("modality", "text")
            score = hit.score
            table_blob = hit.entity.get("table_blob", "")

            # 1. 强制过滤空Blob的表格
            if modality in ["table", "column", "row"] and not table_blob:
                continue
            # 2. 过滤低相关性表格
            if modality in ["table", "column", "row"] and score > 0.6:
                continue

            filtered_hits.append(hit)
        hits = filtered_hits

        # 打印详细命中统计
        text_count = len([h for h in hits if h.entity.get('modality') == 'text'])
        table_count = len([h for h in hits if h.entity.get('modality') != 'text'])
        print(f"[DEBUG] 过滤后命中: 文本={text_count} | 有效表格={table_count}")

        # 融合逻辑
        fusion_map = {}
        for rank, hit in enumerate(hits, start=1):
            ent = hit.entity
            meta = ent.get("metadata") or {}
            item = {
                "text": ent.get("text"),
                "table": self._decompress_table(ent.get("table_blob")),
                "metadata": meta,
                "modality": ent.get("modality", ""),
            }
            doc_id = self._build_hit_id(item)
            if doc_id not in fusion_map:
                fusion_map[doc_id] = {
                    "fusion_score": 0.0,
                    "item": item,
                }
            # 【核心7】表格权重更高
            current_weight = self.w_table if item["modality"] in ["table", "column", "row"] else self.w_text
            fusion_map[doc_id]["fusion_score"] += current_weight * (1.0 / rank)

        if not fusion_map:
            return []

        # 筛选候选
        fused_items = sorted(fusion_map.values(), key=lambda x: x["fusion_score"], reverse=True)
        fused_items = fused_items[: top_k * self.candidate_multiplier]

        # 重排
        candidate_dicts = [{"text": fi["item"]["text"], "table": fi["item"]["table"]} for fi in fused_items]
        candidate_modalities = [fi["item"]["modality"] for fi in fused_items]
        rerank_scores = self.reranker.rerank(query, candidate_dicts, candidate_modalities, query_type=qtype)

        # 分数融合
        final = []
        for i, (fi, fs, rs, mod) in enumerate(
                zip(fused_items, [x["fusion_score"] for x in fused_items], rerank_scores, candidate_modalities)):
            f_norm = fs / max([x["fusion_score"] for x in fused_items]) if max(
                [x["fusion_score"] for x in fused_items]) > 0 else 0.5
            r_norm = rs / max(rerank_scores) if max(rerank_scores) > 0 else 0.5
            gamma = self.gamma_map.get(mod, self.gamma)
            final_score = gamma * r_norm + (1 - gamma) * f_norm

            final.append({
                "text": fi["item"]["text"],
                "table": fi["item"]["table"],
                "metadata": fi["item"]["metadata"],
                "modality": mod,
                "score": round(float(1 - final_score), 4),
            })
            print(f"  [{i + 1}] {mod:6} | 融合分:{fs:.4f} | 重排分:{rs:.4f} | 最终分:{1 - final_score:.4f}")

        final.sort(key=lambda x: x["score"])
        return final[:top_k]

    def retrieve_context(self, query: str, top_k: int = 5):
        hits = self.retrieve(query, top_k=top_k)
        context_parts = []
        for idx, hit in enumerate(hits):
            # 给每个检索结果加编号，方便LLM识别
            context_parts.append(f"\n--- 参考资料 {idx + 1} ---")
            context_text = self._build_context_text(hit)
            if context_text:
                context_parts.append(context_text)
        return "\n".join(context_parts)