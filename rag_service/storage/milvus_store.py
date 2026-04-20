# # storage/milvus_store.py
# from pymilvus import (
#     connections, FieldSchema, CollectionSchema,
#     DataType, Collection, utility
# )
# from config.settings import settings
# import numpy as np

# class Chunk:
#     """一个文本或表格块"""
#     def __init__(self, text, metadata):
#         self.text = text
#         self.metadata = metadata


# class MilvusVectorStore:
#     """
#     Milvus 向量存储与检索类
#     - 自动连接 Milvus
#     - 自动创建 collection
#     - 提供 add / search 功能
#     """

#     def __init__(self):
#         self.collection_name = settings.MILVUS_COLLECTION
#         self.dim = settings.MILVUS_DIM

#         # 连接 Milvus
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=str(settings.MILVUS_PORT)
#         )

#         # 检查 collection 是否存在
#         if not utility.has_collection(self.collection_name):
#             self._create_collection()

#         # 加载 collection
#         self.collection = Collection(self.collection_name)
#         self.collection.load()

#     # ------------------------------------------------------
#     # 创建 collection
#     # ------------------------------------------------------
#     def _create_collection(self):
#         print(f"[Milvus] Creating collection: {self.collection_name}")

#         fields = [
#             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#             FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
#             FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#             FieldSchema(name="metadata", dtype=DataType.JSON)
#         ]

#         schema = CollectionSchema(
#             fields=fields,
#             description="Insurance Knowledge Base"
#         )

#         collection = Collection(name=self.collection_name, schema=schema)

#         # 创建索引
#         index_params = {
#             "index_type": settings.MILVUS_INDEX_TYPE,
#             "metric_type": settings.MILVUS_METRIC_TYPE,
#             "params": {"M": 8, "efConstruction": 64}
#         }

#         collection.create_index(field_name="vector", index_params=index_params)
#         print(f"[Milvus] Collection `{self.collection_name}` created with index.")
#         return collection

#     # ------------------------------------------------------
#     # 插入数据
#     # ------------------------------------------------------
#     def add(self, embeddings, chunks):
#         """
#         向 Milvus 插入一批数据
#         参数：
#           embeddings: List[np.ndarray]  向量
#           chunks: List[Chunk]           对应文本块
#         """
#         if len(embeddings) == 0:
#             return

#         texts = [c.text for c in chunks]
#         metas = [c.metadata for c in chunks]

#         # 插入顺序必须与 collection 定义匹配
#         insert_data = [
#             #[None] * len(embeddings),  # auto_id 主键
#             embeddings,
#             texts,
#             metas
#         ]

#         self.collection.insert(insert_data)
#         self.collection.flush()
#         print(f"[Milvus] ✅ Inserted {len(embeddings)} records.")

#     # ------------------------------------------------------
#     # 向量检索
#     # ------------------------------------------------------
#     def similarity_search(self, query_embedding, top_k=5, filters=None):
#         """
#         执行相似度搜索
#         参数：
#           query_embedding: np.ndarray
#           top_k: 检索结果数
#           filters: dict，可按 metadata 过滤
#         返回：
#           [(Chunk, distance), ...]
#         """
#         search_params = {
#             "metric_type": settings.MILVUS_METRIC_TYPE,
#             "params": {"ef": 50}
#         }

#         expr = None
#         if filters:
#             expr = " and ".join([
#                 f'metadata["{k}"] == "{v}"' for k, v in filters.items()
#             ])

#         results = self.collection.search(
#             data=[query_embedding],
#             anns_field="vector",
#             param=search_params,
#             limit=top_k,
#             expr=expr,
#             output_fields=["text", "metadata"]
#         )

#         hits = []
#         for hit in results[0]:
#             text = hit.entity.get("text")
#             meta = hit.entity.get("metadata")
#             hits.append((Chunk(text, meta), float(hit.distance)))

#         return hits

# from pymilvus import (
#     connections, FieldSchema, CollectionSchema,
#     DataType, Collection, utility
# )
# from config.settings import settings
# import numpy as np
#
# class MilvusVectorStore:
#     """
#     多模态向量存储
#     支持：
#     - 文本向量 bge-m3
#     - 表格向量 TAPAS
#     """
#
#
#     def __init__(self):
#         self.collection_name = "IRAG_MM"
#         self.text_dim = 1024
#         self.table_dim = 768
#
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=settings.MILVUS_PORT,
#         )
#
#         if not utility.has_collection(self.collection_name):
#             self._create_collection()
#
#         self.collection = Collection(self.collection_name)
#         self.collection.load()
#
#
#     # ------------------------------------------------------------------
#     # 创建全新 IRAG_MM collection
#     # ------------------------------------------------------------------
#     def _create_collection(self):
#
#         print(f"[Milvus] Creating multi-modal collection: {self.collection_name}")
#
#         fields = [
#             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#
#             FieldSchema(
#                 name="text_vector",
#                 dtype=DataType.FLOAT_VECTOR,
#                 dim=self.text_dim,
#                 description="Text embedding (BGE-M3)"
#             ),
#
#             FieldSchema(
#                 name="table_vector",
#                 dtype=DataType.FLOAT_VECTOR,
#                 dim=self.table_dim,
#                 description="Table embedding (TAPAS)"
#             ),
#
#             FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#             #FieldSchema(name="table_json", dtype=DataType.JSON),
#             FieldSchema(name="table_blob",dtype=DataType.VARCHAR,max_length=65535), # to store table as string
#             FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
#             FieldSchema(name="metadata", dtype=DataType.JSON)
#         ]
#
#         schema = CollectionSchema(
#             fields=fields,
#             description="IRAG Multi-Modal Knowledge Base"
#         )
#
#         collection = Collection(self.collection_name, schema)
#
#         # 为两个向量字段分别创建索引
#         index_params = {
#             "index_type": "HNSW",
#             "metric_type": "COSINE",
#             "params": {"M": 8, "efConstruction": 64}
#         }
#
#         collection.create_index("text_vector", index_params)
#         collection.create_index("table_vector", index_params)
#
#         print("[Milvus] Multi-vector collection created.")
#
#     def add_records(self, records):
#         """
#         接收结构化 records，然后写入 Milvus（行模式）。
#         每条记录是一行。
#         """
#
#         import numpy as np
#
#         if not records:
#             return
#
#         text_dim = self.text_dim
#         table_dim = self.table_dim
#
#         # --------------------------------------------------
#         # 强力向量清洗器（递归 flatten + 强制 float + 定长）
#         # --------------------------------------------------
#         def sanitize_vec(v, dim):
#             if v is None:
#                 return [0.0] * dim
#
#             flat = []
#
#             def _flatten(x):
#                 if isinstance(x, (list, tuple, np.ndarray)):
#                     for e in x:
#                         _flatten(e)
#                 else:
#                     try:
#                         flat.append(float(x))
#                     except:
#                         flat.append(0.0)
#
#             _flatten(v)
#
#             if not flat:
#                 flat = [0.0] * dim
#
#             arr = np.array(flat, dtype="float32").reshape(-1)
#
#             if arr.shape[0] < dim:
#                 pad = np.zeros(dim - arr.shape[0], dtype="float32")
#                 arr = np.concatenate([arr, pad])
#             elif arr.shape[0] > dim:
#                 arr = arr[:dim]
#
#             return arr.tolist()
#
#         # --------------------------------------------------
#         # 构造 row-based 插入格式
#         # --------------------------------------------------
#         rows = []
#         for r in records:
#             tv = sanitize_vec(r.get("text_vec"), text_dim)
#             ttv = sanitize_vec(r.get("table_vec"), table_dim)
#
#             row = {
#                 "text_vector": tv,
#                 "table_vector": ttv,
#                 "text": r.get("text") or "",
#                 #"table_json": r.get("table_json") or {},
#                 "table_blob": r.get("table_blob") or "",
#                 "modality": r.get("modality") or "",
#                 "metadata": r.get("metadata") or {},
#             }
#             rows.append(row)
#
#         # --------------------------------------------------
#         # Debug：打印一条 sample 看看结构是否正确
#         # --------------------------------------------------
#         if rows:
#             print("\n[DEBUG] Example Insert Row:")
#             for k, v in rows[0].items():
#                 if isinstance(v, list):
#                     print(f"  {k}: list[{len(v)}]")
#                 else:
#                     print(f"  {k}: {v}")
#
#         # --------------------------------------------------
#         # 最终插入——行模式
#         # --------------------------------------------------
#         self.collection.insert(rows)
#         self.collection.flush()
#
#     # ------------------------------------------------------------------
#     # 搜索（默认 text_vector）
#     # ------------------------------------------------------------------
#     def search_text(self, query_vector, top_k=5):
#
#         results = self.collection.search(
#             data=[query_vector],
#             anns_field="text_vector",
#             param={"metric_type": "COSINE"},
#             limit=top_k,
#             output_fields=["text", "table_blob","modality", "metadata"]
#         )
#
#         return results[0]
#
#
#     # ------------------------------------------------------------------
#     # 搜索表格
#     # ------------------------------------------------------------------
#     def search_table(self, query_vector, top_k=5):
#
#         results = self.collection.search(
#             data=[query_vector],
#             anns_field="table_vector",
#             param={"metric_type": "COSINE"},
#             limit=top_k,
#             output_fields=["text",  "table_blob","modality", "metadata"]
#         )
#
#         return results[0]
# storage/milvus_store.py
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    DataType,
    CollectionSchema,
    utility
)
from config.settings import settings


class MilvusVectorStore:
    """
    IRAG 向量存储 (bge-large-zh-v1.5, 1024-dim, COSINE)
    Collection: rag_collection
    Schema: id, modality, text, table_blob, vector(1024), metadata
    """

    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.vector_dim = 1024
        self.text_dim = self.vector_dim
        self.table_dim = self.vector_dim

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )

        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)
        if self.collection.num_entities > 0:
            self.collection.load()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="table_blob", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, "IRAG 多模态保险知识库")
        collection = Collection(self.collection_name, schema)
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024},
            },
        )
        print(f"[Milvus] Collection '{self.collection_name}' created.")

    def add_records(self, records: list):
        import numpy as np

        def _to_vec(v, dim):
            """Flatten / pad / truncate to a plain float list of length dim."""
            if v is None:
                return [0.0] * dim
            if isinstance(v, np.ndarray):
                v = v.reshape(-1).astype("float32").tolist()
            elif not isinstance(v, list):
                v = list(v)
            if len(v) > dim:
                v = v[:dim]
            elif len(v) < dim:
                v = v + [0.0] * (dim - len(v))
            return v

        entities = []
        for record in records:
            # Support both "vector" (legacy) and "text_vec"/"table_vec" keys
            raw_vec = record.get("vector")
            if raw_vec is None:
                raw_vec = record.get("text_vec")
            if raw_vec is None:
                raw_vec = record.get("table_vec")
            vec = _to_vec(raw_vec, self.vector_dim)
            # Milvus VARCHAR max_length is in bytes. Truncate by encoding to be safe.
            raw_text = (record.get("text") or "")
            text_val = raw_text.encode("utf-8")[:65000].decode("utf-8", errors="ignore")
            raw_blob = (record.get("table_blob") or "")
            table_blob_val = raw_blob.encode("utf-8")[:65000].decode("utf-8", errors="ignore")
            entities.append({
                "modality": record.get("modality") or "",
                "text": text_val,
                "table_blob": table_blob_val,
                "vector": vec,
                "metadata": record.get("metadata") or {},
            })
        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_vector, modality: str = None, top_k: int = 10):
        try:
            self.collection.load()
        except Exception:
            pass
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
        expr = f"modality == '{modality}'" if modality else None
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "table_blob", "modality", "metadata"],
        )
        return results[0]
