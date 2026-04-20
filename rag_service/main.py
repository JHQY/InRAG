from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# 连接 Milvus Lite
client = MilvusClient(uri="http://127.0.0.1:19530")
print(client.list_collections())

# 嵌入模型（你之前也是用这个）
embedder = SentenceTransformer("BAAI/bge-m3")

# 测试查询
query = "重疾险的等待期通常是多久？"

vec = embedder.encode([query])[0]

# 搜索前 5 条
results = client.search(
    collection_name="insurance_docs",
    data=[vec],
    limit=5,
    output_fields=["text", "source"]
)

print("\nQuery:", query)
for i, hit in enumerate(results[0], start=1):
    print(f"\n=== Hit {i} ===")
    print("Score:", hit["distance"])
    print("Source:", hit["entity"]["source"])
    print(hit["entity"]["text"][:200], "...")
