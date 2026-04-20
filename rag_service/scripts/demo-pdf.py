from ingestion.parser import parse_pdf
from embedding.embedder import Embedder
import json

FILE = "sourcepdf\AXA\critical_illness\LPPM 774-2211C Smart Living  Living  Extra Living CI Product Brochure_HK&MA_Chi.pdf"

blocks = parse_pdf(FILE)
print(f"解析得到 {len(blocks)} 个 block")

tables = [b for b in blocks if b["modality"] == "table"]
print(f"其中表格数量: {len(tables)}\n")

embedder = Embedder()

for i, t in enumerate(tables, 1):
    print(f"\n=== 表格 {i} ===")

    header = t["table"]["header"]
    rows   = t["table"]["rows"]

    print("header:", header)
    print("rows (前3行):", rows[:3])

    # TAPAS embedding 测试
    try:
        vec = embedder.embed_table(header, rows)
        print("TAPAS embedding 维度:", len(vec))
    except Exception as e:
        print("❌ TAPAS 失败:", e)
