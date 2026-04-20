import pdfplumber
from ingestion.cleaner import TableCleaner
from embedding.embedder import Embedder

# ==== 1. 配置 PDF 路径 ====
pdf_path = r"sourcepdf\\prudential\\medical\\prumed-lifelong-care-plan-product-brochure.pdf"

cleaner = TableCleaner()
embedder = Embedder()

print("\n==============================")
print("STEP 1: 解析 PDF 页内表格")
print("==============================")

with pdfplumber.open(pdf_path) as pdf:
    for page_idx, page in enumerate(pdf.pages, start=1):
        print(f"\n--- Page {page_idx} ---")

        tables = page.extract_tables()
        if not tables:
            print("❌ 没检测到表格")
            continue

        print(f"✔ 检测到 {len(tables)} 个原始表格")

        for tbl_idx, tbl in enumerate(tables):
            print(f"\n>>> 原始表格 #{tbl_idx} 内容（前 3 行预览）:")
            for r in tbl[:3]:
                print(r)

            header = tbl[0]
            rows = tbl[1:]

            # STEP 2: 清洗表格
            print("\n==============================")
            print("STEP 2: 清洗表格 clean_table")
            print("==============================")
            df, text_version = cleaner.clean_table(header, rows)

            if df is None:
                print("❌ cleaner 清洗失败 → fallback 文本")
                print("fallback 文本预览:")
                print(text_version[:300], "...")
            else:
                print("✔ cleaner 生成 DataFrame:")
                print(df.head())

                # STEP 3: 送入 TAPAS embedding
                print("\n==============================")
                print("STEP 3: 表格 embedding")
                print("==============================")

                try:
                    header = df.columns.tolist()
                    rows = df.values.tolist()
                    vec = embedder.embed_table(header, rows)
                    print("✔ 表格 embedding OK，向量维度:", len(vec))
                except Exception as e:
                    print("❌ embed_table 错误：", e)
