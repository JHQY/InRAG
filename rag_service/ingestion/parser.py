# '''PDF parser placeholder'''
# import pdfplumber   
# from pdfminer.high_level import extract_text
# import re

# def parse_pdf(file_path):
#     all_chunks = []

#     try:
#         with pdfplumber.open(file_path) as pdf:
#             for page_num, page in enumerate(pdf.pages,start=1):
#                 text = page.extract_text() or ""
#                 text = re.sub(r'\s+', ' ', text).strip()
#                 if text:
#                     all_chunks.append({
#                         "page_number": page_num,
#                         "modality": "text",
#                         "text": text
#                     })
                
#                 tables = page.extract_tables()
#                 for tbl in tables:
#                     if not tbl or len(tbl) < 2:
#                         continue
#                     rows = []
#                     headers = tbl[0]
#                     for row in tbl[1:]:
#                         pairs = []
#                         for h, v in zip(headers, row):
#                             pairs.append(f"{h}: {v}")
#                         if pairs:
#                             rows.append(" | ".join(pairs))
#                     if rows:
#                         table_text = "table:\n" + "\n".join(rows)
#                         all_chunks.append({
#                             "page_number": page_num,
#                             "modality": "table",
#                             "text": table_text
#                         })
#     except Exception as e:
#         print(f"Error parsing PDF: {e}")

#     return all_chunks

import pdfplumber
from ingestion.cleaner import TableCleaner
cleaner = TableCleaner()


def parse_pdf(pdf_path):
    """
    返回 blocks 列表：
    每个 block 是：
    {
        "modality": "text" 或 "table",
        "text": "...",            # 文本块专用
        "table": {                # 表格块专用
            "header": [...],
            "rows": [...]
        },
        "metadata": {
            "page_number": ...,
            "source_file": ...
        }
    }
    """

    blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # --- 1. 文本块 ---
            text = page.extract_text()
            if text:
                blocks.append({
                    "modality": "text",
                    "text": text,
                    "table": None,
                    "metadata": {
                        "page_number": page_num,
                        "source_file": pdf_path
                    }
                })

            # --- 2. 表格块 ---
            tables = page.extract_tables()
            for tbl in tables:
                if not tbl or len(tbl) == 0:
                    continue

                header = tbl[0]
                rows = tbl[1:]

                df, text_version = cleaner.clean_table(header, rows)

                if df is not None:
                    # ✔ 成功 → 结构化表格
                    blocks.append({
                        "modality": "table",
                        "text": None,
                        "table": {
                            "header": df.columns.tolist(),
                            "rows": df.values.tolist()
                        },
                        "metadata": {
                            "page_number": page_num,
                            "source_file": pdf_path
                        }
                    })
                else:
                    # ❌ 失败 → 降级为 text
                    blocks.append({
                        "modality": "text",
                        "text": text_version,
                        "table": None,
                        "metadata": {
                            "page_number": page_num,
                            "source_file": pdf_path
                        }
                    })
    return blocks
