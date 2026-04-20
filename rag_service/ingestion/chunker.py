# '''Chunker for splitting parsed text blocks into smaller segments'''
# from typing import List, Dict

# def chunk_blocks(blocks: List[Dict], max_len: int = 500, overlap: int = 50):
#     """
#     Split text blocks into smaller chunks with overlap.
#     Each output chunk preserves metadata such as company/category/page number.
#     """

#     chunks = []

#     for block in blocks:
#         text = block.get("text", "").strip()
#         if not text:
#             continue

#         metadata = block.get("metadata", {})
#         # 自动从上层路径结构补齐公司与险种（如果存在）
#         if not metadata:
#             source = block.get("source", "")
#             if "/" in source or "\\" in source:
#                 parts = source.replace("\\", "/").split("/")
#                 if len(parts) >= 3:
#                     metadata = {
#                         "source": source,
#                         "company": parts[-3],
#                         "category": parts[-2],
#                         "page_number": block.get("page_number", None)
#                     }
#                 else:
#                     metadata = {"source": source}
#             else:
#                 metadata = {"source": source}

#         # 分词逻辑：优先按空格切分（英语）；若文本无空格，则按字符切
#         words = text.split() if " " in text else list(text)
#         if len(words) <= max_len:
#             chunks.append({"text": text, "metadata": metadata})
#             continue

#         # 滑动窗口切分
#         start = 0
#         while start < len(words):
#             end = min(start + max_len, len(words))
#             piece = " ".join(words[start:end]) if " " in text else "".join(words[start:end])
#             chunks.append({
#                 "text": piece,
#                 "metadata": metadata
#             })
#             start += max_len - overlap

#     return chunks

# def chunk_text(text: str, max_len: int = 500, overlap: int = 50) -> List[str]:
#     """
#     Split a single text string into smaller chunks with overlap.
#     """
#     chunks = []
#     words = text.split() if " " in text else list(text)
#     if len(words) <= max_len:
#         return [text]

#     start = 0
#     while start < len(words):
#         end = min(start + max_len, len(words))
#         piece = " ".join(words[start:end]) if " " in text else "".join(words[start:end])
#         chunks.append(piece)
#         start += max_len - overlap

#     return chunks

"""
Chunker for IRAG multi-modal pipeline.

- 文本块：根据 max_length 分段切块
- 表格块：保持结构，不进行 chunk
"""

import re

def chunk_blocks(blocks, max_length=500, overlap=50):
    """
    输入: 
        blocks = parse_pdf() 返回的结构化块列表
    输出:
        chunks = 切好/不切的 block 列表，每个 chunk 包含 text 或 table 结构
    """

    chunks = []

    for b in blocks:

        modality = b["modality"]

        # --- 1. 文本块：进行 chunk 切分 ---
        if modality == "text":
            text = b["text"].strip()
            if not text:
                continue

            # 将长文本按 max_length 切分
            text_chunks = split_text(text, max_length, overlap)

            for tc in text_chunks:
                chunks.append({
                    "modality": "text",
                    "text": tc,
                    "table": None,
                    "metadata": b["metadata"]
                })

        # --- 2. 表格块：不进行 chunk ---
        elif modality == "table":
            chunks.append({
                "modality": "table",
                "text": None,
                "table": b["table"],     # header + rows 保持不变
                "metadata": b["metadata"]
            })

        else:
            # 防未来扩展
            continue

    return chunks



def split_text(text, max_length=500, overlap=50):
    """
    将文本按照 max_length 切成多段
    """

    # 清理文本，防止奇怪的间距
    text = re.sub(r'\s+', ' ', text)

    words = text.split(" ")
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_length
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap 保留部分上下文

        if start < 0:
            start = 0

    return chunks
