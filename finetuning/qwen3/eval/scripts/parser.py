import fitz
import os
import json
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 提取文本，按 y 坐标排序，可处理多列格式"""
    doc = fitz.open(pdf_path)
    blocks = []

    for page in doc:
        text_dict = page.get_text("dict")   # 得到块级别结构
        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue  # 只要文字块

            for line in block["lines"]:
                for span in line["spans"]:
                    blocks.append({
                        "text": span["text"],
                        "x0": span["bbox"][0],
                        "y0": span["bbox"][1],
                        "font": span["font"],
                        "size": span["size"],
                    })

    # --- 关键：按y坐标排序，保持阅读顺序 ---
    blocks = sorted(blocks, key=lambda x: (round(x["y0"], 1), x["x0"]))

    # 拼接文本
    result = []
    last_y = None
    current_line = ""

    for blk in blocks:
        y = blk["y0"]
        text = blk["text"].strip()

        if not text:
            continue

        if last_y is None:
            last_y = y

        # 换行逻辑：y坐标变化大于2时换行
        if abs(y - last_y) > 2:
            result.append(current_line)
            current_line = text
        else:
            # 同一行
            if current_line == "":
                current_line = text
            else:
                current_line += " " + text

        last_y = y

    # 最后一行
    if current_line:
        result.append(current_line)

    return "\n".join(result)


def batch_process_pdf(input_dir, output_json="parsed_terms.jsonl"):
    """批量解析整个文件夹中的 PDF"""
    pdf_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    )

    output_path = os.path.join(input_dir, output_json)
    with open(output_path, "w", encoding="utf-8") as wf:
        for pdf in tqdm(pdf_files):
            path = os.path.join(input_dir, pdf)
            try:
                text = extract_text_from_pdf(path)
            except Exception as e:
                text = f"<<ERROR: {e} >>"

            item = {
                "file_name": pdf,
                "content": text
            }
            wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n解析完成：结果已保存到\n{output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True,
                        help="PDF 文件目录，例如 ~/IRF/eval/terms")
    args = parser.parse_args()

    batch_process_pdf(args.pdf_dir)
