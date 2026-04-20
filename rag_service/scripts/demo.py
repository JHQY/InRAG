from embedding.embedder import Embedder
import numpy as np

def cosine_sim(a, b):
    return float(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))

def demo_table_channel():
    embedder = Embedder()

    query = "换地板可以赔多少？"
    q_vec = embedder.embed_query_table(query)

    # 表格转成 DataFrame
    import pandas as pd
    header = ["Benefit", "Coverage", "Deductible"]
    rows = [
        ["地板损坏赔偿", "80%",  "2000"],
        ["意外水损赔偿", "100%", "0"],
    ]
    df = pd.DataFrame(rows, columns=header)

    t_vec = embedder.embed_table(header, rows)

    score = cosine_sim(q_vec, t_vec)
    print(f"Query vs Table 相似度：{score:.4f}")

if __name__ == "__main__":
    demo_table_channel()
