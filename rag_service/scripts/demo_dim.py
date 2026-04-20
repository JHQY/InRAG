from embedding.embedder import Embedder
import numpy as np

if __name__ == "__main__":
    embedder = Embedder()
    text = "测试一下 text embedding 真实输出"

    vec = embedder.embed_text([text])
    print("embed_text([text]) 返回对象类型：", type(vec))
    print("embed_text([text]) shape:", np.array(vec).shape)

    v = vec[0]
    print("vec[0] 的类型：", type(v))
    print("vec[0] 的 shape:", np.array(v).shape)
    print("vec[0] 的前 5 项：", np.array(v)[:5])
