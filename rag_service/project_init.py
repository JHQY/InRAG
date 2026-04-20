import os

# 项目主目录
ROOT = "insurance_kb"

# 目录层级
DIRS = [
    "config",
    "embedding",
    "storage",
    "retrieval",
    "ingestion",
    "scripts"
]

# 各模块下要创建的文件
FILES = {
    "": ["README.md", "requirements.txt", ".env.example", "interface.py"],
    "config": ["__init__.py", "settings.py"],
    "embedding": ["__init__.py", "embedder.py"],
    "storage": ["__init__.py", "milvus_store.py"],
    "retrieval": ["__init__.py", "retriever.py"],
    "ingestion": ["__init__.py", "loader.py", "parser.py", "chunker.py", "indexer.py"],
    "scripts": ["build_index.py"]
}

# 最小文件模板（可留空）
TEMPLATES = {
    "README.md": "# Insurance Knowledge Base\n\nAuto-generated project skeleton.",
    "requirements.txt": "torch\nsentence-transformers\npymilvus\npdfminer.six\nnumpy\npython-dotenv\ntqdm",
    ".env.example": "MILVUS_HOST=127.0.0.1\nMILVUS_PORT=19530\nMILVUS_COLLECTION=insurance_kb\nMILVUS_DIM=768\nEMBEDDING_MODEL_NAME=all-mpnet-base-v2",
    "__init__.py": "",
    "interface.py": "'''Unified interface placeholder'''",
    "settings.py": "'''Configuration file placeholder'''",
    "embedder.py": "'''Text embedding placeholder'''",
    "milvus_store.py": "'''Milvus storage placeholder'''",
    "retriever.py": "'''Retriever placeholder'''",
    "loader.py": "'''Loader placeholder'''",
    "parser.py": "'''PDF parser placeholder'''",
    "chunker.py": "'''Chunker placeholder'''",
    "indexer.py": "'''Indexer placeholder'''",
    "build_index.py": "'''Build index script placeholder'''"
}

def create_structure():
    print(f"📁 Initializing project at: {ROOT}/")
    os.makedirs(ROOT, exist_ok=True)

    for d in DIRS:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
        for f in FILES.get(d, []):
            file_path = os.path.join(path, f)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as fp:
                    fp.write(TEMPLATES.get(f, ""))
                print(f"  ✅ Created: {file_path}")

    # 顶层文件
    for f in FILES[""]:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(TEMPLATES.get(f, ""))
            print(f"  ✅ Created: {path}")

    print("\n🎉 Project skeleton generated successfully!")

if __name__ == "__main__":
    create_structure()
