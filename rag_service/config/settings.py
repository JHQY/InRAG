'''Configuration file placeholder'''
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
   PROJECT_NAME: str = "IRAG"
   MILVUS_HOST: str = os.getenv("MILVUS_HOST", "127.0.0.1")
   MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
   MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "IRAG")
   MILVUS_DIM: int = int(os.getenv("MILVUS_DIM", 768))
   MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "IP")
   MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
   EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
   DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))

settings = Settings()                          