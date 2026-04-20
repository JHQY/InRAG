


# scripts/drop_irag_mm.py
from pymilvus import connections, utility
from config.settings import settings

if __name__ == "__main__":
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )

    name = "IRAG_MM"
    if utility.has_collection(name):
        print(f"⚠️ Dropping collection: {name}")
        utility.drop_collection(name)
        print("✅ Dropped.")
    else:
        print(f"ℹ️ Collection {name} does not exist.")
