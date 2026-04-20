from pymilvus import connections, utility
from config.settings import settings

connections.connect(
    alias="default",
    host=settings.MILVUS_HOST,
    port=str(settings.MILVUS_PORT)
)   

utility.drop_collection(settings.MILVUS_COLLECTION)
print(f"[Milvus] Dropped collection: {settings.MILVUS_COLLECTION}")