from pymilvus import MilvusClient
client = MilvusClient("http://127.0.0.1:19530")
print(client.list_collections())
