import chromadb

# 创建客户端（持久化模式）
client = chromadb.PersistentClient(path="./my_db")
collection = client.get_or_create_collection(name="test")

# 查询（无需任何大模型）
results = collection.query(
    query_texts=["动物有哪些"],
    n_results=1
)
print(results['documents'][0])  # 输出：['苹果是一种水果']

