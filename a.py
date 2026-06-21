import chromadb

# 创建客户端（持久化模式）
client = chromadb.PersistentClient(path="./my_db")
collection = client.get_or_create_collection(name="test")

# 直接存储文本（Chroma 自带的嵌入模型会自动向量化）
collection.add(
    documents=["苹果是一种水果", "猫是一种动物"],
    ids=["doc1", "doc2"]
)

# 查询（无需任何大模型）
results = collection.query(
    query_texts=["水果有哪些"],
    n_results=1
)
print(results['documents'][0])  # 输出：['苹果是一种水果']

