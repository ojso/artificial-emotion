import chromadb
client = chromadb.Client()
collection = client.get_or_create_collection(name="my_knowledge_base")
collection.add(
    documents=["这是关于AI的第一篇文档", "这是关于RAG的第二篇文档"],
    metadatas=[{"source": "wiki"}, {"source": "blog"}], # 可选的元数据，用于过滤
    ids=["doc1", "doc2"]  # 每个文档的唯一ID
)
results = collection.query(
    query_texts=["RAG是什么?"],  # Chroma 会自动将你的问题向量化
    n_results=2                # 返回最相似的2条结果
)
print(results['documents'][0])

