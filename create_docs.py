import os

# 创建示例文档
os.makedirs("docs", exist_ok=True)
with open("docs/sample.txt", "w", encoding="utf-8") as f:
    f.write("""
检索增强生成（RAG）是一种结合了检索和生成方法的自然语言处理技术。
它先从知识库中检索相关文档，再基于这些文档生成回答。
RAG 可以有效减少大模型的幻觉问题，提高回答的准确性和可靠性。
""")
