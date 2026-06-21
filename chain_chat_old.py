import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 导入必要的库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI  # 推荐使用 openai 兼容接口
from langchain_classic.chains import RetrievalQA

# 1. 加载已有的 Chroma 向量库
persist_directory = "./chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 2. 通过 OpenAI 兼容接口连接本地 llama.cpp server
# 注意：本地运行无需真实的 OpenAI Key，但参数不能为空，可随意填写
llm = ChatOpenAI(
    openai_api_key="EMPTY", 
    openai_api_base="http://localhost:8001/v1",  # 替换为您实际的 llama.cpp server 地址
    temperature=0.7,
    max_tokens=512
)

# 3. 构建 RAG 问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",    # 将检索到的上下文一次性塞给 LLM
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 3}   # 每次检索返回最相关的 3 个文本块
    ),
    return_source_documents=True   # 返回引用的源文档，便于追溯
)


# 4. 交互式问答循环

while True:
    query = input("\n? 请输入您的问题 (输入 'q' 退出): ")
    if query.lower() == 'q':
        break
    result = qa_chain.invoke({"query": query})
    print(f"\n💡 回答:\n{result['result']}")
    print("\n📚 参考片段:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"[{i+1}] {doc.page_content[:100]}...")

