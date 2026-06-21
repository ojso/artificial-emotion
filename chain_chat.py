import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 导入必要的库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI  # 推荐使用 openai 兼容接口

# langchain 生成RAG新方式
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. retriever
persist_directory = "./chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever()

# 2. 通过 OpenAI 兼容接口连接本地 llama.cpp server
# 注意：本地运行无需真实的 OpenAI Key，但参数不能为空，可随意填写
llm = ChatOpenAI(
    openai_api_key="EMPTY", 
    openai_api_base="http://localhost:8001/v1",  # 替换为您实际的 llama.cpp server 地址
    temperature=0.7,
    max_tokens=512
)

# 3. 构建 RAG 问答链

# 假设你已经有了 llm (你的模型) 和 retriever (检索器)
# llm = ChatOpenAI(base_url="...", api_key="...")
# retriever = vector_store.as_retriever()

# 3.1. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "用以下上下文来回答问题。如果你不知道答案，就说不知道。\n\n上下文：{context}"),
    ("human", "{input}"),
])

# 3.2. 创建 "填充文档" 链
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 3.3. 创建最终的检索链
chain = create_retrieval_chain(retriever, question_answer_chain)


# 4. 交互式问答循环

while True:
    query = input("\n? 请输入您的问题 (输入 'q' 退出): ")
    if query.lower() == 'q':
        break
    response = chain.invoke({"input": query})
    print(response['answer'])

