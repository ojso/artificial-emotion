import os
import shutil
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loader = TextLoader("docs/sample.txt", encoding="utf-8")
documents = loader.load()

# 如果需要加载 PDF，使用 PyPDFLoader
# loader = PyPDFLoader("docs/your_file.pdf")
# documents = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 中文友好分隔符
)
chunks = text_splitter.split_documents(documents)
print(f"文档已分割为 {len(chunks)} 个文本块")

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建向量数据库（持久化存储）
persist_directory = "./chroma_db"

# 核心覆盖逻辑：如果目录已存在，则强制删除
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"已清空旧向量数据库：{persist_directory}")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)


print(f"向量数据库已保存至 {persist_directory}")
