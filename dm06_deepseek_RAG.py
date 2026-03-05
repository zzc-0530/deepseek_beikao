
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# 构建知识库
def build_knowledge_base(texts: list[str]):
    """
    输入
    :param texts:
    :return:
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    db = Chroma.from_documents(chunks, embeddings)
    return db

def search_knowledge(db, query: str, k=3):
    """

    :param db:
    :param query:
    :param k:
    :return:
    """
    results = db.similarity_search(query, k=k)

    return "\n\n".join([r.page_content for r in results])
