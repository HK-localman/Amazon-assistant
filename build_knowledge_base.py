import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)


#当前项目的父亲目录
BASE_DIR =Path(__file__).resolve().parent

#商品资料文件位置
PRODUCT_FILE = BASE_DIR / "knowledge" / "products.json"

#chromadb数据库保存位置
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

#创建适合中英文资料的embedding函数
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    device="cpu",
    normalize_embeddings=True
)

#创建本地持久化chromadb客户端
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

#获取或创建商品知识库集合
collection = chroma_client.get_or_create_collection(
    name="product_knowledge",
    embedding_function=embedding_function
)

#读取json商品资料
with open (PRODUCT_FILE,"r",encoding="utf-8")as file:
    products = json.load(file)


ids = []
documents = []
metadatas = []

for product in products:

    product_id = product["product_id"]

    document = (
        f"商品编号：{product_id}\n"
        f"商品名称：{product['product_name']}\n"
        f"商品品类：{product['category']}\n"
        f"商品材质：{product['material']}\n"
        f"商品特点：{'、'.join(product['features'])}\n"
        f"使用场景：{'、'.join(product['scenarios'])}\n"
        f"目标站点：{product['marketplace']}\n"
        f"关键词：{', '.join(product['keywords'])}"
    )#文档

    metadata = {
        "product_id": product_id,
        "product_name":product["product_name"],
        "category":product["category"],
        "marketplace":product["marketplace"]
    }#元数据

    ids.append(product_id)
    documents.append(document)
    metadatas.append(metadata)

#将商品资料写入chromadb
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("商品知识库创建成功。")
print(f"本次写入的商品数量：{len(ids)}")
print(f"知识库当前总记录数：{collection.count()}")
print(f"数据库保存位置：{CHROMA_DB_DIR}")


