from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import(
    SentenceTransformerEmbeddingFunction
)

#当前项目的根目录
BASE_DIR = Path(__file__).resolve().parent

#chromadb数据库位置
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

#使用与入库时相同的嵌入模型
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    device="cpu",
    normalize_embeddings=True
)

#连接已经存在的本地chromadb
chroma_client = chromadb.PersistentClient(
    name="product_knowledge",
    embedding_function=embedding_function
)

# 获取已经创建的商品知识库集合
collection = chroma_client.get_collection(
    name="product_knowledge",
    embedding_function=embedding_function
)

# 接收用户的检索问题
query = input("请输入需要检索的商品需求")

# 从知识库中寻找最相关的一件商品
results = collection.query(
    query_texts=[query],
    n_results=1,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

# 读取第一条检索结果
result_id = results["ids"][0][0]
result_document = results["documents"][0][0]
result_metadata = results["metadatas"][0][0]
result_distance = results["distances"][0][0]

print("\n检索完成。")
print(f"匹配记录ID：{result_id}")
print(f"商品名称：{result_metadata['product_name']}")
print(f"商品品类：{result_metadata['category']}")
print(f"目标站点：{result_metadata['marketplace']}")
print(f"向量距离：{result_distance}")
print(f"\n完整商品资料：\n{result_document}")