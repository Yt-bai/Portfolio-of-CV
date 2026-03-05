"""
RAG Engine for Customer Service Agent
实现向量检索和知识库管理
"""
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from typing import List, Tuple

load_dotenv()


class RAGEngine:
    """RAG检索引擎"""

    def __init__(self, index_name: str = "customer-service"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_name = index_name
        self.vector_store = None

    def load_documents(self, path: str):
        """加载PDF文档并进行分块"""
        loader = PyPDFLoader(path)
        documents = loader.load()

        # Chunking策略：512 tokens + 50 tokens overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "?", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        return splits

    def create_index(self, documents):
        """创建向量索引并上传到Pinecone"""
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # 创建index（如果不存在）
        if self.index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding维度
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # 上传向量
        self.vector_store = Pinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.7) -> List:
        """检索相关文档（带相似度阈值过滤）"""
        if not self.vector_store:
            # 连接已有index
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(self.index_name)
            self.vector_store = Pinecone(
                index=index,
                embedding=self.embeddings
            )

        # 相似度检索
        results = self.vector_store.similarity_search_with_score(
            query, k=k
        )

        # 过滤低相似度结果（阈值=0.7）
        filtered = [
            doc for doc, score in results
            if score >= threshold
        ]

        return filtered


# 模拟数据生成器（用于演示）
def create_sample_knowledge_base():
    """创建示例知识库文档"""
    sample_text = """
    产品功能介绍

    1. 智能客服系统
    我们的智能客服系统基于AI Agent技术，能够自动回答用户问题、查询订单、处理退款。

    2. 订单查询功能
    用户可以通过订单号查询订单状态。支持的状态包括：待支付、已支付、已发货、已完成。

    3. 退款政策
    我们支持7天无理由退款。退款申请提交后，将在3-5个工作日内处理完成。

    4. 物流查询
    订单发货后，用户可以查询物流信息。我们与顺丰速运合作，确保快速配送。

    5. 账户管理
    用户可以管理个人信息、修改密码、绑定邮箱等功能。

    6. 产品定价
    我们采用订阅制定价，分为基础版（99元/月）、专业版（199元/月）、企业版（联系我们）。

    7. 技术支持
    提供7x24小时技术支持。遇到问题时，可以转接人工客服。
    """

    # 保存为文本文件
    with open("data/knowledge_base.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    return "data/knowledge_base.txt"


if __name__ == "__main__":
    # 测试RAG引擎
    print("初始化RAG引擎...")

    # 创建示例知识库
    os.makedirs("data", exist_ok=True)
    kb_path = create_sample_knowledge_base()

    print(f"知识库已创建: {kb_path}")
