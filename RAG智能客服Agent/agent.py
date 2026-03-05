"""
Customer Service Agent with RAG
基于ReAct模式的智能客服Agent
"""
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import Dict, Any
import os


class CustomerServiceAgent:
    """智能客服Agent"""

    def __init__(self, use_rag: bool = True):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.use_rag = use_rag
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        """定义工具集"""

        # 模拟知识库
        knowledge_base = """
        产品功能：智能客服系统、订单查询、退款处理、物流查询
        退款政策：7天无理由退款，3-5个工作日到账
        订阅价格：基础版99元/月，专业版199元/月
        技术支持：7x24小时服务
        """

        # 工具1: RAG知识库检索
        def search_knowledge(query: str) -> str:
            """搜索产品知识库"""
            # 简化版：直接返回知识库内容
            return knowledge_base

        # 工具2: 查询订单
        def query_order(order_id: str) -> str:
            """查询订单状态"""
            # 模拟数据库查询
            orders = {
                "12345": "订单12345已发货，预计明天送达",
                "67890": "订单67890正在处理中",
                "11111": "订单11111已完成"
            }
            return orders.get(order_id, f"订单{order_id}不存在")

        # 工具3: 退款处理
        def process_refund(order_id: str) -> str:
            """处理退款申请"""
            return f"订单{order_id}的退款申请已提交，预计3-5个工作日到账"

        # 工具4: 转人工
        def transfer_to_human(reason: str) -> str:
            """转接人工客服"""
            return f"已为您转接人工客服，原因：{reason}。请稍候，人工客服将在1分钟内接入。"

        return [
            Tool(name="SearchKnowledge", func=search_knowledge,
                 description="搜索产品知识库、FAQ、使用指南。输入：搜索关键词"),
            Tool(name="QueryOrder", func=query_order,
                 description="查询订单状态和物流信息。输入：订单号"),
            Tool(name="ProcessRefund", func=process_refund,
                 description="处理退款申请。输入：订单号"),
            Tool(name="TransferHuman", func=transfer_to_human,
                 description="转接人工客服。输入：转接原因")
        ]

    def _create_agent(self):
        """创建ReAct Agent"""
        # 使用标准prompt
        prompt = hub.pull("hwchase17/openai-functions-agent")

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5  # 防止无限循环
        )

        return agent_executor

    def chat(self, user_input: str) -> str:
        """对话"""
        try:
            response = self.agent.invoke({
                "input": user_input
            })
            return response["output"]
        except Exception as e:
            return f"抱歉，处理您的请求时出现错误：{str(e)}"

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()


class SimpleRAGCustomerService:
    """
    简化版RAG客服（使用TF-IDF检索 + LLM生成）
    核心特性：真正的检索增强，而不是简单的Prompt模板
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # ✅ 真正的知识库（结构化文档）
        self.knowledge_base = [
            {
                "id": 1,
                "title": "产品功能介绍",
                "content": "我们的智能客服系统基于AI Agent技术，能够自动回答用户问题、查询订单状态、处理退款申请、查询物流信息。系统支持多轮对话，具备上下文记忆能力。"
            },
            {
                "id": 2,
                "title": "退款政策说明",
                "content": "我们支持7天无理由退款政策。用户提交退款申请后，系统将在1-3个工作日内审核，审核通过后3-5个工作日内原路退回。退款期间可随时查询进度。"
            },
            {
                "id": 3,
                "title": "订阅价格方案",
                "content": "我们提供三种订阅方案：基础版99元/月，包含基础客服功能；专业版199元/月，包含高级分析和多渠道支持；企业版价格面议，提供定制化服务和专属技术支持。"
            },
            {
                "id": 4,
                "title": "订单状态查询",
                "content": "用户可以通过订单号查询订单状态。支持的状态包括：待支付、已支付、待发货、已发货、已完成、已退款。已发货订单支持实时物流追踪。"
            },
            {
                "id": 5,
                "title": "技术支持服务",
                "content": "我们提供7x24小时技术支持服务。用户可以通过在线客服、邮箱、电话等方式联系我们。对于复杂问题，支持转接人工客服，平均响应时间1分钟。"
            },
            {
                "id": 6,
                "title": "物流配送说明",
                "content": "我们与顺丰速运、京东物流等主流快递公司合作，确保快速配送。订单发货后，用户可实时查看物流信息。大部分地区支持次日达和隔日达服务。"
            },
            {
                "id": 7,
                "title": "账户管理功能",
                "content": "用户可以管理个人信息、修改登录密码、绑定邮箱和手机号、设置安全问题。支持查看历史订单、下载发票、管理收货地址等功能。"
            },
            {
                "id": 8,
                "title": "企业版特色功能",
                "content": "企业版支持多用户协作、权限管理、数据报表、API接入、私有化部署。提供专属客户经理和定制化开发服务，适合大型企业和高并发场景。"
            }
        ]

        # ✅ 初始化TF-IDF检索器
        self._init_retriever()

    def _init_retriever(self):
        """初始化检索器（使用TF-IDF）"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            # ✅ 构建文档库
            documents = [doc["title"] + " " + doc["content"] for doc in self.knowledge_base]

            # ✅ 创建TF-IDF向量化器
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                stop_words=None  # 中文不使用英文停用词
            )

            # ✅ 训练并向量化文档
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.retriever_initialized = True

        except ImportError:
            print("⚠️  警告：scikit-learn未安装，将使用关键词匹配代替TF-IDF")
            self.retriever_initialized = False

    def _retrieve_relevant(self, query: str, top_k: int = 2):
        """
        检索相关知识（核心RAG逻辑）

        Args:
            query: 用户查询
            top_k: 返回top-k个最相关的文档

        Returns:
            检索到的相关文档列表
        """
        if not self.retriever_initialized:
            # 降级方案：关键词匹配
            return self._keyword_retrieve(query, top_k)

        # ✅ TF-IDF检索
        query_vec = self.vectorizer.transform([query])

        # ✅ 计算相似度（余弦相似度）
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # ✅ 获取top-k索引
        top_indices = similarities.argsort()[-top_k:][::-1]

        # ✅ 返回检索结果
        results = []
        for idx in top_indices:
            result = {
                **self.knowledge_base[idx],
                "score": float(similarities[idx])  # 相似度分数
            }
            results.append(result)

        return results

    def _keyword_retrieve(self, query: str, top_k: int = 2):
        """降级方案：基于关键词的简单检索"""
        query_lower = query.lower()

        # 计算每个文档的关键词匹配分数
        scored_docs = []
        for doc in self.knowledge_base:
            score = 0
            text = (doc["title"] + " " + doc["content"]).lower()

            # 简单的关键词匹配
            for char in query_lower:
                if char in text and char not in "的了吗是呢在啊吧哦哦：，。！？、":
                    score += 1

            scored_docs.append({**doc, "score": score})

        # 排序并返回top-k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def chat(self, user_input: str) -> str:
        """
        RAG增强的对话（核心：检索 → 生成）

        流程：
        1. 检索相关文档（retrieve）
        2. 构建提示词（prompt with context）
        3. LLM生成答案（generate）
        """
        # ✅ Step 1: 检索相关文档
        relevant_docs = self._retrieve_relevant(user_input, top_k=2)

        # ✅ Step 2: 构建上下文
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(
                f"[文档{i}] {doc['title']}\n{doc['content']}\n(相关度: {doc['score']:.2f})"
            )
        context = "\n\n".join(context_parts)

        # ✅ Step 3: RAG Prompt（检索增强生成）
        prompt = f"""你是智能客服助手。请基于以下检索到的知识库文档回答用户问题。

=== 检索到的知识库 ===
{context}

=== 用户问题 ===
{user_input}

=== 回答要求 ===
1. 答案必须基于上述知识库文档
2. 如果知识库中没有相关信息，明确告知用户
3. 保持准确、友好、专业
4. 可以适当整合多个文档的信息

请提供回答："""

        # ✅ Step 4: LLM生成
        response = self.llm.invoke(prompt)

        # ✅ 返回答案（附上检索信息）
        answer = response.content

        # 调试信息（可选）
        # print(f"🔍 检索到 {len(relevant_docs)} 个相关文档")
        # print(f"📊 最高相关度: {relevant_docs[0]['score']:.2f}")

        return answer

    def get_retrieval_stats(self, query: str) -> dict:
        """获取检索统计信息（用于调试和演示）"""
        relevant_docs = self._retrieve_relevant(query, top_k=2)

        return {
            "query": query,
            "num_retrieved": len(relevant_docs),
            "top_doc": {
                "title": relevant_docs[0]["title"],
                "score": relevant_docs[0]["score"]
            } if relevant_docs else None,
            "method": "TF-IDF" if self.retriever_initialized else "Keyword Matching"
        }


# 创建演示版Agent
def create_demo_agent(use_rag: bool = True):
    """
    创建演示版Agent

    Args:
        use_rag: 是否使用RAG版本（默认True）

    Returns:
        Agent实例
    """
    if use_rag:
        print("✅ 使用 SimpleRAGCustomerService (TF-IDF检索)")
        return SimpleRAGCustomerService()
    else:
        print("⚠️  使用完整版 CustomerServiceAgent (需要LangChain配置)")
        return CustomerServiceAgent(use_rag=True)


if __name__ == "__main__":
    print("初始化智能客服Agent...")

    # 创建简化版Agent用于演示
    agent = create_demo_agent()

    # 测试用例
    test_queries = [
        "你们的产品支持什么功能？",
        "查询订单12345",
        "我要退款",
        "转人工"
    ]

    print("\n=== 测试对话 ===\n")
    for query in test_queries:
        print(f"用户: {query}")
        response = agent.chat(query)
        print(f"客服: {response}\n")
        print("-" * 50)
