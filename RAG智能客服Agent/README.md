# 🤖 RAG智能客服Agent
## ⚙️ 技术栈：LangChain（Tool Calling / ReAct）、Pinecone Vector DB、OpenAI Embedding（text-embedding-3-small）
设计思路：

根据PRD需求选择**GPT-4o模型**

采用ChatPromptTemplate+MessagesPlaceholder方法构建Policy & History & Input & Tool-calling Prompt

添加基于语义检索的**RAG（512 token chunking）**

并对多轮对话添加**记忆管理**

开发**四种工具调用**（知识库、订单查询、退款、转人工）

根据客服的工作模式，选择**LangChain**作为开发框架

结合**ReAct推理模式**，交付可上线 MVP（目前仅支持后端）

```text
day1-rag-agent/
├── src/
│   ├── rag_engine.py       # RAG
│   └── agent.py            # Agent
├── data/
│   └── knowledge_base.txt  # 自建知识库
├── tests/                  # 放测试用例
├── app.py                  # Streamlit UI前端，可补充
├── requirements.txt        # 依赖
└── README.md
