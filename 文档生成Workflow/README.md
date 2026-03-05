# 🖥️ 多步骤文档生成Workflow系统                                                  
## 📈 技术栈：LangGraph、LangChain、Prompt Engineering（目前仅支持后端）
🧱 设计思路：

根据需求选择**Claude 3.5 Sonnet模型**

将文档生成任务拆解为Planning→Drafting→Reviewing→Finalizing共4个**节点**

使用**PromptTemplate**分别定义4个核心Prompt

**基于LangGraph搭建状态机工作流**

做好**异常处理**，支持错误回退/重试

```text
day2-workflow/
├── src/
│   └── workflow.py         # 工作流实现
├── tests/                  # 放测试用例
├── app.py                  # Streamlit UI前端，可补充
├── requirements.txt        # 依赖
└── README.md
