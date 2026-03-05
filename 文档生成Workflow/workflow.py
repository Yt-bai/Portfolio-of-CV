"""
Document Generation Workflow using LangGraph
基于状态机的多步骤文档生成系统
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from operator import add
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import time


class DocState(TypedDict):
    """文档生成状态"""
    requirement: str          # 需求
    outline: List[str]        # 大纲
    draft: str                # 草稿
    review_comments: str      # 审核意见
    final_doc: str            # 最终文档
    current_step: str         # 当前步骤
    errors: Annotated[List[str], add]  # 错误列表


class DocumentWorkflow:
    """文档生成工作流"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """构建状态机工作流"""
        workflow = StateGraph(DocState)

        # 添加节点
        workflow.add_node("planning", self.planning_node)
        workflow.add_node("drafting", self.drafting_node)
        workflow.add_node("reviewing", self.reviewing_node)
        workflow.add_node("finalizing", self.finalizing_node)

        # 添加边（定义流程）
        workflow.add_edge("planning", "drafting")
        workflow.add_edge("drafting", "reviewing")
        workflow.add_edge("reviewing", "finalizing")
        workflow.add_edge("finalizing", END)

        # 设置入口
        workflow.set_entry_point("planning")

        return workflow.compile()

    def planning_node(self, state: DocState) -> DocState:
        """规划节点：生成大纲"""
        print("📋 [Planning] 生成文档大纲...")

        prompt = PromptTemplate(
            input_variables=["requirement"],
            template="""基于以下需求生成文档大纲。

需求: {requirement}

请生成一个结构化的文档大纲（包括章节和小节），每个章节用数字编号。
格式示例：
1. 简介
2. 核心概念
  2.1 什么是XX
  2.2 为什么需要XX
3. 实现步骤
4. 总结

请直接输出大纲内容："""
        )

        chain = prompt | self.llm
        result = chain.invoke({"requirement": state["requirement"]})

        # 解析大纲
        outline_text = result.content.strip()
        outline = self._parse_outline(outline_text)

        print(f"✅ 生成了 {len(outline)} 个章节")

        return {
            **state,
            "outline": outline,
            "current_step": "PLANNING"
        }

    def drafting_node(self, state: DocState) -> DocState:
        """起草节点：生成初稿"""
        print("✍️  [Drafting] 生成文档初稿...")

        prompt = PromptTemplate(
            input_variables=["requirement", "outline"],
            template="""基于以下需求和大纲生成完整的技术文档。

需求: {requirement}

大纲:
{outline}

请生成完整的文档内容（约2000字），要求：
1. 内容详实，有技术深度
2. 结构清晰，逻辑严谨
3. 包含代码示例或案例
4. 专业术语准确

请生成文档："""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "requirement": state["requirement"],
            "outline": "\n".join(state["outline"])
        })

        print("✅ 文档初稿生成完成")

        return {
            **state,
            "draft": result.content,
            "current_step": "DRAFTING"
        }

    def reviewing_node(self, state: DocState) -> DocState:
        """审核节点：检查质量"""
        print("🔍 [Reviewing] 审核文档质量...")

        prompt = PromptTemplate(
            input_variables=["draft"],
            template="""作为技术文档审核专家，请审核以下文档，提供修改建议。

文档:
{draft}

请从以下角度审核：
1. 结构完整性 - 是否覆盖了所有必要内容
2. 内容准确性 - 技术细节是否准确
3. 表达清晰度 - 是否易于理解
4. 改进建议 - 具体的修改意见

请提供审核意见（500字以内）："""
        )

        chain = prompt | self.llm
        result = chain.invoke({"draft": state["draft"]})

        print("✅ 文档审核完成")

        return {
            **state,
            "review_comments": result.content,
            "current_step": "REVIEWING"
        }

    def finalizing_node(self, state: DocState) -> DocState:
        """定稿节点：整合审核意见"""
        print("✨ [Finalizing] 生成最终版本...")

        prompt = PromptTemplate(
            input_variables=["draft", "review_comments"],
            template="""基于审核意见修改文档，生成最终版本。

原稿:
{draft}

审核意见:
{review_comments}

请根据审核意见修改文档，生成最终版本。
保持原文优点，改进不足之处。

请生成最终文档："""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "draft": state["draft"],
            "review_comments": state["review_comments"]
        })

        print("✅ 最终文档生成完成")

        return {
            **state,
            "final_doc": result.content,
            "current_step": "FINALIZING"
        }

    def _parse_outline(self, outline_text: str) -> List[str]:
        """解析大纲文本为列表"""
        lines = outline_text.split("\n")
        outline = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # 移除编号
                clean_line = line
                for i in range(10):
                    clean_line = clean_line.replace(f"{i}.", "").replace(f"{i}", "", 1)
                outline.append(clean_line.strip())

        return outline if outline else ["简介", "核心概念", "实现步骤", "总结"]

    def run(self, requirement: str) -> DocState:
        """运行工作流"""
        print("\n" + "="*60)
        print("🚀 启动文档生成工作流")
        print("="*60 + "\n")

        initial_state = {
            "requirement": requirement,
            "outline": [],
            "draft": "",
            "review_comments": "",
            "final_doc": "",
            "current_step": "START",
            "errors": []
        }

        start_time = time.time()

        result = self.workflow.invoke(initial_state)

        elapsed = time.time() - start_time

        print("\n" + "="*60)
        print(f"✅ 工作流完成！总耗时: {elapsed:.2f}秒")
        print("="*60 + "\n")

        return result


class StepByStepWorkflow:
    """
    显式多步骤工作流（不依赖LangGraph，但展示真正的多步骤编排）
    核心特性：4个明确的独立步骤，每步可单独优化
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.steps = [
            "PLANNING",    # 生成大纲
            "DRAFTING",    # 生成初稿
            "REVIEWING",   # 审核质量
            "FINALIZING"   # 生成定稿
        ]

    def generate(self, requirement: str) -> dict:
        """
        执行4步骤工作流

        流程：
        1. Planning - 分析需求，生成大纲
        2. Drafting - 基于大纲生成初稿
        3. Reviewing - 审核初稿，提出修改意见
        4. Finalizing - 整合审核意见，生成最终版本

        Args:
            requirement: 文档需求

        Returns:
            包含所有步骤结果和最终文档的字典
        """
        print("\n" + "="*60)
        print("🚀 启动4步骤文档生成工作流")
        print("="*60 + "\n")

        # ✅ Step 1: Planning（规划）
        print("📋 Step 1/4: PLANNING - 生成文档大纲")
        outline = self._planning_node(requirement)
        print(f"✅ 完成：生成了 {len(outline)} 个章节\n")

        # ✅ Step 2: Drafting（起草）
        print("✍️  Step 2/4: DRAFTING - 生成文档初稿")
        draft = self._drafting_node(requirement, outline)
        print(f"✅ 完成：初稿字数约 {len(draft)} 字\n")

        # ✅ Step 3: Reviewing（审核）
        print("🔍 Step 3/4: REVIEWING - 审核文档质量")
        review_comments = self._reviewing_node(draft)
        print(f"✅ 完成：审核完成\n")

        # ✅ Step 4: Finalizing（定稿）
        print("✨ Step 4/4: FINALIZING - 生成最终版本")
        final_doc = self._finalizing_node(draft, review_comments)
        print(f"✅ 完成：最终文档字数约 {len(final_doc)} 字\n")

        print("="*60)
        print("🎉 工作流完成！")
        print("="*60 + "\n")

        return {
            "requirement": requirement,
            "outline": outline,
            "draft": draft,
            "review_comments": review_comments,
            "final_doc": final_doc,
            "current_step": "FINALIZING",
            "steps_executed": self.steps
        }

    def _planning_node(self, requirement: str) -> list:
        """
        Planning节点：生成文档大纲

        核心逻辑：
        1. 分析需求，提取关键信息
        2. 生成结构化大纲（多级标题）
        3. 确保大纲覆盖所有必要内容
        """
        prompt = PromptTemplate(
            input_variables=["requirement"],
            template="""你是技术文档规划专家。请基于以下需求生成详细的文档大纲。

需求：{requirement}

要求：
1. 大纲要结构清晰，层次分明
2. 包含主要章节（一级标题）和子章节（二级标题）
3. 确保覆盖主题的各个方面
4. 使用数字编号格式

示例格式：
1. 简介
   1.1 背景介绍
   1.2 核心价值
2. 核心概念
   2.1 基本定义
   2.2 技术原理
3. 实现步骤
   3.1 环境准备
   3.2 代码实现
4. 总结

请直接输出大纲内容（每行一个章节标题）："""
        )

        chain = prompt | self.llm
        result = chain.invoke({"requirement": requirement})

        # 解析大纲文本为列表
        outline = self._parse_outline(result.content)

        return outline

    def _drafting_node(self, requirement: str, outline: list) -> str:
        """
        Drafting节点：生成文档初稿

        核心逻辑：
        1. 基于需求和大纲生成内容
        2. 确保内容详实、技术准确
        3. 包含代码示例或案例
        """
        prompt = PromptTemplate(
            input_variables=["requirement", "outline"],
            template="""你是技术文档作者。请基于以下需求和大纲生成完整的技术文档初稿。

需求：{requirement}

大纲：
{outline}

要求：
1. 内容详实，有技术深度
2. 结构清晰，逻辑严谨
3. 包含具体的代码示例或案例
4. 专业术语准确
5. 每个章节内容充分（约200-300字/章节）

请生成完整的文档初稿："""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "requirement": requirement,
            "outline": "\n".join(outline)
        })

        return result.content

    def _reviewing_node(self, draft: str) -> str:
        """
        Reviewing节点：审核文档质量

        核心逻辑：
        1. 从结构、内容、表达三个维度审核
        2. 提供具体的修改建议
        3. 指出优点和不足
        """
        prompt = PromptTemplate(
            input_variables=["draft"],
            template="""你是技术文档审核专家。请审核以下文档初稿，提供修改建议。

文档初稿：
{draft}

请从以下维度审核（提供500字以内的审核意见）：

1. 结构完整性
   - 是否覆盖了主题的各个方面
   - 章节安排是否合理
   - 逻辑是否连贯

2. 内容准确性
   - 技术细节是否准确
   - 是否有事实性错误
   - 代码示例是否正确

3. 表达清晰度
   - 语言是否易懂
   - 是否有歧义表达
   - 专业术语使用是否恰当

4. 改进建议
   - 具体的修改意见
   - 需要补充的内容
   - 需要优化的部分

请提供审核意见："""
        )

        chain = prompt | self.llm
        result = chain.invoke({"draft": draft})

        return result.content

    def _finalizing_node(self, draft: str, review_comments: str) -> str:
        """
        Finalizing节点：生成最终版本

        核心逻辑：
        1. 整合审核意见
        2. 修改初稿的不足
        3. 保持原有优点
        4. 生成高质量最终文档
        """
        prompt = PromptTemplate(
            input_variables=["draft", "review_comments"],
            template="""你是技术文档编辑。请基于审核意见修改文档，生成最终版本。

原稿：
{draft}

审核意见：
{review_comments}

要求：
1. 根据审核意见修改文档的不足之处
2. 保持原文的优点
3. 确保修改后文档质量明显提升
4. 语言专业、表达准确

请生成最终文档："""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "draft": draft,
            "review_comments": review_comments
        })

        return result.content

    def _parse_outline(self, outline_text: str) -> list:
        """解析大纲文本为列表"""
        lines = outline_text.split("\n")
        outline = []

        for line in lines:
            line = line.strip()
            # 保留有编号的行或以"-"开头的行
            if line and (line[0].isdigit() or line.startswith("-")):
                # 移除编号，保留内容
                clean_line = line
                for i in range(10):
                    clean_line = clean_line.replace(f"{i}.", "").replace(f"{i}", "", 1)
                clean_line = clean_line.strip()
                if clean_line:
                    outline.append(clean_line)

        # 如果解析失败，返回默认大纲
        if not outline:
            return ["简介", "核心概念", "实现步骤", "最佳实践", "总结"]

        return outline

    def get_workflow_stats(self) -> dict:
        """获取工作流统计信息"""
        return {
            "num_steps": len(self.steps),
            "steps": self.steps,
            "workflow_type": "Step-by-Step (Explicit)"
        }


def create_demo_workflow(use_explicit_steps: bool = True):
    """
    创建演示版工作流

    Args:
        use_explicit_steps: 是否使用显式多步骤版本（默认True）

    Returns:
        Workflow实例
    """
    if use_explicit_steps:
        print("✅ 使用 StepByStepWorkflow (4步骤显式流程)")
        return StepByStepWorkflow()
    else:
        print("⚠️  使用完整版 DocumentWorkflow (需要LangGraph)")
        return DocumentWorkflow()


if __name__ == "__main__":
    print("初始化文档生成工作流...")

    # 创建简化版工作流
    workflow = create_demo_workflow()

    # 测试用例
    requirement = "写一篇关于RAG技术的技术文档，包括原理、架构和实现步骤"

    result = workflow.generate(requirement)

    print("\n" + "="*60)
    print("📄 生成的文档")
    print("="*60 + "\n")
    print(result["final_doc"])
