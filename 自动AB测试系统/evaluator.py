"""
LLM Evaluation Framework
LLM评估系统，支持自定义指标和A/B测试
"""
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np


@dataclass
class TestCase:
    """测试用例"""
    input: str
    expected_output: str
    metadata: Dict[str, Any]  # 元数据（类别、难度等）


@dataclass
class EvaluationResult:
    """评估结果"""
    test_case: TestCase
    actual_output: str
    score: float  # 0-1分
    metrics: Dict[str, float]
    latency_ms: float


class LLMEvaluator:
    """LLM评估器"""

    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1_score": self._f1_score,
            "semantic_similarity": self._semantic_similarity
        }

    def evaluate(
        self,
        model: Callable[[str], str],
        test_cases: List[TestCase],
        metrics: List[str] = None
    ) -> List[EvaluationResult]:
        """评估模型"""
        if metrics is None:
            metrics = ["accuracy", "f1_score"]

        results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    self._evaluate_single,
                    model,
                    test_case,
                    metrics
                )
                for test_case in test_cases
            ]

            for future in futures:
                results.append(future.result())

        return results

    def _evaluate_single(
        self,
        model: Callable[[str], str],
        test_case: TestCase,
        metrics: List[str]
    ) -> EvaluationResult:
        """评估单个测试用例"""
        # 执行推理
        start_time = time.time()
        actual_output = model(test_case.input)
        latency_ms = (time.time() - start_time) * 1000

        # 计算指标
        metric_scores = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                metric_scores[metric_name] = self.metrics[metric_name](
                    test_case.expected_output,
                    actual_output
                )

        # 计算总分
        score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0.0

        return EvaluationResult(
            test_case=test_case,
            actual_output=actual_output,
            score=score,
            metrics=metric_scores,
            latency_ms=latency_ms
        )

    def _accuracy(self, expected: str, actual: str) -> float:
        """准确率"""
        return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0

    def _f1_score(self, expected: str, actual: str) -> float:
        """F1分数（简化版：基于关键词匹配）"""
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words:
            return 0.0

        # 计算精确率和召回率
        common_words = expected_words & actual_words

        precision = len(common_words) / len(actual_words) if actual_words else 0.0
        recall = len(common_words) / len(expected_words) if expected_words else 0.0

        # F1分数
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _semantic_similarity(self, expected: str, actual: str) -> float:
        """语义相似度（简化版：基于TF-IDF）"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([expected, actual])

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            # 如果sklearn不可用，返回简单匹配分数
            return self._f1_score(expected, actual)


class ABTester:
    """A/B测试器"""

    def __init__(self):
        self.test_results = []

    def run_test(
        self,
        variants: Dict[str, Callable[[str], str]],
        test_cases: List[TestCase],
        metrics: List[str] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """运行A/B测试"""
        results = {}

        for variant_name, model in variants.items():
            print(f"\n📊 测试Variant: {variant_name}")

            # 评估每个variant
            evaluator = LLMEvaluator()
            eval_results = evaluator.evaluate(
                model=model,
                test_cases=test_cases,
                metrics=metrics or ["accuracy", "f1_score"]
            )

            results[variant_name] = eval_results

            # 打印统计
            scores = [r.score for r in eval_results]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            print(f"  平均分: {mean_score:.3f} ± {std_score:.3f}")

        self.test_results.append({
            "variants": variants,
            "test_cases": test_cases,
            "results": results
        })

        return results

    def generate_report(self) -> str:
        """生成A/B测试报告"""
        if not self.test_results:
            return "没有测试结果"

        report = "# A/B测试报告\n\n"

        for test in self.test_results:
            results = test["results"]

            report += "## 测试结果\n\n"
            report += "| Variant | 平均分 | 标准差 | 最佳 |\n"
            report += "|---------|--------|--------|------|\n"

            best_variant = None
            best_score = -1

            for variant_name, eval_results in results.items():
                scores = [r.score for r in eval_results]
                mean = np.mean(scores)
                std = np.std(scores)

                is_best = False
                if mean > best_score:
                    best_score = mean
                    best_variant = variant_name
                    is_best = True

                winner_mark = "✅" if is_best else ""
                report += f"| {variant_name} | {mean:.3f} | {std:.3f} | {winner_mark} |\n"

            report += f"\n**最佳Variant**: {best_variant} (得分: {best_score:.3f})\n\n"
            report += "---\n\n"

        return report


class SimpleEvaluator:
    """简化版评估器（用于演示）"""

    def __init__(self):
        self.evaluator = LLMEvaluator()
        self.ab_tester = ABTester()

    def create_test_cases(self) -> List[TestCase]:
        """创建示例测试用例"""
        return [
            TestCase(
                input="什么是AI Agent？",
                expected_output="AI Agent是能自主感知环境并采取行动的智能系统",
                metadata={"category": "定义", "difficulty": "easy"}
            ),
            TestCase(
                input="如何实现RAG？",
                expected_output="RAG通过向量数据库检索相关文档，结合LLM生成准确答案",
                metadata={"category": "技术", "difficulty": "medium"}
            ),
            TestCase(
                input="LangChain是什么？",
                expected_output="LangChain是开发LLM应用的框架，提供链、Agent、工具等组件",
                metadata={"category": "工具", "difficulty": "easy"}
            ),
            TestCase(
                input="什么是ReAct模式？",
                expected_output="ReAct是推理和行动结合的模式，通过Thought-Action-Observation循环解决问题",
                metadata={"category": "概念", "difficulty": "medium"}
            )
        ]

    def create_mock_models(self, use_real_llm: bool = False) -> Dict[str, Callable[[str], str]]:
        """
        创建测试模型

        Args:
            use_real_llm: 是否使用真实LLM（需要API Key）

        Returns:
            模型字典
        """
        if use_real_llm:
            # ✅ 真实LLM版本
            from langchain_openai import ChatOpenAI

            llm_gpt4 = ChatOpenAI(model="gpt-4o", temperature=0)
            llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            def real_gpt4(input_text: str) -> str:
                """真实的GPT-4模型"""
                response = llm_gpt4.invoke(input_text)
                return response.content

            def real_gpt35(input_text: str) -> str:
                """真实的GPT-3.5模型"""
                response = llm_gpt35.invoke(input_text)
                return response.content

            return {
                "GPT-4o (Real)": real_gpt4,
                "GPT-3.5-Turbo (Real)": real_gpt35
            }

        else:
            # ✅ 演示版：基于规则的伪LLM
            def rule_based_agent_high_quality(input_text: str) -> str:
                """
                高质量规则Agent（模拟较好的LLM）

                核心逻辑：基于关键词和模板生成高质量回答
                """
                # 知识库
                knowledge = {
                    "AI Agent": "AI Agent是能自主感知环境并采取行动以实现目标的智能系统。它具备感知、推理、决策和执行的能力，可以在复杂环境中独立完成任务。",
                    "RAG": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。它通过向量数据库检索相关文档，将检索结果作为上下文输入LLM，从而生成准确、有依据的回答，有效解决LLM幻觉问题。",
                    "LangChain": "LangChain是开发LLM应用的强大框架。它提供链（Chain）、代理（Agent）、工具（Tool）、记忆（Memory）等核心组件，支持快速构建复杂的AI应用，是LLM应用开发的主流框架。",
                    "ReAct": "ReAct是一种结合推理和行动的Agent模式。它通过Thought（思考）→Action（行动）→Observation（观察）的循环，让Agent能够动态规划行动步骤并调整策略，提高问题解决的准确性。"
                }

                # 关键词匹配
                for keyword, answer in knowledge.items():
                    if keyword.lower() in input_text.lower():
                        return answer

                # 兜底回答
                if "什么是" in input_text:
                    topic = input_text.replace("什么是", "").replace("？", "").replace("?", "").strip()
                    return f"{topic}是一个重要的AI概念。它结合了多个技术维度，包括数据处理、模型训练和推理优化等。"
                elif "如何" in input_text or "怎么" in input_text:
                    return "实现这个功能需要以下步骤：1. 需求分析 2. 架构设计 3. 代码实现 4. 测试验证 5. 部署上线。"
                else:
                    return "这是一个很有价值的问题。从技术和实践角度来看，我们需要综合考虑多个因素，包括场景需求、技术选型和工程实现等。"

            def rule_based_agent_low_quality(input_text: str) -> str:
                """
                低质量规则Agent（模拟较差的LLM）

                核心逻辑：简单的关键词匹配，回答简短
                """
                # 简化的关键词匹配
                if "AI Agent" in input_text or "Agent" in input_text:
                    return "Agent是一种程序。"
                elif "RAG" in input_text:
                    return "RAG是一种检索技术。"
                elif "LangChain" in input_text:
                    return "LangChain是一个Python库。"
                elif "ReAct" in input_text:
                    return "ReAct是一种模式。"
                else:
                    return "我不太确定。"

            return {
                "Model-A (高质量规则)": rule_based_agent_high_quality,
                "Model-B (低质量规则)": rule_based_agent_low_quality
            }

    def run_demo(self):
        """运行演示"""
        print("\n" + "="*60)
        print("🔍 LLM评估系统演示")
        print("="*60)

        # 创建测试用例
        test_cases = self.create_test_cases()
        print(f"\n✅ 创建了 {len(test_cases)} 个测试用例")

        # 创建模型
        models = self.create_mock_models()
        print(f"✅ 创建了 {len(models)} 个模型")

        # 运行A/B测试
        print("\n🚀 开始A/B测试...")
        results = self.ab_tester.run_test(
            variants=models,
            test_cases=test_cases,
            metrics=["accuracy", "f1_score", "semantic_similarity"]
        )

        # 生成报告
        print("\n" + "="*60)
        print("📊 测试报告")
        print("="*60)
        report = self.ab_tester.generate_report()
        print(report)

        return results


def create_demo_evaluator():
    """创建演示版评估器"""
    return SimpleEvaluator()


if __name__ == "__main__":
    print("初始化LLM评估系统...")

    # 创建并运行演示
    evaluator = create_demo_evaluator()
    evaluator.run_demo()
