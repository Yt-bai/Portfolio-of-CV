# 🛸 自研Agent评估与A/B测试系统
传统上，对大模型应用的评估测试依赖人工执行与汇总，自动化与标准化程度低，评估效率低、结果难复现，导致每次迭代优化时都浪费大量资源，拉低ROI，影响上线速度。

## 🪐 功能描述：

实现一套模块化LLM评估框架，支持A/B两种模型/工作流统一接入，标准化Callable[[str], str]推理接口，实现模型能力的可插拔对比。

## 🌐 细节实现：

对每个Variant并发计算每个测试用例的Accuracy、F1、TF-IDF-based Semantic Similarity及Latency，求平均值得到score1、score2、、、、、

对所有测试用例的平均值（score[i]）再求平均值，得到最终模型评分，记作Score

自动输出A/B测试报告，包含最终得分的平均值与标准差，均分更高者标记为最佳Variant

该框架支持指标扩展与自定义，确保评估体系具备可演进能力

目前仅支持后端

```text
day3-evaluation/
├── src/
│   └── evaluator.py       # 测试系统主体
├── data/
│   └── test_cases.json    # 放在线输入给A/B模型的测试用例
├── tests/                 # 放离线评估测试用例
├── app.py                 # Streamlit UI前端，可补充
├── requirements.txt       # 依赖
└── README.md
