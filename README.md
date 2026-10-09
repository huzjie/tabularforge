# tabularforge — 表格基础模型（Tabular Foundation Model）框架

> 把 NVIDIA **Kumo Tabular** 的核心工程思路，做成一个零依赖、可直接运行的参考实现。
> 一句话：**给它一张填了一部分的表，它把剩下的格子补出来——不用训练、不碰真实数据。**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero-dep](https://img.shields.io/badge/dependencies-zero-orange.svg)](requirements.txt)

---

## 这个项目解决什么技术问题

表格数据（Excel/CSV/数据库表）一直是深度学习里最尴尬的形态：模型要么把每行当一条
样本、每列当特征（传统 ML），要么硬塞进语言模型（把表拍平成文本）。**表格基础模型
（Tabular Foundation Model）换了个思路**：像 BERT 学语言那样，用「挖掉格子再猜回来」
（masked cell modeling）去学表格的通用的表示，于是能零样本补全、分类、回归、甚至
按 schema 生成整张表。

Kumo Tabular 的关键工程决策是 **预训练数据完全来自结构因果模型（SCM）合成的人造表**，
而不是真实数据——这同时解决了隐私、数据量、标注三个老大难。本仓库把这套思路拆成
可复用、可运行的模块。

---

## 可复用的 4 个核心思路（干货）

### 1. Masked Cell Modeling —— 「表格版 BERT」

挖掉一个格子，用**同一行的其它格子**当上下文去猜它。训练目标就是掩码语言建模
（`tabularforge/model/mcm.py`）：

```python
from tabularforge.model.tabfm import TabFM
model = TabFM(dim=64)                      # 纯 Python、确定性、零依赖
ctx = model.encode_context(row, col_idx, row_idx, columns)  # 行内上下文
pred = model.predict_categorical(row, col_idx, row_idx, columns, domain)
```

要点：上下文 = 「值嵌入 + 列嵌入 + 行嵌入」三者相加（不是拼接），这样**任意形状的
表都坍缩到同一个定宽向量**，模型才能跨表泛化。

### 2. SCM 合成数据 —— 不碰真实数据的无限标注流

采样一张因果 DAG，按拓扑序让每列 = 父列的函数 + 噪声，于是每张表都自带真值：

```python
from tabularforge.scm.synth import synthesize_table
table, specs, parents = synthesize_table(seed=42, n_columns=8, n_rows=256)
# parents[j] 就是第 j 列的因果父集 —— 真值已知，天然可监督
```

这套数据的价值：**无限量**（DAG+噪声空间无限）、**零隐私**（全人造）、**可对拍**
（顺带能验证因果发现，见 `scm/causal.py`）。

### 3. 确定性可训练 Mock —— 一条干净的训练曲线

`mock` 后端用一个 `skill` 参数控制「查真值 Oracle vs 猜」的频率，`train_step` 让它
单调爬升，于是 imputation 准确率从 0.5 一路到 1.0：

```python
from tabularforge.bench.imputation import run_imputation
mock = get_backend("mock", cfg)
print(run_imputation(mock, cfg)["accuracy"])   # ~0.50
for _ in range(10):
    mock.train_step()                          # skill 单调 +0.05
print(run_imputation(mock, cfg)["accuracy"])   # ~1.00
```

技巧：训练信号**只升不降**（`skill = min(1.0, skill + 0.05)`），别用「对 +1 错 -1」，
否则 skill 低时会被压回 0。

### 4. 批式推理 —— 把列统计缓存一次、所有格子复用

Kumo 相对 LimiX-2 快约 17× 的手段：缓存列级/行级统计，而不是每个格子都重新编码整
行。`tabularforge/bench/speed.py` 用确定性的运算量计数演示同一思路（naive
`O(格×列)` → 批式 `O(列+行+格)`）。

---

## 快速开始

```bash
# 零依赖，克隆即可跑
git clone https://github.com/huzjie/tabularforge.git
cd tabularforge

python -m tabularforge doctor      # 五后端 + 五基准 + 训练演示全量冒烟
python -m tabularforge bench       # TabArena 风格基准
python -m tabularforge train       # 训练演示（准确率 0.5 -> 1.0）
python -m tabularforge serve       # 起 OpenAI 兼容 HTTP 服务（:8000）
```

补全一张 CSV：

```bash
python -m tabularforge.examples.impute_csv data.csv
```

---

## 后端

| 后端 | 说明 | 依赖 |
|---|---|---|
| `mock` | 确定性可训练桩（演示训练曲线） | 无 |
| `cpu` | 真实 TabFM 前向，纯 Python | 无 |
| `openai` | OpenAI 兼容 API，无 key 回退 cpu | 无 |
| `vllm` | vLLM 端点，未配置回退 cpu | 无 |
| `transformers` | HF 表格模型接入，未装回退 cpu | 可选 |

---

## HTTP API

```
GET  /health          POST /v1/impute      POST /v1/classify
GET  /v1/models       POST /v1/regress     POST /v1/generate      POST /v1/train
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/impute \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"a":1.0,"b":null},{"a":2.0,"b":null}]}'
```

---

## 目录结构

```
tabularforge/
  scm/        结构因果模型合成数据引擎 + 因果发现
  tabular/    Table/Column + 类型推断 + 细胞编码
  model/      TabFM 掩码建模 + 任务头
  backends/   五后端（mock/cpu/openai/vllm/transformers）
  engine/     统一入口
  train/      训练器 + 数据集
  bench/      TabArena 风格五任务基准
  serving/    OpenAI 兼容 stdlib HTTP
  cli/        CLI（doctor/impute/generate/train/bench/serve）
  examples/   可运行示例（CSV 补全 / 因果发现 / 服务演示）
  tests/      单元测试
  configs/    默认配置
docs/         架构 / MCM / SCM / 后端 / 基准 / API / 因果发现
prompts/      表格补全 / schema 生成 / 依赖分析提示词
recipes/      一键脚本
Dockerfile / docker-compose / k8s / helm / .github/workflows
```

## 许可

MIT。仅供学习与技术交流，所实现的「表格基础模型」为参考实现，不代表 NVIDIA Kumo
Tabular 官方代码。
