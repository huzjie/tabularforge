# 后端说明

| 后端 | 说明 | 可训练 | 依赖 |
|---|---|---|---|
| `mock` | 确定性可训练桩：`skill` 控制「查真值 vs 猜」的概率，`train_step` 单调升 skill | ✅ | 无 |
| `cpu` | 真实 TabFM 前向（编码器 + MLP），纯 Python | ❌ | 无 |
| `openai` | 走 OpenAI 兼容 Chat API，无 key 时回退 cpu | ❌ | 无 |
| `vllm` | vLLM 托管的 OpenAI 端点，未配置时回退 cpu | ❌ | 无 |
| `transformers` | 预留 HF 表格模型接入，未安装时回退 cpu | ❌ | `transformers`（可选） |

## 切换后端

```python
from tabularforge.config import load_config
from tabularforge.engine import TabularEngine

cfg = load_config()
cfg.backend = "cpu"
engine = TabularEngine(cfg)
```

CLI 也支持 `--backend cpu`。mock 后端是演示「训练曲线」用的：imputation 准确率会从
0.5 单调爬升到 1.0，因为它的 `skill` 决定它多频繁地查结构因果模型这个「真值 Oracle」。
