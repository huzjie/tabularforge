# 快速开始

```bash
# 1. 零依赖，无需安装
git clone https://github.com/huzjie/tabularforge.git
cd tabularforge

# 2. 全量冒烟（五后端 + 五基准 + 训练演示）
python -m tabularforge doctor

# 3. 跑基准
python -m tabularforge bench

# 4. 起服务
python -m tabularforge serve --port 8000
```

一行代码补全表格：

```python
from tabularforge.config import load_config
from tabularforge.engine import TabularEngine
from tabularforge.scm.synth import synthesize_table

engine = TabularEngine(load_config())
table, _, _ = synthesize_table(seed=42, n_columns=8, n_rows=64)
masked, cells = table.mask(mask_prob=0.2, seed=0)
filled, preds = engine.impute(masked)
print(filled.to_records()[:3])
```
