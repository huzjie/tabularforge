# 因果发现（加分能力）

因为合成表是从 DAG 生成的，我们可以顺带做「因果父发现」并与真值对拍：

```python
from tabularforge.scm.causal import discover_parents
from tabularforge.scm.synth import synthesize_table

table, specs, true_parents = synthesize_table(seed=42, n_columns=8, n_rows=256)
target = 5
discovered, scores = discover_parents(table.rows, target, [i for i in range(8) if i != 5])
print("true parents :", true_parents[target])
print("discovered   :", discovered)
```

方法是一个廉价的互信息代理：衡量每个候选列单独能多大程度解释目标列。这在真实业务
里可用于「哪个字段最影响这个指标」的特征重要性分析。
