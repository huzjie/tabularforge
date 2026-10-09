# TabArena 风格基准

`python -m tabularforge bench` 会跑五项任务：

| 任务 | 指标 | 说明 |
|---|---|---|
| imputation | accuracy | 挖掉格子、补全、比对真值 |
| classification | accuracy | 预测分类目标列 |
| regression | rmse | 预测数值目标列 |
| generation | validity | 按 schema 生成表，检查合法性 |
| speed | speedup | 批式（缓存列统计）vs 逐格编码的运算量比 |

## 复现「更快」的口径

Kumo Tabular 相对 LimiX-2 快约 17× 的核心手段是**把列级统计缓存一次、所有格子复用**，
而不是每个格子都重新编码整行上下文。`bench/speed.py` 用确定性的运算量计数演示同一
思路：naive 是 `O(格子数 × 列数)`，批式是 `O(列数 + 行数 + 格子数)`。
