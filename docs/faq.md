# FAQ

**Q: 真的零依赖吗？**
A: 是的。核心库只用 Python 标准库（`hashlib`、`random`、`http.server`、`csv`、`json`）。
可选后端（openai/vllm/transformers）才需要对应环境变量或第三方库，未配置时自动回退 cpu。

**Q: mock 后端的准确率为什么是 0.5 -> 1.0？**
A: mock 是「训练曲线」的演示桩：它的 `skill` 决定它查结构因果模型真值的概率。初始 0.5，
`train_step` 单调加 0.05，到 1.0 时每次都查真值，准确率就是 1.0。

**Q: cpu 后端准确率不高，为什么？**
A: cpu 后端是真实但**极小的** TabFM（64 维、hash 权重、随机初始化、未训练），它展示的是
架构与数据流，不是预训练权重。真实部署应接 openai/vllm/transformers 后端或你自己的权重。

**Q: 能用在真实 CSV 上吗？**
A: 能。`examples/impute_csv.py` 读 CSV、推断 schema、补全缺失格、写出结果。
