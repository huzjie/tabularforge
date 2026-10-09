# 贡献指南

欢迎提交 PR。开发约定：

1. 核心库保持**零依赖**（只用标准库）。
2. 所有随机性走 `tabularforge/utils/stable.py`，保证确定性。
3. 新增后端需用 `@register_backend("name")` 注册并在 `backends/__init__.py` 显式导入。
4. 新增基准需在 `bench/tabarena.py` 的 `run_all` 里注册。

运行检查：

```bash
python -m compileall -q tabularforge
python -m unittest discover -s tabularforge/tests -t .
python -m tabularforge doctor
```
