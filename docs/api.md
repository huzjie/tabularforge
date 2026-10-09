# HTTP API

启动：`python -m tabularforge serve --port 8000`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 后端健康 |
| GET | `/v1/models` | 列出可用后端（OpenAI 风格） |
| POST | `/v1/impute` | 补全缺失格 |
| POST | `/v1/classify` | 预测分类目标列 |
| POST | `/v1/regress` | 预测数值目标列 |
| POST | `/v1/generate` | 按 schema 生成表 |
| POST | `/v1/train` | 训练一步（可训练后端） |

## 示例

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/impute \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"a":1.0,"b":null},{"a":2.0,"b":null}]}'
```
