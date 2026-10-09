"""OpenAI-compatible backend.

Delegates table imputation to any OpenAI-compatible chat API by serializing the
table as JSON and asking the model to return the missing cells. Falls back to
the CPU backend when no API key is configured, so the framework still runs.

Set ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` to use a real endpoint.
"""
import json
import os
from urllib import request

from .base import Backend, register_backend
from .cpu import CpuBackend


@register_backend("openai")
class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._fallback = CpuBackend(cfg)

    def impute(self, table):
        if not self.api_key:
            return self._fallback.impute(table)
        missing = [(r, c) for r, c in table.missing_cells()]
        if not missing:
            return {}
        prompt = (
            "You are a table imputation engine. Given the following table as JSON "
            "(cells equal to null are missing), return ONLY a JSON object mapping "
            '"row,col" -> value for every missing cell.\n\n'
            + json.dumps(table.to_records(), ensure_ascii=False)
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return self._parse(text, table)
        except Exception:
            return self._fallback.impute(table)

    @staticmethod
    def _parse(text, table):
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        raw = json.loads(m.group(0))
        out = {}
        for k, v in raw.items():
            try:
                r, c = (int(x) for x in k.split(","))
            except (ValueError, AttributeError):
                continue
            if 0 <= r < table.n_rows and 0 <= c < table.n_cols:
                out[(r, c)] = v
        return out

    def predict_categorical(self, row, col_idx, row_idx, columns, domain):
        return self._fallback.predict_categorical(row, col_idx, row_idx, columns, domain)

    def predict_numeric(self, row, col_idx, row_idx, columns):
        return self._fallback.predict_numeric(row, col_idx, row_idx, columns)

    def classify(self, table, target_col):
        return self._fallback.classify(table, target_col)

    def regress(self, table, target_col):
        return self._fallback.regress(table, target_col)

    def generate(self, columns, n_rows):
        return self._fallback.generate(columns, n_rows)

    def train_step(self):
        return 0.0
