"""OpenAI-compatible stdlib HTTP server (zero dependencies).

Endpoints
---------
GET  /health                 -> backend health
GET  /v1/models              -> list available backends (OpenAI-style)
POST /v1/impute              -> impute missing cells of a table
POST /v1/classify            -> predict a categorical target column
POST /v1/regress             -> predict a numeric target column
POST /v1/generate            -> generate a table from a schema
POST /v1/train               -> one training step (for the trainable backend)

Body format (impute/classify/regress)::
    {"records": [{"a": 1, "b": null}, ...], "columns": [{"name": "a", "dtype": "numeric", ...}], "target_col": "b"}

``columns`` and ``target_col`` are optional; the schema is inferred when absent.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..engine import TabularEngine
from ..tabular.schema import infer_schema
from ..tabular.table import Table, Column
from .schemas import json_body, send_json


def _to_columns(raw_columns):
    if not raw_columns:
        return None
    return [Column(c.get("name", f"c{i}"), c.get("dtype", "text"), c.get("domain"))
            for i, c in enumerate(raw_columns)]


class TabularRequestHandler(BaseHTTPRequestHandler):
    engine = None  # set by serve()

    def _engine(self):
        return self.engine or TabularRequestHandler.engine

    def log_message(self, fmt, *args):  # silence default logging
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            return send_json(self, {"status": "ok", **self._engine().health()})
        if self.path.startswith("/v1/models"):
            from ..backends import list_backends
            return send_json(self, {"object": "list", "data": [
                {"id": name, "object": "model"} for name in list_backends()]})
        return send_json(self, {"error": "not found"}, status=404)

    def do_POST(self):
        body = json_body(self)
        engine = self._engine()
        try:
            if self.path.startswith("/v1/impute"):
                records = body.get("records", [])
                columns = _to_columns(body.get("columns"))
                out = engine.impute_records(records, columns)
                return send_json(self, {"records": out})
            if self.path.startswith("/v1/classify"):
                records = body.get("records", [])
                columns = _to_columns(body.get("columns")) or infer_schema(records)
                table = Table.from_records(records, columns)
                target = body.get("target_col", 0)
                target_idx = target if isinstance(target, int) else                     next((i for i, c in enumerate(columns) if c.name == target), 0)
                return send_json(self, {"predictions": engine.classify(table, target_idx)})
            if self.path.startswith("/v1/regress"):
                records = body.get("records", [])
                columns = _to_columns(body.get("columns")) or infer_schema(records)
                table = Table.from_records(records, columns)
                target = body.get("target_col", 0)
                target_idx = target if isinstance(target, int) else                     next((i for i, c in enumerate(columns) if c.name == target), 0)
                return send_json(self, {"predictions": engine.regress(table, target_idx)})
            if self.path.startswith("/v1/generate"):
                columns = _to_columns(body.get("columns")) or [Column("c0", "numeric"), Column("c1", "categorical", ["a", "b"])]
                n_rows = int(body.get("n_rows", 8))
                rows = engine.generate(columns, n_rows)
                return send_json(self, {"records": [
                    {c.name: r[i] for i, c in enumerate(columns)} for r in rows]})
            if self.path.startswith("/v1/train"):
                skill = engine.train_step()
                return send_json(self, {"skill": skill})
            return send_json(self, {"error": "not found"}, status=404)
        except Exception as exc:  # pragma: no cover - defensive
            return send_json(self, {"error": str(exc)}, status=400)


def serve(cfg, host="127.0.0.1", port=8000):
    engine = TabularEngine(cfg)
    TabularRequestHandler.engine = engine
    server = HTTPServer((host, port), TabularRequestHandler)
    print(f"[tabularforge] serving on http://{host}:{port} (backend={cfg.backend})", flush=True)
    return server


class TabularHTTPServer:
    """Thin wrapper so callers can start/stop cleanly."""

    def __init__(self, cfg, host="127.0.0.1", port=8000):
        self.cfg = cfg
        self.host = host
        self.port = port
        self._server = None

    def start(self):
        self._server = serve(self.cfg, self.host, self.port)

    def run(self):
        self.start()
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
