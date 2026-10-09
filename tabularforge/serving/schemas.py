"""Request/response payload helpers (plain dicts, no extra deps)."""
import json


def json_body(handler):
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


def send_json(handler, obj, status=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
