"""Start the HTTP server and curl it (demo).

Run:
    python -m tabularforge.examples.serve_demo
"""
import json
import sys
import threading
import time
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularforge.config import load_config
from tabularforge.serving import serve


def main():
    cfg = load_config()
    server = serve(cfg, host="127.0.0.1", port=8765)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    def post(path, payload):
        req = request.Request(f"http://127.0.0.1:8765{path}",
                              data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    print("health:", post("/health", {}) if False else request.urlopen("http://127.0.0.1:8765/health").read().decode())

    resp = post("/v1/impute", {"records": [
        {"a": 1.0, "b": None},
        {"a": 2.0, "b": None},
        {"a": None, "b": "x"},
    ]})
    print("impute:", resp)

    server.shutdown()
    return 0


if __name__ == "__main__":
    main()
