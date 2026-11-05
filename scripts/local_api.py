#!/usr/bin/env python3
"""Local stand-in for "API Gateway HTTP API -> Lambda".

THIS IS AN EMULATOR WRITTEN FOR THIS PROJECT, not AWS SAM local or LocalStack
(both need a Docker daemon). It builds an API Gateway payload-format-2.0 event
for `POST /predict`, calls the real `lambda_fn/handler.py`, and returns the
handler's status/body unchanged. Unknown routes get API Gateway's own 404 body.

    python scripts/local_api.py --port 3000
    curl -s -X POST localhost:3000/predict -d @events/benign_body.json
"""

import argparse
import json
import logging
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.steps import load_handler  # noqa: E402

# On AWS the Lambda runtime ships log lines to CloudWatch; locally we print them.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

LAMBDA = load_handler(REPO_ROOT / "lambda_fn" / "model.json")  # one "warm container"


class Gateway(BaseHTTPRequestHandler):
    server_version = "LocalHttpApi/1.0"

    def _send(self, status, body, headers=None):
        data = body.encode()
        self.send_response(status)
        for k, v in (headers or {"Content-Type": "application/json"}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, json.dumps({"message": "Not Found"}))
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else None
        event = {
            "version": "2.0",
            "routeKey": "POST /predict",
            "rawPath": "/predict",
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "requestContext": {"http": {"method": "POST", "path": "/predict"}},
            "body": raw,
            "isBase64Encoded": False,
        }
        resp = LAMBDA.lambda_handler(event, SimpleNamespace(aws_request_id=str(uuid.uuid4())))
        self._send(resp["statusCode"], resp["body"], resp.get("headers"))

    def do_GET(self):  # only POST /predict is defined, as in template.yaml
        self._send(404, json.dumps({"message": "Not Found"}))

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] " + fmt % args + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3000)
    args = ap.parse_args()
    print(f"Local API listening on http://127.0.0.1:{args.port}/predict", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Gateway).serve_forever()
