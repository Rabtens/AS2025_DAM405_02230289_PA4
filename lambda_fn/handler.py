"""AWS Lambda handler: tumour classification with a logistic-regression model.

Design notes
------------
* Standard library only. The model is exported by the training pipeline as a
  small JSON document (weights already folded with the scaler), so the
  deployment package needs no numpy/scikit-learn and cold starts stay short.
* The model is loaded once at import time (the INIT phase), so warm
  invocations only pay for the dot product.
* Accepts both an API Gateway / Function-URL proxy event (JSON in ``body``)
  and a direct invocation payload (the JSON object itself).

Request body (either form)::

    {"features": [30 floats]}
    {"instances": [[30 floats], [30 floats], ...]}
    {"features": {"mean radius": 14.1, ...}}      # named features

Response: API Gateway proxy format, JSON body with label and probability.
"""

import base64
import json
import logging
import math
import os
import time

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_INIT_START = time.perf_counter()
_MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
)

with open(_MODEL_PATH, "r", encoding="utf-8") as _fh:
    MODEL = json.load(_fh)

FEATURES = MODEL["feature_names"]
WEIGHTS = MODEL["weights"]
BIAS = MODEL["bias"]
LABELS = MODEL["class_labels"]  # {"0": "malignant", "1": "benign"}
MODEL_VERSION = MODEL.get("model_version", "unknown")
INIT_MS = (time.perf_counter() - _INIT_START) * 1000.0
_COLD_START = True


class BadRequest(ValueError):
    """Raised for client errors; mapped to HTTP 400."""


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _to_vector(item):
    """Convert one instance (list or dict) into a validated list of floats."""
    if isinstance(item, dict):
        missing = [name for name in FEATURES if name not in item]
        if missing:
            raise BadRequest(f"missing features: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        item = [item[name] for name in FEATURES]
    if not isinstance(item, (list, tuple)):
        raise BadRequest("each instance must be a list of numbers or an object of named features")
    if len(item) != len(FEATURES):
        raise BadRequest(f"expected {len(FEATURES)} features, got {len(item)}")
    vec = []
    for value in item:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadRequest("features must be numeric")
        if not math.isfinite(value):
            raise BadRequest("features must be finite numbers")
        vec.append(float(value))
    return vec


def predict_one(vec):
    """Return (label_id, probability_of_class_1) for a validated vector."""
    z = BIAS + sum(w * x for w, x in zip(WEIGHTS, vec))
    p1 = _sigmoid(z)
    return (1 if p1 >= 0.5 else 0), p1


def _parse_event(event):
    """Extract the JSON payload from a proxy event or a direct invocation."""
    if not isinstance(event, dict):
        raise BadRequest("event must be a JSON object")
    if "body" in event:  # API Gateway / Function URL proxy event
        body = event.get("body")
        if body is None or body == "":
            raise BadRequest("empty request body")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        try:
            return json.loads(body) if isinstance(body, str) else body
        except json.JSONDecodeError as exc:
            raise BadRequest(f"body is not valid JSON: {exc.msg}") from exc
    return event  # direct invocation


def _response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    global _COLD_START
    started = time.perf_counter()
    cold, _COLD_START = _COLD_START, False
    request_id = getattr(context, "aws_request_id", "local")
    try:
        payload = _parse_event(event)
        if not isinstance(payload, dict):
            raise BadRequest("payload must be a JSON object")
        if "instances" in payload:
            raw = payload["instances"]
            if not isinstance(raw, list) or not raw:
                raise BadRequest("'instances' must be a non-empty list")
            if len(raw) > 100:
                raise BadRequest("at most 100 instances per request")
            batch = True
        elif "features" in payload:
            raw, batch = [payload["features"]], False
        else:
            raise BadRequest("provide 'features' or 'instances'")

        predictions = []
        for item in raw:
            label_id, p1 = predict_one(_to_vector(item))
            predictions.append(
                {
                    "label": LABELS[str(label_id)],
                    "label_id": label_id,
                    "probability_benign": round(p1, 6),
                    "confidence": round(max(p1, 1.0 - p1), 6),
                }
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        body = {
            "predictions": predictions if batch else None,
            "prediction": None if batch else predictions[0],
            "model_version": MODEL_VERSION,
            "cold_start": cold,
            "latency_ms": round(latency_ms, 3),
        }
        body = {k: v for k, v in body.items() if v is not None}
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": 200,
                    "n": len(predictions),
                    "cold_start": cold,
                    "latency_ms": round(latency_ms, 3),
                    "init_ms": round(INIT_MS, 3) if cold else None,
                }
            )
        )
        return _response(200, body)
    except BadRequest as exc:
        logger.warning(json.dumps({"request_id": request_id, "status": 400, "error": str(exc)}))
        return _response(400, {"error": str(exc)})
    except Exception:  # never leak internals to the caller
        logger.exception("unhandled error (request_id=%s)", request_id)
        return _response(500, {"error": "internal error"})
