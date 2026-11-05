import base64
import json
import math

import pytest


def call(handler, payload, proxy=True, **event_extra):
    event = {"body": json.dumps(payload), **event_extra} if proxy else payload
    resp = handler.lambda_handler(event, None)
    return resp["statusCode"], json.loads(resp["body"])


def test_single_prediction_matches_sigmoid(tiny_handler):
    status, body = call(tiny_handler, {"features": [2.0, 0.5]})
    assert status == 200
    p = 1 / (1 + math.exp(-1.5))
    assert body["prediction"]["label"] == "benign"
    assert body["prediction"]["probability_benign"] == pytest.approx(p, abs=1e-6)
    assert body["model_version"] == "test-1"


def test_negative_class(tiny_handler):
    _, body = call(tiny_handler, {"features": [0.0, 3.0]})
    assert body["prediction"]["label"] == "malignant"
    assert body["prediction"]["label_id"] == 0


def test_direct_invocation_without_proxy_envelope(tiny_handler):
    status, body = call(tiny_handler, {"features": [1, 0]}, proxy=False)
    assert status == 200 and body["prediction"]["label"] == "benign"


def test_batch_and_named_features(tiny_handler):
    status, body = call(tiny_handler, {"instances": [[1, 0], {"a": 0, "b": 2}]})
    assert status == 200
    assert [p["label"] for p in body["predictions"]] == ["benign", "malignant"]


def test_base64_body(tiny_handler):
    raw = base64.b64encode(json.dumps({"features": [1, 0]}).encode()).decode()
    resp = tiny_handler.lambda_handler({"body": raw, "isBase64Encoded": True}, None)
    assert resp["statusCode"] == 200


def test_cold_start_flag_only_on_first_call(tiny_handler):
    first = call(tiny_handler, {"features": [1, 0]})[1]["cold_start"]
    second = call(tiny_handler, {"features": [1, 0]})[1]["cold_start"]
    assert first is True and second is False


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [1.0]},                 # wrong length
        {"features": [1.0, "x"]},            # non-numeric
        {"features": [1.0, True]},           # bool is not a number here
        {"features": {"a": 1.0}},            # missing named feature
        {"instances": []},                   # empty batch
        {"instances": [[1, 0]] * 101},       # over batch limit
        {"nothing": 1},                      # neither key
        [1, 2],                              # not an object
    ],
)
def test_bad_requests_return_400(tiny_handler, payload):
    status, body = call(tiny_handler, payload)
    assert status == 400 and "error" in body


def test_non_finite_values_rejected(tiny_handler):
    resp = tiny_handler.lambda_handler({"body": '{"features": [NaN, 1]}'}, None)
    assert resp["statusCode"] == 400


@pytest.mark.parametrize("body", [None, "", "{not json"])
def test_missing_or_invalid_json_body(tiny_handler, body):
    assert tiny_handler.lambda_handler({"body": body}, None)["statusCode"] == 400


def test_extreme_inputs_do_not_overflow(tiny_handler):
    _, hi = call(tiny_handler, {"features": [1e6, 0]})
    _, lo = call(tiny_handler, {"features": [-1e6, 0]})
    assert hi["prediction"]["probability_benign"] == 1.0
    assert lo["prediction"]["probability_benign"] == 0.0


def test_internal_errors_do_not_leak_details(tiny_handler, monkeypatch):
    def boom(_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(tiny_handler, "predict_one", boom)
    resp = tiny_handler.lambda_handler({"body": json.dumps({"features": [1, 0]})}, None)
    assert resp["statusCode"] == 500
    assert "secret" not in resp["body"]
