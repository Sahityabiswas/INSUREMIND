import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo_server import APIError, DemoServer, SessionStore, validate_settings


@pytest.fixture
def demo_url():
    server = DemoServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(url, data=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    request_headers = {"Content-Type": "application/json"} if body else {}
    request_headers.update(headers or {})
    with urlopen(Request(url, body, request_headers), timeout=10) as response:
        return json.load(response)


def test_demo_session_replay_and_safe_exit(demo_url):
    session = request(demo_url + "/api/sessions", {"policy": "rule", "generator": "template"})
    messages = f"{demo_url}/api/sessions/{session['id']}/messages"
    first = request(messages, {"message": "I need health insurance for my family.", "request_id": "first"})
    assert first["response"]["state"]["need"] == "FAMILY_HEALTH"
    assert first["response"]["source"] == "template"
    assert request(messages, {"message": first["customer"], "request_id": "first"}) == first
    with pytest.raises(HTTPError) as conflict:
        request(messages, {"message": "Different text", "request_id": "first"})
    assert conflict.value.code == 409
    stop = request(messages, {"message": "No, I am not interested. Please stop.", "request_id": "stop"})
    assert stop["response"]["action"] == "RESPECT_REJECTION"
    assert stop["response"]["closed"]
    snapshot = request(f"{demo_url}/api/sessions/{session['id']}")
    assert len(snapshot["history"]) == 2 and snapshot["closed"]
    with pytest.raises(HTTPError) as closed:
        request(messages, {"message": "Again", "request_id": "next"})
    assert closed.value.code == 409


@pytest.mark.parametrize("settings", [{"age": 17}, {"age": True}, {"age": "35"}, {"age": 101},
                                    {"budget": "free"}, {"policy": "fake"}, {"profile": "fake"}, {"generator": "fake"}])
def test_demo_settings_validation(settings):
    with pytest.raises(APIError) as error:
        validate_settings(settings)
    assert error.value.status == 400


def test_demo_isolated_sessions_and_message_limits():
    store = SessionStore()
    first = store.create({"policy": "rule", "generator": "template"})
    second = store.create({"policy": "rule", "generator": "template"})
    store.reply(first["id"], {"message": "I need family health insurance.", "request_id": "one"})
    assert len(store.get(first["id"]).history) == 1
    assert not store.get(second["id"]).history
    for message in ("", " " * 5, "x" * 2001):
        with pytest.raises(APIError) as error:
            store.reply(second["id"], {"message": message, "request_id": "bad"})
        assert error.value.status == 400
    record = store.get(second["id"])
    with record.lock:
        with pytest.raises(APIError) as busy:
            store.reply(second["id"], {"message": "Hello", "request_id": "busy"})
        assert busy.value.status == 409


def test_demo_product_eligibility_and_local_security(demo_url):
    products = request(demo_url + "/api/products?age=35&budget=low&need=FAMILY_HEALTH")["products"]
    assert [product["product_id"] for product in products if product["eligible"]] == ["HLTH-5L"]
    assert not any(product["eligible"] for product in request(demo_url + "/api/products?age=99&need=RETIREMENT")["products"])
    with pytest.raises(HTTPError) as blocked:
        request(demo_url + "/api/sessions", {}, {"Origin": "https://untrusted.example"})
    assert blocked.value.code == 403
    with pytest.raises(HTTPError) as mime:
        request(demo_url + "/api/sessions", {}, {"Content-Type": "text/plain"})
    assert mime.value.code == 415
    with pytest.raises(HTTPError) as unknown:
        request(demo_url + "/../README.md")
    assert unknown.value.code == 404


def test_demo_assets_and_measured_results(demo_url):
    for path in ("/", "/app.js", "/styles.css", "/vendor/lucide.min.js", "/assets/reward-curve.png"):
        with urlopen(demo_url + path, timeout=5) as response:
            assert response.status == 200 and response.read()
            assert response.headers["X-Content-Type-Options"] == "nosniff"
    results = request(demo_url + "/api/results")["metrics"]
    assert {row["agent"] for row in results} == {"random", "rule", "supervised", "ppo"}
    assert all(isinstance(row["avg_reward"], float) for row in results)


def test_demo_rejects_malformed_json(demo_url):
    for body in (b"{", b"[]", b'"not an object"'):
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(demo_url + "/api/sessions", body, {"Content-Type": "application/json"}), timeout=5)
        assert invalid.value.code == 400
