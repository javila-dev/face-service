def test_health_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "insightface"
    assert body["model"]
    assert body["version"]


def test_health_not_ready(client_not_ready):
    resp = client_not_ready.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_health_does_not_require_auth(client):
    resp = client.get("/health")
    assert resp.status_code != 401
