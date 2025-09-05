from fastapi.testclient import TestClient

try:
    from src.app import app
except Exception:
    from src.main import app  # type: ignore


client = TestClient(app)


def test_capture_status_endpoint_exists():
    r = client.get("/v1/capture_status")
    assert r.status_code == 200
    body = r.json()
    assert "running" in body

