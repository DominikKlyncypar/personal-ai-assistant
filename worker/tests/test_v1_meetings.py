from fastapi.testclient import TestClient

try:
    from src.app import app  # prefer the app factory module
except Exception:  # fallback if running in legacy layout
    from src.main import app  # type: ignore


client = TestClient(app)


def test_meeting_export_endpoints_exist():
    # Create a meeting
    r = client.post("/v1/meeting/new", params={"title": "Test"})
    assert r.status_code == 200
    mid = r.json()["meeting_id"]

    # JSON export
    rj = client.get(f"/v1/meeting/{mid}/export.json")
    assert rj.status_code == 200
    body = rj.json()
    assert "meeting" in body and "utterances" in body

    # Markdown export
    rm = client.get(f"/v1/meeting/{mid}/export.md")
    assert rm.status_code == 200
    assert rm.text.startswith("# Meeting")

