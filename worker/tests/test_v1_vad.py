from fastapi.testclient import TestClient

try:
    from src.app import app
except Exception:
    from src.main import app  # type: ignore


client = TestClient(app)


def test_vad_check_exists_and_returns_fields():
    r = client.get("/v1/vad_check")
    assert r.status_code == 200
    body = r.json()
    # With no audio, it should still be ok and include keys
    assert body.get("ok") is True
    for key in ["speech", "rms", "db", "running", "speech_frames", "silence_frames", "events"]:
        assert key in body

