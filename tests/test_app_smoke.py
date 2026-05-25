from fastapi.testclient import TestClient

from app.main import app


def test_app_importable():
    assert app is not None


def test_index_ok():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
