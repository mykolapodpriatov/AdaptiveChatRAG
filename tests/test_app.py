"""Tests for the FastAPI admin panel (``app.py``).

Uses FastAPI's ``TestClient`` so the suite does not need Streamlit or a
Telegram token. The autouse in-memory DB fixture in ``conftest.py`` binds
``SessionLocal`` before each request, so ``get_db`` hits the same throwaway
SQLite database as the rest of the suite.
"""
import pytest
from fastapi.testclient import TestClient

from app import app

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": ADMIN_KEY}


def test_root_is_unauthenticated(client):
    response = client.get("/")
    assert response.status_code == 200


def test_stats_missing_key_is_401(client):
    response = client.get("/stats/feedback")
    assert response.status_code == 401


def test_stats_wrong_key_is_401(client):
    response = client.get("/stats/feedback", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_stats_correct_key_is_200(client, auth_headers):
    response = client.get("/stats/feedback", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total_feedback"] == 0


def test_history_missing_key_is_401(client):
    response = client.get("/history/session-1")
    assert response.status_code == 401


def test_history_wrong_key_is_401(client):
    response = client.get("/history/session-1", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_history_correct_key_is_200(client, auth_headers):
    response = client.get("/history/session-1", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []
