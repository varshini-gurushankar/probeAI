"""Server tests for the non-LLM paths (no API key / quota needed).

The interview/synthesis/eval endpoints require live Gemini calls and are exercised
manually; here we verify routing, static serving, study metadata, and the
no-session guards deterministically.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from probeai.server import app  # noqa: E402

client = TestClient(app)


def test_study_endpoint_returns_discussion_guide():
    r = client.get("/api/study")
    assert r.status_code == 200
    data = r.json()
    assert data["title"]
    assert len(data["objectives"]) >= 1
    assert data["turn_budget"] >= 1
    # Coverage starts fully uncovered.
    assert all(o["status"] == "uncovered" for o in data["coverage"])


def test_index_and_static_assets_served():
    assert "<title>" in client.get("/").text
    assert client.get("/styles.css").status_code == 200
    assert client.get("/app.js").status_code == 200


def test_turn_without_session_is_409():
    r = client.post("/api/turn", json={"answer": "hello"})
    assert r.status_code == 409


def test_reset_is_ok():
    assert client.post("/api/reset").json() == {"ok": True}
