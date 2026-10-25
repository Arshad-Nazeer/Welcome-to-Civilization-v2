"""
backend/tests/test_comments_delete.py
-------------------------------------
Unit tests for deleting comments from history and database.
"""

import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.db import save_comment, get_comment_by_id


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.anyio
async def test_delete_comment_lifecycle(client):
    # 1. Create a dummy comment in the store
    doc = {
        "id": "test_del_12345",
        "original_text": "This is a comment to be deleted.",
        "censored_text": "This is a comment to be deleted.",
        "censored_segments": [{"text": "This is a comment to be deleted.", "censored": False}],
        "has_profanity": False,
        "policy_version": "v1.0",
        "headline": {
            "model_id": "3_distilroberta",
            "model_name": "Model 3: DistilRoBERTa",
            "action": "Allow",
            "action_color": "green",
            "severity": {"score": 0.5, "band": "Low"},
            "needs_review": False,
            "reasoning": ["All model outputs below watch zones."],
        },
        "models": {},
    }
    saved_id = await save_comment(doc)
    assert saved_id == "test_del_12345"

    # Verify it exists
    fetched = await get_comment_by_id(saved_id)
    assert fetched is not None
    assert fetched["id"] == saved_id

    # 2. Delete via API
    res = client.delete(f"/api/comments/{saved_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["id"] == saved_id

    # 3. Verify it is gone
    fetched_after = await get_comment_by_id(saved_id)
    assert fetched_after is None

    # 4. Verify API returns 404 for GET
    res_get = client.get(f"/api/comments/{saved_id}")
    assert res_get.status_code == 404

    # 5. Verify deleting non-existent ID returns 404
    res_del_again = client.delete(f"/api/comments/{saved_id}")
    assert res_del_again.status_code == 404
