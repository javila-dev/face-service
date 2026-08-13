import pytest
from fastapi import HTTPException

from app.deps.auth import require_api_key


async def test_valid_x_api_key_accepted():
    key = await require_api_key(x_api_key="test-key", authorization=None)
    assert key == "test-key"


async def test_valid_bearer_token_accepted():
    key = await require_api_key(x_api_key=None, authorization="Bearer test-key")
    assert key == "test-key"


async def test_unknown_key_rejected():
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(x_api_key="not-a-key", authorization=None)
    assert exc_info.value.status_code == 401


async def test_missing_key_rejected():
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(x_api_key=None, authorization=None)
    assert exc_info.value.status_code == 401
