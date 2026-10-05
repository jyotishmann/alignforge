import time

import httpx
import pytest

from alignforge.eval import judge_client


def _resp(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, object] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status, headers=headers, json=json, request=httpx.Request("POST", "http://test")
    )


def test_openai_judge_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIGNFORGE_JUDGE_API_KEY", "test-key")
    replies = iter(
        [
            _resp(429, headers={"retry-after": "0"}),
            _resp(200, json={"choices": [{"message": {"content": "A"}}]}),
        ]
    )
    urls: list[str] = []

    def fake_post(url: str, **_: object) -> httpx.Response:
        urls.append(url)
        return next(replies)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    judge = judge_client.make_openai_judge(base_url="https://api.groq.com/openai/v1/")
    assert judge("sys", "user") == "A"
    assert urls == ["https://api.groq.com/openai/v1/chat/completions"] * 2


def test_openai_judge_fails_fast_on_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIGNFORGE_JUDGE_API_KEY", "bad-key")
    calls: list[int] = []

    def fake_post(url: str, **_: object) -> httpx.Response:
        calls.append(1)
        return _resp(401)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        judge_client.make_openai_judge()("sys", "user")
    assert len(calls) == 1  # a bad key is never retried
