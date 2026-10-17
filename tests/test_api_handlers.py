import pytest
import pydantic_temporal_example.api as api_mod
from pydantic_temporal_example.models import URLVerificationEvent, AppMentionEvent, MessageChannelsEvent
from starlette.responses import JSONResponse, Response


DUMMY_TOKEN = "t"


@pytest.mark.asyncio
async def test_handle_url_verification_event():
    evt = URLVerificationEvent(type="url_verification", token=DUMMY_TOKEN, challenge="xyz")
    resp: JSONResponse = await api_mod.handle_url_verification_event(evt)
    assert resp.body and b"xyz" in resp.body


@pytest.mark.asyncio
async def test_handle_app_mention_event_starts_workflow(monkeypatch):
    class DummySettings:
        temporal_task_queue = "q"
    monkeypatch.setattr(api_mod, "get_settings", lambda: DummySettings(), raising=True)

    started = {}
    class FakeTemporalClient:
        async def start_workflow(self, *args, **kwargs):
            started["args"] = args
            started["kwargs"] = kwargs
    event = AppMentionEvent(
        type="app_mention", user="U", text="hi", ts="1.1", channel="C", event_ts="1.1", thread_ts=None
    )
    resp: Response = await api_mod.handle_app_mention_event(event, FakeTemporalClient())  # type: ignore[arg-type]
    assert resp.status_code == 204
    assert "id" in started["kwargs"] and "task_queue" in started["kwargs"]


@pytest.mark.asyncio
async def test_handle_message_channels_event_when_no_workflow(monkeypatch):
    # Avoid importing real TemporalError by overriding the symbol used in the module.
    class DummyTemporalError(Exception):
        def __init__(self):
            super().__init__("no workflow")
    monkeypatch.setattr(api_mod, "TemporalError", DummyTemporalError, raising=True)

    class FakeHandle:
        async def describe(self):
            raise DummyTemporalError()
        async def signal(self, *args, **kwargs): ...
    class FakeTemporalClient:
        def get_workflow_handle_for(self, *_args, **_kwargs): return FakeHandle()
    event = MessageChannelsEvent(
        type="message", user="U", text="yo", ts="1.1", channel="C", event_ts="1.1", channel_type="channel", thread_ts=None
    )
    resp: Response = await api_mod.handle_message_channels_event(event, FakeTemporalClient())  # type: ignore[arg-type]
    assert resp.status_code == 204
