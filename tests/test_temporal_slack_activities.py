import pytest
from types import SimpleNamespace
import pydantic_temporal_example.temporal.slack_activities as sa
from pydantic_temporal_example.models import SlackConversationsRepliesRequest, SlackMessageID, SlackReply, SlackReaction


class FakeSlack:
    def __init__(self):
        self.calls = 0

    async def conversations_replies(self, *, channel, ts, oldest=None, cursor=None):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return {"messages": [{"text": "first"}], "has_more": True, "response_metadata": {"next_cursor": "NEXT"}}
        return {"messages": [{"text": "second"}], "has_more": False}

    async def chat_postMessage(self, *, channel, thread_ts, text=None, blocks=None):
        return SimpleNamespace(data={"ok": True, "channel": channel, "ts": thread_ts, "text": text, "blocks": blocks})

    async def chat_delete(self, *, channel, ts):
        return SimpleNamespace(data={"ok": True, "channel": channel, "ts": ts})

    async def reactions_add(self, *, name, channel, timestamp):  # noqa: ARG002
        return SimpleNamespace(data={"ok": True, "name": name})

    async def reactions_remove(self, *, name, channel, timestamp):  # noqa: ARG002
        return SimpleNamespace(data={"ok": True, "name": name})


@pytest.mark.asyncio
async def test_slack_thread_pagination(monkeypatch):
    fake = FakeSlack()
    monkeypatch.setattr(sa, "_get_slack_client", lambda: fake, raising=True)
    req = SlackConversationsRepliesRequest(channel="C", ts="1.1", oldest=None)
    msgs = await sa.slack_conversations_replies(req)
    assert [m["text"] for m in msgs] == ["first", "second"]


@pytest.mark.asyncio
async def test_post_delete_and_reactions(monkeypatch):
    fake = FakeSlack()
    monkeypatch.setattr(sa, "_get_slack_client", lambda: fake, raising=True)
    thread = SlackMessageID(channel="C", ts="1.1")
    out = await sa.slack_chat_post_message(SlackReply(thread=thread, content="hi"))
    assert out["ok"] and out["text"] == "hi"
    out = await sa.slack_chat_delete(thread)
    assert out["ok"]
    out = await sa.slack_reactions_add(SlackReaction(message=thread, name="thumbsup"))
    assert out["ok"]
    out = await sa.slack_reactions_remove(SlackReaction(message=thread, name="thumbsup"))
    assert out["ok"]
