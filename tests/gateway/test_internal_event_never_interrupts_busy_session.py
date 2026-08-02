"""Regression test: internal synthetic events must never interrupt a busy session.

Reported by @Heeervas (June 2026): an ``async_delegation`` completion from a
``delegate_task(background=true)`` subagent re-enters the originating gateway
session as an internal ``MessageEvent``. When that session was busy running a
turn, the completion was treated exactly like a user TEXT message and hit the
default ``busy_input_mode='interrupt'`` path — calling
``running_agent.interrupt()`` and aborting the active turn, plus sending a
"⚡ Interrupting current task" ack. The same shape affects background-process
completions (terminal ``notify_on_complete``), which also re-enter as internal
events.

The fix: ``_handle_active_session_busy_message`` queues ``internal=True`` events
directly (no interrupt or ack) and they cascade as new turns after the current
one finishes. This preserves strict message-role alternation and the design
invariant that a completion surfaces as a NEW turn only when idle, never
spliced into a running turn.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402


def _make_internal_event(text: str = "[async delegation completed]") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
        internal=True,
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 128
    runner._session_source_cache = {}
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []  # no active subagents at completion time
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


@pytest.mark.asyncio
async def test_internal_event_does_not_interrupt_busy_session() -> None:
    """The async-delegation completion must not abort the active turn."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"  # the default that caused the bug
    adapter = _make_adapter()
    event = _make_internal_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    runner._is_user_authorized = MagicMock(
        side_effect=AssertionError("internal events must bypass user authorization")
    )

    handled = await runner._handle_active_session_busy_message(event, sk)

    # The runner owns internal-event queueing so the base adapter cannot merge
    # completion evidence into a user turn or reinterpret it as control input.
    assert handled is True
    assert adapter._pending_messages[sk] is event
    # The active turn must survive.
    parent.interrupt.assert_not_called()
    # No "⚡ Interrupting current task" (or any) ack for a synthetic event.
    adapter._send_with_retry.assert_not_called()


def test_user_input_is_queued_ahead_of_internal_completion() -> None:
    """User work wins the next slot without dropping completion evidence."""
    runner = _make_runner()
    adapter = _make_adapter()
    internal = _make_internal_event("[completion]")
    user = _make_internal_event("new user instruction")
    user.internal = False
    session_key = build_session_key(internal.source)
    runner.adapters[internal.source.platform] = adapter

    runner._queue_or_replace_pending_event(session_key, internal)
    runner._queue_or_replace_pending_event(session_key, user)

    assert adapter._pending_messages[session_key] is user
    assert runner._session_state(session_key).conversation.queued_events == [internal]

    first = adapter._pending_messages.pop(session_key)
    second = runner._promote_queued_event(session_key, adapter, None)
    assert first is user
    assert second is internal


class _LifecycleAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="test", typing_indicator=False),
            Platform.TELEGRAM,
        )
        self.sent: list[str] = []
        self.documents: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def send(self, chat_id, content, **kwargs):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_document(self, chat_id, file_path, **kwargs):
        self.documents.append(file_path)
        return SendResult(success=True, message_id=f"doc-{len(self.documents)}")


def _lifecycle_event(text: str, *, internal: bool) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="user1",
        ),
        message_id=f"msg-{text[:8]}",
        internal=internal,
    )


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out waiting for asynchronous gateway work")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_completion_stays_behind_already_debounced_user_input() -> None:
    """A completion cannot jump ahead of a queue-mode user debounce burst."""
    runner = _make_runner()
    adapter = _LifecycleAdapter()
    user = _lifecycle_event("new user instruction", internal=False)
    completion = _lifecycle_event("[completion]", internal=True)
    session_key = build_session_key(user.source)
    runner.adapters[Platform.TELEGRAM] = adapter

    await adapter._queue_text_debounce(session_key, user)
    assert session_key not in adapter._pending_messages

    handled = await runner._handle_active_session_busy_message(
        completion, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is user
    assert runner._session_state(session_key).conversation.queued_events == [
        completion
    ]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_idle_completion_enters_exactly_one_follow_on_turn() -> None:
    runner = _make_runner()
    adapter = _LifecycleAdapter()
    source = _lifecycle_event("seed", internal=False).source
    session_key = build_session_key(source)
    seen: list[MessageEvent] = []

    async def handler(event):
        seen.append(event)
        return "follow-on response"

    adapter.set_message_handler(handler)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(origin=source)},
    )
    completion = {
        "type": "completion",
        "session_id": "proc-idle",
        "session_key": session_key,
        "started_at": 1.0,
        "command": "echo done",
        "exit_code": 0,
        "output": "done",
    }
    text = "[IMPORTANT: Background process proc-idle completed normally.]"

    assert await runner._deliver_completion_notification(text, completion) is True
    assert await runner._deliver_completion_notification(text, completion) is None
    await _wait_until(lambda: len(seen) == 1 and len(adapter.sent) == 1)

    assert seen[0].internal is True
    assert seen[0].text == text
    assert adapter.sent == ["follow-on response"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_busy_completion_waits_for_parent_response_then_runs_once() -> None:
    runner = _make_runner()
    adapter = _LifecycleAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    parent_started = asyncio.Event()
    release_parent = asyncio.Event()
    seen: list[MessageEvent] = []

    async def handler(event):
        seen.append(event)
        if not event.internal:
            parent_started.set()
            await release_parent.wait()
            return "parent response"
        return "completion follow-on"

    adapter.set_message_handler(handler)
    parent = _lifecycle_event("parent request", internal=False)
    completion = _lifecycle_event("[completion evidence]", internal=True)

    await adapter.handle_message(parent)
    await asyncio.wait_for(parent_started.wait(), timeout=2)
    await adapter.handle_message(completion)

    assert seen == [parent]
    assert not adapter.sent

    release_parent.set()
    await _wait_until(lambda: len(seen) == 2 and len(adapter.sent) == 2)

    assert seen == [parent, completion]
    assert adapter.sent == ["parent response", "completion follow-on"]
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_internal_text_is_opaque_to_base_command_routing() -> None:
    adapter = _LifecycleAdapter()
    handler = AsyncMock(return_value="must not run inline")
    adapter.set_message_handler(handler)
    event = _lifecycle_event("/status", internal=True)
    session_key = build_session_key(event.source)
    guard = asyncio.Event()
    guard.set()
    adapter._active_sessions[session_key] = guard

    await adapter.handle_message(event)

    handler.assert_not_awaited()
    assert adapter._pending_messages[session_key] is event
    assert adapter.has_pending_interrupt(session_key) is False


@pytest.mark.asyncio
async def test_internal_completion_paths_remain_text_not_attachments(tmp_path) -> None:
    paths = []
    for name in ("result.json", "completion.json", "final-report.md"):
        path = tmp_path / name
        path.write_text("artifact", encoding="utf-8")
        paths.append(str(path))

    completion_text = "Completion output:\n" + "\n".join(paths)
    adapter = _LifecycleAdapter()
    adapter.set_message_handler(AsyncMock(return_value=completion_text))
    internal = _lifecycle_event(completion_text, internal=True)

    await adapter.handle_message(internal)
    await _wait_until(lambda: len(adapter.sent) == 1)

    assert adapter.documents == []
    assert all(path in adapter.sent[0] for path in paths)
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_user_turn_can_still_send_bare_local_file(tmp_path) -> None:
    report = tmp_path / "requested-report.md"
    report.write_text("report", encoding="utf-8")
    adapter = _LifecycleAdapter()
    adapter.set_message_handler(AsyncMock(return_value=f"Here it is: {report}"))

    await adapter.handle_message(_lifecycle_event("send the report", internal=False))
    await _wait_until(lambda: len(adapter.documents) == 1)

    assert adapter.documents == [str(report)]
    await adapter.cancel_background_tasks()

