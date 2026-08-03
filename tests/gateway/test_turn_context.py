"""Unit tests for the TurnContext/TurnRunner seam extracted from
``GatewayRunner._run_agent_inner`` (gateway/turn_context.py + gateway/run.py).

The extraction contract: the closure bodies moved onto ``TurnRunner`` methods
byte-identically (modulo local -> ctx.field rewrites), with every closed-over
local carried as a ``TurnContext`` field. These tests pin the seam's wiring —
shared mutable containers, no-queue early returns — not the progress behavior
itself (that's covered by test_run_progress_topics.py et al.).
"""

import asyncio
import queue as queue_mod
from types import SimpleNamespace

import pytest

from agent.intent_capability import InputProvenance
from gateway.config import Platform
from gateway.platforms.api_server import _classify_api_input_provenance
from gateway.turn_context import TurnContext, classify_input_provenance


def _make_runner(ctx):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

    return TurnRunner(_StubGatewayRunner(), ctx)


class TestTurnContext:
    def test_defaults_are_independent_containers(self):
        a, b = TurnContext(), TurnContext()
        a.last_progress_msg[0] = "x"
        a.repeat_count[0] = 3
        a._cleanup_msg_ids.append("1")
        assert b.last_progress_msg == [None]
        assert b.repeat_count == [0]
        assert b._cleanup_msg_ids == []
        assert b.input_provenance is InputProvenance.UNCLASSIFIED

    def test_shared_containers_visible_to_outer_scope(self):
        # The outer body and the runner share the SAME list objects, so
        # mutation through the ctx is visible to locals captured elsewhere.
        last_progress_msg = [None]
        ctx = TurnContext(last_progress_msg=last_progress_msg)
        ctx.last_progress_msg[0] = "🔍 web_search"
        assert last_progress_msg[0] == "🔍 web_search"


class TestInputProvenance:
    @staticmethod
    def _event(platform=Platform.TELEGRAM, **overrides):
        source = SimpleNamespace(
            platform=platform,
            is_bot=False,
            delivered_via_upstream_relay=False,
        )
        values = {"source": source, "internal": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_direct_user_authority_classifies_authenticated_gateway_input(self):
        assert (
            classify_input_provenance(self._event())
            is InputProvenance.DIRECT_USER_GATEWAY
        )
        assert (
            classify_input_provenance(self._event(Platform.API_SERVER))
            is InputProvenance.DIRECT_USER_API
        )
        assert (
            classify_input_provenance(self._event(Platform.HOMEASSISTANT))
            is InputProvenance.DIRECT_USER_GATEWAY
        )

    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            (_event.__func__(internal=True), InputProvenance.INTERNAL),
            (_event.__func__(Platform.WEBHOOK), InputProvenance.WEBHOOK),
            (_event.__func__(Platform.MSGRAPH_WEBHOOK), InputProvenance.WEBHOOK),
            (_event.__func__(_hermes_startup_restore_replay=True), InputProvenance.REPLAY_ONLY),
            (_event.__func__(replay_only=True), InputProvenance.REPLAY_ONLY),
            (
                _event.__func__(
                    text="[Continuing toward your standing goal]\nGoal: ship"
                ),
                InputProvenance.SYNTHETIC,
            ),
            (_event.__func__(), InputProvenance.DIRECT_USER_GATEWAY),
        ],
    )
    def test_input_provenance_fails_closed_for_non_user_events(self, event, expected):
        if expected is InputProvenance.DIRECT_USER_GATEWAY:
            event.source.is_bot = True
            expected = InputProvenance.SYNTHETIC
        assert classify_input_provenance(event) is expected

    def test_input_provenance_denies_upstream_relay_and_missing_source(self):
        relay_event = self._event()
        relay_event.source.delivered_via_upstream_relay = True
        assert classify_input_provenance(relay_event) is InputProvenance.RELAY
        assert (
            classify_input_provenance(SimpleNamespace(internal=False, source=None))
            is InputProvenance.UNCLASSIFIED
        )
        assert (
            classify_input_provenance(self._event(SimpleNamespace(value="unknown")))
            is InputProvenance.UNCLASSIFIED
        )

    def test_api_input_provenance_header_can_only_downgrade(self):
        request = SimpleNamespace(headers={})
        assert (
            _classify_api_input_provenance(request)
            is InputProvenance.DIRECT_USER_API
        )

        request.headers["X-Hermes-Internal-Provenance"] = "background"
        assert (
            _classify_api_input_provenance(request)
            is InputProvenance.BACKGROUND
        )

        request.headers["X-Hermes-Internal-Provenance"] = "direct_user_cli"
        assert (
            _classify_api_input_provenance(request)
            is InputProvenance.UNCLASSIFIED
        )

    @pytest.mark.asyncio
    async def test_api_wake_self_post_marks_background_provenance(self, monkeypatch):
        import aiohttp
        from gateway.wake import _self_post_chat_completion

        captured = {}

        class _Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def read(self):
                return b""

        class _Session:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def post(self, _url, *, json, headers):
                captured["body"] = json
                captured["headers"] = headers
                return _Response()

        monkeypatch.setattr(aiohttp, "ClientSession", _Session)
        monkeypatch.setattr(aiohttp, "ClientTimeout", lambda **_kwargs: object())
        adapter = SimpleNamespace(
            _host="127.0.0.1",
            _port=8642,
            _api_key="test-key",
            _model_name="hermes-agent",
        )

        await _self_post_chat_completion(
            adapter,
            text="background completion",
            session_id="session-1",
        )

        assert captured["headers"]["X-Hermes-Internal-Provenance"] == "background"
        assert captured["headers"]["X-Hermes-Session-Id"] == "session-1"
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "background completion"}
        ]


class TestTurnRunner:
    def test_methods_exist_and_bind(self):
        from gateway.run import TurnRunner

        ctx = TurnContext()
        runner = _make_runner(ctx)
        assert callable(runner.progress_callback)
        assert asyncio.iscoroutinefunction(TurnRunner.send_progress_messages)
        assert runner._ctx is ctx

    def test_send_progress_messages_no_queue_returns(self):
        ctx = TurnContext(progress_queue=None)
        runner = _make_runner(ctx)
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_send_progress_messages_no_adapter_returns(self):
        ctx = TurnContext(progress_queue=queue_mod.Queue())
        runner = _make_runner(ctx)  # stub adapter resolver returns None
        assert asyncio.run(runner.send_progress_messages()) is None
