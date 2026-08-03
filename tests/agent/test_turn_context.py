"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import threading
import types
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_context import TurnContext, build_turn_context
from hermes_state import SessionDB

from agent.intent_capability import (
    ApprovedAddendum,
    AuthorityConstraint,
    AuthorityDenied,
    CapabilityLookup,
    ExactScope,
    InputProvenance,
    OneGoalTaskAuthority,
    ParentObjective,
    TaskBinding,
    activate_turn_authority,
    audit_intent_capability,
    bind_trusted_input,
    current_direct_user_receipt,
    current_intent_capability_ledger,
    digest_text,
    issue_intent_capability,
)


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.requested_provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        # Make the fake compressor honour the ContextEngine contract that the
        # real code now relies on (should_compress_info returns a (bool, reason)
        # tuple). Without it build_turn_context raises AttributeError.
        def _fake_should_compress(tokens=None):
            return False

        def _fake_should_compress_info(tokens=None):
            return (False, None)

        self.context_compressor.should_compress = _fake_should_compress
        self.context_compressor.should_compress_info = _fake_should_compress_info
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._emit_warning = MagicMock()
        self._last_ctx_overflow_warn = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        self._session_messages = []
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

    def _warn_context_overflow_blocked(self, reason, preflight_tokens, threshold_tokens):
        # Mirror the real AIAgent helper so tests can assert the warning fired.
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if self._last_ctx_overflow_warn != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠ Context is over the compression threshold "
                f"(~{preflight_tokens:,} tokens >= {threshold_tokens:,}) "
                f"but compression is currently blocked ({reason})."
            )

    def _clear_context_overflow_warn(self):
        self._last_ctx_overflow_warn = None

    # --- methods the prologue calls ---
    def _ensure_db_session(self):
        self._ensure_db_prompt_at_call = self._cached_system_prompt

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        self._persist_calls += 1


def _make_agent_with_cooldown(db_path, session_id, *, cooldown_until=None):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )

    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="cli")
    if cooldown_until is not None:
        db.record_compression_failure_cooldown(session_id, cooldown_until, "timeout")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    agent.context_compressor = compressor
    agent._session_db = db
    return agent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect (telling aux tools the live main provider/model).
    That writes a module-level global these unit tests don't care about and
    which would otherwise leak into sibling tests (e.g. provider-parity
    resolution) when the per-test process isolation plugin is disabled. Stub
    it out so the prologue tests stay hermetic.
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def _intent_authority(*, goals=("implement G4",), task_hash=None):
    objective = ParentObjective.from_content("objective-1", "Repair Hermes safely")
    addendum = ApprovedAddendum.from_content(
        "addendum-1", objective.objective_id, "G4 is approved"
    )
    scope = ExactScope.from_content(
        "Intent-bound capability issuer only",
        ("No repository-write routing", "No deployment"),
    )
    task_content = '{"task":"g4"}'
    binding = TaskBinding.from_content(
        project_id="hermes-agent-engineering",
        component="core-agent-runtime",
        canonical_repository="/repo/hermes-agent",
        execution_repository="/repo/hermes-g4",
        task_id="g4-task",
        task_content=task_content,
        task_hash=digest_text(task_content) if task_hash is None else task_hash,
        scope=scope,
        constraints=(AuthorityConstraint.from_value("single_writer", "codex"),),
    )
    return OneGoalTaskAuthority(
        parent_objective=objective,
        approved_addendum=addendum,
        goals=tuple(goals),
        task=binding,
    )


def test_returns_turn_context_with_user_message_appended():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert isinstance(ctx, TurnContext)
    assert ctx.user_message == "hello"
    # The user turn was appended and indexed.
    assert ctx.messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.active_system_prompt == "SYSTEM"


def test_turn_start_replaces_stale_parent_history_with_compression_child():
    agent = _FakeAgent()
    stale_history = [{"role": "user", "content": "stale parent"}]
    compacted_history = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "child tail"},
    ]

    def _recover(_agent):
        _agent.session_id = "compression-child"
        return compacted_history

    log_context = MagicMock()
    with patch(
        "agent.turn_context.recover_rotated_compression_session",
        side_effect=_recover,
    ):
        ctx = _build(
            agent,
            conversation_history=stale_history,
            set_session_context=log_context,
        )

    assert agent.session_id == "compression-child"
    assert agent._current_turn_id.startswith("compression-child:")
    log_context.assert_called_once_with("compression-child")
    assert ctx.conversation_history == compacted_history
    assert ctx.messages == compacted_history + [{"role": "user", "content": "hello"}]
    assert all(message.get("content") != "stale parent" for message in ctx.messages)


def test_applies_agent_side_effects():
    agent = _FakeAgent()
    _build(agent)
    # Retry counters reset, guardrails reset, vision re-armed, turn counted.
    assert agent._invalid_tool_retries == 0
    assert agent._tool_guardrails.reset_called is True
    assert agent._vision_supported is True
    assert agent._user_turn_count == 1
    # Crash-resilience persistence fired once.
    assert agent._persist_calls == 1
    # task/turn ids assigned on the agent.
    assert agent._current_task_id
    assert agent._current_turn_id










def test_pending_cli_message_uses_clean_override_for_api_local_note():
    """A noted API message reuses the clean staged dict and its DB marker."""
    agent = _FakeAgent()
    staged = {"role": "user", "content": "clean prompt", "_db_persisted": True}
    agent._pending_cli_user_message = staged

    ctx = _build(
        agent,
        user_message="[MODEL NOTE]\n\nclean prompt",
        persist_user_message="clean prompt",
    )

    assert ctx.messages[-1] is staged
    assert ctx.messages[-1]["content"] == "[MODEL NOTE]\n\nclean prompt"
    assert ctx.messages[-1]["_db_persisted"] is True
    assert agent._pending_cli_user_message is None








def test_ensure_db_session_runs_after_system_prompt_restore():
    """Regression for #45499.

    On a fresh API/gateway agent (``_cached_system_prompt is None``) the DB
    session row must be created AFTER the system prompt is restored/built, so
    the persisted snapshot is written non-NULL. If ``_ensure_db_session()``
    ran first it would insert ``system_prompt=NULL`` and trip the misleading
    "stored system prompt is null; rebuilding" warning plus a first-turn
    prefix cache miss.
    """
    agent = _FakeAgent()
    agent._cached_system_prompt = None  # fresh agent, no cached prompt yet

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"

    _build(agent, restore_or_build_system_prompt=_restore)

    # The prompt was populated before the DB row was created.
    assert agent._ensure_db_prompt_at_call == "REBUILT-SYSTEM"
    assert agent._cached_system_prompt == "REBUILT-SYSTEM"


# ── Between-turns MCP refresh (cache-safe late-binding) ──────────────────────
#
# A slow MCP server that connects after the agent's build-time tool snapshot
# must become callable by the user's NEXT turn — without mutating an in-flight
# turn's cached request prefix. The prologue is exactly that boundary, so the
# refresh hook lives here. These assert the contract (R1/R2/R6 in the spec),
# not timing permutations.


def test_between_turns_refresh_adds_late_tool_when_servers_registered():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()

    new_def = {"type": "function", "function": {"name": "mcp_x_tool", "description": "", "parameters": {}}}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)


def test_intent_capability_missing_provenance_is_denied():
    with activate_turn_authority(
        session_id="session-1",
        turn_id="turn-1",
        platform="cli",
        clean_user_message="ship G4",
    ):
        assert current_direct_user_receipt() is None
        with pytest.raises(AuthorityDenied):
            issue_intent_capability(_intent_authority())


@pytest.mark.parametrize(
    "origin",
    [
        InputProvenance.UNCLASSIFIED,
        InputProvenance.INTERNAL,
        InputProvenance.BACKGROUND,
        InputProvenance.CRON,
        InputProvenance.COMPACTION,
        InputProvenance.TODO,
        InputProvenance.MEMORY,
        InputProvenance.REVIEW,
        InputProvenance.MIGRATION,
        InputProvenance.BOOTSTRAP,
        InputProvenance.SYNTHETIC,
        InputProvenance.RELAY,
        InputProvenance.SUBAGENT,
        InputProvenance.REPLAY_ONLY,
        InputProvenance.WEBHOOK,
        "direct_user_cli",
    ],
)
def test_direct_user_authority_denies_non_user_origins(origin):
    with bind_trusted_input(
        origin=origin,
        session_id="session-1",
        platform="cli",
        source_identity="source-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            assert current_direct_user_receipt() is None
            with pytest.raises(AuthorityDenied):
                issue_intent_capability(_intent_authority())


@pytest.mark.parametrize(
    ("origin", "platform"),
    [
        (InputProvenance.DIRECT_USER_CLI, "api_server"),
        (InputProvenance.DIRECT_USER_GATEWAY, "unknown_plugin"),
        (InputProvenance.DIRECT_USER_API, "cli"),
        (InputProvenance.DIRECT_USER_ACP, "gateway"),
    ],
)
def test_direct_user_authority_denies_source_platform_substitution(origin, platform):
    with bind_trusted_input(
        origin=origin,
        session_id="session-1",
        platform=platform,
        source_identity="source-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform=platform,
            clean_user_message="ship G4",
        ):
            assert current_direct_user_receipt() is None


def test_direct_user_authority_mints_one_exact_capability_and_cleans_up():
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            receipt = current_direct_user_receipt()
            assert receipt is not None
            assert receipt.session_id == "session-1"
            assert receipt.turn_id == "turn-1"
            assert b"ship G4" not in receipt.canonical_bytes

            ledger = current_intent_capability_ledger()
            capability = issue_intent_capability(_intent_authority())
            lookup = CapabilityLookup.from_capability(capability)
            assert ledger.exact_lookup(lookup) == capability
            assert ledger.revalidate(capability, lookup) is True
            mismatched_lookups = (
                replace(lookup, project_id="other-project"),
                replace(lookup, canonical_repository="/repo/other"),
                replace(lookup, execution_repository="/workspace/other"),
                replace(lookup, session_id="other-session"),
                replace(lookup, turn_id="other-turn"),
                replace(lookup, task_id="other-task"),
                replace(lookup, task_hash="0" * 64),
                replace(lookup, capability_digest="0" * 64),
                replace(lookup, status="revoked"),
            )
            assert all(
                ledger.exact_lookup(forged) is None
                for forged in mismatched_lookups
            )
            evidence = audit_intent_capability(capability)
            assert evidence.constraint_names == ("single_writer",)
            assert "Repair Hermes safely" not in repr(evidence)
            assert "ship G4" not in repr(evidence)
            assert "/repo/hermes-g4" not in repr(evidence)
            with pytest.raises(AuthorityDenied):
                issue_intent_capability(_intent_authority())

        assert ledger.exact_lookup(lookup) is None
        assert ledger.revalidate(capability, lookup) is False
        assert current_direct_user_receipt() is None


def test_intent_capability_rejects_zero_or_multiple_goals_and_hash_mismatch():
    valid = _intent_authority()
    invalid = [
        _intent_authority(goals=()),
        _intent_authority(goals=("g1", "g2")),
        _intent_authority(task_hash=""),
        _intent_authority(task_hash="0" * 64),
        replace(valid, schema_version=999),
        replace(
            valid,
            parent_objective=replace(
                valid.parent_objective, content_digest="not-a-digest"
            ),
        ),
        replace(
            valid,
            task=replace(
                valid.task,
                scope=replace(
                    valid.task.scope,
                    exclusions=(),
                    exclusions_digest=digest_text("[]"),
                ),
            ),
        ),
    ]
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        for index, authority in enumerate(invalid):
            with activate_turn_authority(
                session_id="session-1",
                turn_id=f"turn-{index}",
                platform="cli",
                clean_user_message="ship G4",
            ):
                with pytest.raises(AuthorityDenied):
                    issue_intent_capability(authority)


def test_intent_capability_rejects_content_digest_mismatch_and_stale_record():
    authority = _intent_authority()
    forged_objective = replace(
        authority.parent_objective, content_digest="0" * 64
    )
    forged_authority = replace(authority, parent_objective=forged_objective)

    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            with pytest.raises(AuthorityDenied):
                issue_intent_capability(forged_authority)
            capability = issue_intent_capability(authority)
            lookup = CapabilityLookup.from_capability(capability)
            ledger = current_intent_capability_ledger()
            with patch(
                "agent.intent_capability.time.time_ns",
                return_value=capability.expires_at_ns + 1,
            ):
                assert ledger.exact_lookup(lookup) is None
                assert ledger.revalidate(capability, lookup) is False


def test_direct_user_authority_rejects_cross_session_message_and_turn_reuse():
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        for session_id, message in [
            ("session-2", "ship G4"),
            ("session-1", "different message"),
        ]:
            with activate_turn_authority(
                session_id=session_id,
                turn_id="turn-wrong",
                platform="cli",
                clean_user_message=message,
            ):
                assert current_direct_user_receipt() is None

        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            capability = issue_intent_capability(_intent_authority())
            lookup = CapabilityLookup.from_capability(capability)
            ledger = current_intent_capability_ledger()

        # One ingress claim is single-use even if its Context was copied.
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-2",
            platform="cli",
            clean_user_message="ship G4",
        ):
            assert current_direct_user_receipt() is None
        assert ledger.revalidate(capability, lookup) is False


def test_direct_user_authority_copied_context_cannot_outlive_turn():
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            capability = issue_intent_capability(_intent_authority())
            lookup = CapabilityLookup.from_capability(capability)
            ledger = current_intent_capability_ledger()
            copied = copy_context()

    assert copied.run(current_direct_user_receipt) is None
    assert copied.run(ledger.revalidate, capability, lookup) is False


def test_direct_user_authority_concurrent_sessions_do_not_leak():
    def _worker(session_id):
        message = f"message for {session_id}"
        with bind_trusted_input(
            origin=InputProvenance.DIRECT_USER_CLI,
            session_id=session_id,
            platform="cli",
            source_identity=f"terminal-{session_id}",
            clean_user_message=message,
        ):
            with activate_turn_authority(
                session_id=session_id,
                turn_id=f"turn-{session_id}",
                platform="cli",
                clean_user_message=message,
            ):
                capability = issue_intent_capability(_intent_authority())
                receipt = current_direct_user_receipt()
                assert receipt is not None
                return receipt.session_id, capability.receipt.session_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_worker, ("session-a", "session-b")))

    assert results == [
        ("session-a", "session-a"),
        ("session-b", "session-b"),
    ]
    assert current_direct_user_receipt() is None


def test_intent_capability_is_immutable_and_receipt_cannot_be_substituted():
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="session-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="ship G4",
    ):
        with activate_turn_authority(
            session_id="session-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="ship G4",
        ):
            capability = issue_intent_capability(_intent_authority())
            # CPython 3.13's slots+frozen dataclass path raises TypeError for
            # assignment to a read-only property; older runtimes raise the
            # more specific FrozenInstanceError. Both prove no mutation lands.
            with pytest.raises((FrozenInstanceError, TypeError)):
                capability.session_id = "forged"  # type: ignore[misc]
            with pytest.raises(TypeError):
                issue_intent_capability(  # type: ignore[call-arg]
                    _intent_authority(), receipt=capability.receipt
                )


def test_direct_user_authority_does_not_change_prompt_or_message_bytes():
    agent = _FakeAgent()
    history = [{"role": "assistant", "content": "prior"}]
    with bind_trusted_input(
        origin=InputProvenance.DIRECT_USER_CLI,
        session_id="sess-1",
        platform="cli",
        source_identity="terminal-1",
        clean_user_message="hello",
    ):
        with activate_turn_authority(
            session_id="sess-1",
            turn_id="turn-1",
            platform="cli",
            clean_user_message="hello",
        ):
            ctx = _build(agent, conversation_history=history)

    assert ctx.active_system_prompt == "SYSTEM"
    assert ctx.messages == history + [{"role": "user", "content": "hello"}]


def test_direct_user_authority_wraps_real_run_conversation_and_always_revokes():
    from run_agent import AIAgent

    class _WrapperAgent:
        session_id = "session-1"
        platform = "cli"
        model = "test/model"
        _session_db = None
        _parent_session_id = None
        _relay_pending_turn_id = None

        @staticmethod
        def _conversation_root_id():
            return "session-1"

    captured = {}

    def _inner_run(agent, user_message, *_args, **_kwargs):
        receipt = current_direct_user_receipt()
        assert receipt is not None
        captured["receipt"] = receipt
        captured["ledger"] = current_intent_capability_ledger()
        captured["capability"] = issue_intent_capability(_intent_authority())
        captured["lookup"] = CapabilityLookup.from_capability(
            captured["capability"]
        )
        return {"final_response": "ok", "messages": [], "completed": True}

    coordinator = MagicMock()
    coordinator.acquire_conversation.return_value = object()
    coordinator.begin_turn.return_value = object()

    with patch("agent.relay_runtime.SESSION_COORDINATOR", coordinator), patch(
        "agent.relay_runtime.current_profile_key", return_value="default"
    ), patch(
        "agent.conversation_loop.run_conversation", side_effect=_inner_run
    ), patch(
        "agent.aux_accounting.set_accounting_context", return_value=object()
    ), patch(
        "agent.aux_accounting.reset_accounting_context"
    ), patch(
        "agent.portal_tags.set_conversation_context", return_value=object()
    ), patch(
        "agent.portal_tags.reset_conversation_context"
    ), patch(
        "agent.subagent_lifecycle.bind_subagent_parent",
        side_effect=lambda _agent: nullcontext(),
    ), patch(
        "agent.auxiliary_client.scoped_runtime_main",
        side_effect=lambda _runtime: nullcontext(),
    ), patch(
        "hermes_cli.observability.relay_shared_metrics.start_task_run"
    ), patch(
        "hermes_cli.observability.relay_shared_metrics.finish_task_run"
    ):
        with bind_trusted_input(
            origin=InputProvenance.DIRECT_USER_CLI,
            session_id="session-1",
            platform="cli",
            source_identity="terminal-1",
            clean_user_message="ship G4",
        ):
            result = AIAgent.run_conversation(_WrapperAgent(), "ship G4")

    assert result["final_response"] == "ok"
    assert captured["receipt"].session_id == "session-1"
    assert captured["ledger"].revalidate(
        captured["capability"], captured["lookup"]
    ) is False
    assert current_direct_user_receipt() is None
