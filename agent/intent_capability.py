"""Runtime-owned direct-user intent receipts and exact-task capabilities.

This module is deliberately an in-process boundary.  It has no persistence,
transport, configuration, environment, or model-tool surface.  Trusted ingress
code binds a sealed, single-use input claim in a :class:`ContextVar`; the agent
turn wrapper consumes that claim after Hermes has assigned the real session and
turn identifiers.  Everything else fails closed.

The model-visible prompt and message list never contain these records.  G1 can
adapt :func:`current_intent_capability_ledger` to its controller protocol later,
but the ledger exposed here remains read-only: exact lookup and revalidation are
its only public operations.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Iterator, Optional


AUTHORITY_SCHEMA_VERSION = 1
CAPABILITY_TTL_NS = 5 * 60 * 1_000_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AuthorityDenied(RuntimeError):
    """Raised when issuance is attempted without exact live authority."""


class AuthorityStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class InputProvenance(str, Enum):
    """Closed provenance vocabulary for agent inputs.

    Only the four ``DIRECT_USER_*`` members may mint a receipt.  Every
    non-user source named in the runtime-governance contract is explicit so an
    ingress cannot accidentally become authoritative by falling through a
    truthy/default branch.
    """

    DIRECT_USER_CLI = "direct_user_cli"
    DIRECT_USER_GATEWAY = "direct_user_gateway"
    DIRECT_USER_API = "direct_user_api"
    DIRECT_USER_ACP = "direct_user_acp"

    UNCLASSIFIED = "unclassified"
    INTERNAL = "internal"
    BACKGROUND = "background"
    CRON = "cron"
    COMPACTION = "compaction"
    TODO = "todo"
    MEMORY = "memory"
    REVIEW = "review"
    MIGRATION = "migration"
    BOOTSTRAP = "bootstrap"
    SYNTHETIC = "synthetic"
    RELAY = "relay"
    SUBAGENT = "subagent"
    REPLAY_ONLY = "replay_only"
    WEBHOOK = "webhook"


DIRECT_USER_PROVENANCE = frozenset({
    InputProvenance.DIRECT_USER_CLI,
    InputProvenance.DIRECT_USER_GATEWAY,
    InputProvenance.DIRECT_USER_API,
    InputProvenance.DIRECT_USER_ACP,
})

TRUSTED_DIRECT_USER_GATEWAY_PLATFORMS = frozenset({
    "local",
    "telegram",
    "discord",
    "whatsapp",
    "whatsapp_cloud",
    "slack",
    "signal",
    "mattermost",
    "matrix",
    "homeassistant",
    "email",
    "sms",
    "dingtalk",
    "feishu",
    "wecom",
    "wecom_callback",
    "weixin",
    "bluebubbles",
    "qqbot",
    "yuanbao",
})


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(content: str) -> str:
    """Return the exact UTF-8 SHA-256 digest used by content-bound models."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    return _sha256_bytes(content.encode("utf-8"))


def digest_clean_message(content: Any) -> str:
    """Digest a clean inbound message without retaining its raw bytes."""

    return _sha256_bytes(
        _canonical_json_bytes({"schema": AUTHORITY_SCHEMA_VERSION, "message": content})
    )


def _digest_sequence(values: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(values)))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_exact_text(value: Any, *, allow_empty: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if not allow_empty and not value:
        return False
    return value == value.strip()


def _valid_schema(value: Any) -> bool:
    return type(value) is int and value == AUTHORITY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ParentObjective:
    objective_id: str
    content: str
    content_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_content(cls, objective_id: str, content: str) -> "ParentObjective":
        return cls(objective_id, content, digest_text(content))

    def is_valid(self) -> bool:
        try:
            return (
                _valid_schema(self.schema_version)
                and _is_exact_text(self.objective_id)
                and isinstance(self.content, str)
                and bool(self.content)
                and _is_digest(self.content_digest)
                and digest_text(self.content) == self.content_digest
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class ApprovedAddendum:
    addendum_id: str
    parent_objective_id: str
    content: str
    content_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_content(
        cls,
        addendum_id: str,
        parent_objective_id: str,
        content: str,
    ) -> "ApprovedAddendum":
        return cls(addendum_id, parent_objective_id, content, digest_text(content))

    def is_valid(self) -> bool:
        try:
            return (
                _valid_schema(self.schema_version)
                and _is_exact_text(self.addendum_id)
                and _is_exact_text(self.parent_objective_id)
                and isinstance(self.content, str)
                and bool(self.content)
                and _is_digest(self.content_digest)
                and digest_text(self.content) == self.content_digest
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class ExactScope:
    content: str
    content_digest: str
    exclusions: tuple[str, ...]
    exclusions_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_content(
        cls,
        content: str,
        exclusions: Iterable[str],
    ) -> "ExactScope":
        exact_exclusions = tuple(exclusions)
        return cls(
            content=content,
            content_digest=digest_text(content),
            exclusions=exact_exclusions,
            exclusions_digest=_digest_sequence(exact_exclusions),
        )

    def is_valid(self) -> bool:
        try:
            return (
                _valid_schema(self.schema_version)
                and isinstance(self.content, str)
                and bool(self.content)
                and _is_digest(self.content_digest)
                and digest_text(self.content) == self.content_digest
                and isinstance(self.exclusions, tuple)
                and bool(self.exclusions)
                and all(_is_exact_text(item) for item in self.exclusions)
                and len(set(self.exclusions)) == len(self.exclusions)
                and _is_digest(self.exclusions_digest)
                and _digest_sequence(self.exclusions) == self.exclusions_digest
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class AuthorityConstraint:
    name: str
    value: str
    value_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_value(cls, name: str, value: str) -> "AuthorityConstraint":
        return cls(name=name, value=value, value_digest=digest_text(value))

    def is_valid(self) -> bool:
        try:
            return (
                _valid_schema(self.schema_version)
                and _is_exact_text(self.name)
                and isinstance(self.value, str)
                and _is_digest(self.value_digest)
                and digest_text(self.value) == self.value_digest
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class TaskBinding:
    project_id: str
    component: str
    canonical_repository: str
    execution_repository: str
    task_id: str
    task_content: str
    task_hash: str
    scope: ExactScope
    constraints: tuple[AuthorityConstraint, ...]
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_content(
        cls,
        *,
        project_id: str,
        component: str,
        canonical_repository: str,
        execution_repository: str,
        task_id: str,
        task_content: str,
        scope: ExactScope,
        constraints: Iterable[AuthorityConstraint],
        task_hash: Optional[str] = None,
    ) -> "TaskBinding":
        return cls(
            project_id=project_id,
            component=component,
            canonical_repository=canonical_repository,
            execution_repository=execution_repository,
            task_id=task_id,
            task_content=task_content,
            task_hash=(digest_text(task_content) if task_hash is None else task_hash),
            scope=scope,
            constraints=tuple(constraints),
        )

    def is_valid(self) -> bool:
        try:
            constraint_names = tuple(item.name for item in self.constraints)
            return (
                _valid_schema(self.schema_version)
                and _is_exact_text(self.project_id)
                and _is_exact_text(self.component)
                and _is_exact_text(self.canonical_repository)
                and _is_exact_text(self.execution_repository)
                and _is_exact_text(self.task_id)
                and isinstance(self.task_content, str)
                and bool(self.task_content)
                and _is_digest(self.task_hash)
                and digest_text(self.task_content) == self.task_hash
                and isinstance(self.scope, ExactScope)
                and self.scope.is_valid()
                and isinstance(self.constraints, tuple)
                and all(
                    isinstance(item, AuthorityConstraint) and item.is_valid()
                    for item in self.constraints
                )
                and len(set(constraint_names)) == len(constraint_names)
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class OneGoalTaskAuthority:
    parent_objective: ParentObjective
    approved_addendum: ApprovedAddendum
    goals: tuple[str, ...]
    task: TaskBinding
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    def is_valid(self) -> bool:
        try:
            return (
                _valid_schema(self.schema_version)
                and isinstance(self.parent_objective, ParentObjective)
                and self.parent_objective.is_valid()
                and isinstance(self.approved_addendum, ApprovedAddendum)
                and self.approved_addendum.is_valid()
                and self.approved_addendum.parent_objective_id
                == self.parent_objective.objective_id
                and isinstance(self.goals, tuple)
                and len(self.goals) == 1
                and _is_exact_text(self.goals[0])
                and isinstance(self.task, TaskBinding)
                and self.task.is_valid()
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class DirectUserIntentReceipt:
    receipt_id: str
    session_id: str
    turn_id: str
    platform: str
    source: InputProvenance
    source_identity_digest: str
    user_message_digest: str
    issued_at_ns: int
    expires_at_ns: int
    status: AuthorityStatus
    revoked_at_ns: Optional[int]
    canonical_bytes: bytes
    receipt_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class IntentCapability:
    capability_id: str
    receipt: DirectUserIntentReceipt
    parent_objective: ParentObjective
    approved_addendum: ApprovedAddendum
    goal: str
    task: TaskBinding
    issued_at_ns: int
    expires_at_ns: int
    status: AuthorityStatus
    revoked_at_ns: Optional[int]
    canonical_bytes: bytes
    capability_digest: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @property
    def session_id(self) -> str:
        return self.receipt.session_id

    @property
    def turn_id(self) -> str:
        return self.receipt.turn_id


@dataclass(frozen=True, slots=True)
class CapabilityLookup:
    capability_id: str
    capability_digest: str
    receipt_id: str
    receipt_digest: str
    session_id: str
    turn_id: str
    platform: str
    source: str
    source_identity_digest: str
    user_message_digest: str
    parent_objective_id: str
    parent_objective_digest: str
    approved_addendum_id: str
    approved_addendum_digest: str
    goal_digest: str
    project_id: str
    component: str
    canonical_repository: str
    execution_repository: str
    task_id: str
    task_hash: str
    scope_digest: str
    exclusions_digest: str
    constraints: tuple[tuple[str, str], ...]
    issued_at_ns: int
    expires_at_ns: int
    status: str
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_capability(cls, capability: IntentCapability) -> "CapabilityLookup":
        return cls(
            capability_id=capability.capability_id,
            capability_digest=capability.capability_digest,
            receipt_id=capability.receipt.receipt_id,
            receipt_digest=capability.receipt.receipt_digest,
            session_id=capability.receipt.session_id,
            turn_id=capability.receipt.turn_id,
            platform=capability.receipt.platform,
            source=capability.receipt.source.value,
            source_identity_digest=capability.receipt.source_identity_digest,
            user_message_digest=capability.receipt.user_message_digest,
            parent_objective_id=capability.parent_objective.objective_id,
            parent_objective_digest=capability.parent_objective.content_digest,
            approved_addendum_id=capability.approved_addendum.addendum_id,
            approved_addendum_digest=capability.approved_addendum.content_digest,
            goal_digest=digest_text(capability.goal),
            project_id=capability.task.project_id,
            component=capability.task.component,
            canonical_repository=capability.task.canonical_repository,
            execution_repository=capability.task.execution_repository,
            task_id=capability.task.task_id,
            task_hash=capability.task.task_hash,
            scope_digest=capability.task.scope.content_digest,
            exclusions_digest=capability.task.scope.exclusions_digest,
            constraints=tuple(
                (item.name, item.value_digest) for item in capability.task.constraints
            ),
            issued_at_ns=capability.issued_at_ns,
            expires_at_ns=capability.expires_at_ns,
            status=capability.status.value,
        )


@dataclass(frozen=True, slots=True)
class CapabilityAuditEvidence:
    capability_id: str
    capability_digest: str
    receipt_id: str
    receipt_digest: str
    session_id: str
    turn_id: str
    platform: str
    source: str
    source_identity_digest: str
    user_message_digest: str
    parent_objective_id: str
    parent_objective_digest: str
    approved_addendum_id: str
    approved_addendum_digest: str
    goal_digest: str
    project_id: str
    component: str
    task_id: str
    task_hash: str
    scope_digest: str
    exclusions_digest: str
    issued_at_ns: int
    expires_at_ns: int
    revoked_at_ns: Optional[int]
    status: str
    constraint_names: tuple[str, ...]


def _receipt_payload(receipt: DirectUserIntentReceipt) -> dict[str, Any]:
    return {
        "expires_at_ns": receipt.expires_at_ns,
        "issued_at_ns": receipt.issued_at_ns,
        "platform": receipt.platform,
        "receipt_id": receipt.receipt_id,
        "revoked_at_ns": receipt.revoked_at_ns,
        "schema_version": receipt.schema_version,
        "session_id": receipt.session_id,
        "source": receipt.source.value,
        "source_identity_digest": receipt.source_identity_digest,
        "status": receipt.status.value,
        "turn_id": receipt.turn_id,
        "user_message_digest": receipt.user_message_digest,
    }


def _make_receipt(
    *,
    receipt_id: str,
    session_id: str,
    turn_id: str,
    platform: str,
    source: InputProvenance,
    source_identity_digest: str,
    user_message_digest: str,
    issued_at_ns: int,
    expires_at_ns: int,
    status: AuthorityStatus = AuthorityStatus.ACTIVE,
    revoked_at_ns: Optional[int] = None,
) -> DirectUserIntentReceipt:
    receipt = DirectUserIntentReceipt(
        receipt_id=receipt_id,
        session_id=session_id,
        turn_id=turn_id,
        platform=platform,
        source=source,
        source_identity_digest=source_identity_digest,
        user_message_digest=user_message_digest,
        issued_at_ns=issued_at_ns,
        expires_at_ns=expires_at_ns,
        status=status,
        revoked_at_ns=revoked_at_ns,
        canonical_bytes=b"",
        receipt_digest="",
    )
    canonical = _canonical_json_bytes(_receipt_payload(receipt))
    return replace(
        receipt,
        canonical_bytes=canonical,
        receipt_digest=_sha256_bytes(canonical),
    )


def _receipt_is_valid(
    receipt: Any,
    *,
    now_ns: int,
    require_active: bool = True,
) -> bool:
    try:
        if not isinstance(receipt, DirectUserIntentReceipt):
            return False
        if not _valid_schema(receipt.schema_version):
            return False
        if receipt.source not in DIRECT_USER_PROVENANCE:
            return False
        if not all(
            _is_exact_text(value)
            for value in (
                receipt.receipt_id,
                receipt.session_id,
                receipt.turn_id,
                receipt.platform,
            )
        ):
            return False
        if not _is_digest(receipt.source_identity_digest):
            return False
        if not _is_digest(receipt.user_message_digest):
            return False
        if (
            type(receipt.issued_at_ns) is not int
            or type(receipt.expires_at_ns) is not int
        ):
            return False
        if receipt.issued_at_ns <= 0 or receipt.expires_at_ns <= receipt.issued_at_ns:
            return False
        if receipt.status is AuthorityStatus.ACTIVE:
            if receipt.revoked_at_ns is not None:
                return False
            if require_active and now_ns >= receipt.expires_at_ns:
                return False
        elif receipt.status is AuthorityStatus.REVOKED:
            if require_active or type(receipt.revoked_at_ns) is not int:
                return False
            if receipt.revoked_at_ns < receipt.issued_at_ns:
                return False
        else:
            return False
        expected = _canonical_json_bytes(_receipt_payload(receipt))
        return (
            isinstance(receipt.canonical_bytes, bytes)
            and receipt.canonical_bytes == expected
            and _is_digest(receipt.receipt_digest)
            and _sha256_bytes(expected) == receipt.receipt_digest
        )
    except Exception:
        return False


def _capability_payload(capability: IntentCapability) -> dict[str, Any]:
    return {
        "approved_addendum": {
            "content_digest": capability.approved_addendum.content_digest,
            "id": capability.approved_addendum.addendum_id,
        },
        "capability_id": capability.capability_id,
        "constraints": [
            {"name": item.name, "value_digest": item.value_digest}
            for item in capability.task.constraints
        ],
        "expires_at_ns": capability.expires_at_ns,
        "goal": capability.goal,
        "goal_digest": digest_text(capability.goal),
        "issued_at_ns": capability.issued_at_ns,
        "parent_objective": {
            "content_digest": capability.parent_objective.content_digest,
            "id": capability.parent_objective.objective_id,
        },
        "receipt_digest": capability.receipt.receipt_digest,
        "receipt_id": capability.receipt.receipt_id,
        "revoked_at_ns": capability.revoked_at_ns,
        "schema_version": capability.schema_version,
        "status": capability.status.value,
        "task": {
            "canonical_repository": capability.task.canonical_repository,
            "component": capability.task.component,
            "execution_repository": capability.task.execution_repository,
            "exclusions_digest": capability.task.scope.exclusions_digest,
            "id": capability.task.task_id,
            "project_id": capability.task.project_id,
            "scope_digest": capability.task.scope.content_digest,
            "task_hash": capability.task.task_hash,
        },
        "turn": {
            "session_id": capability.receipt.session_id,
            "turn_id": capability.receipt.turn_id,
        },
    }


def _make_capability(
    receipt: DirectUserIntentReceipt,
    authority: OneGoalTaskAuthority,
) -> IntentCapability:
    capability = IntentCapability(
        capability_id=uuid.uuid4().hex,
        receipt=receipt,
        parent_objective=authority.parent_objective,
        approved_addendum=authority.approved_addendum,
        goal=authority.goals[0],
        task=authority.task,
        issued_at_ns=receipt.issued_at_ns,
        expires_at_ns=receipt.expires_at_ns,
        status=AuthorityStatus.ACTIVE,
        revoked_at_ns=None,
        canonical_bytes=b"",
        capability_digest="",
    )
    canonical = _canonical_json_bytes(_capability_payload(capability))
    return replace(
        capability,
        canonical_bytes=canonical,
        capability_digest=_sha256_bytes(canonical),
    )


def _capability_is_valid(
    capability: Any,
    *,
    now_ns: int,
    require_active: bool = True,
) -> bool:
    try:
        if not isinstance(capability, IntentCapability):
            return False
        if not _valid_schema(capability.schema_version):
            return False
        authority = OneGoalTaskAuthority(
            parent_objective=capability.parent_objective,
            approved_addendum=capability.approved_addendum,
            goals=(capability.goal,),
            task=capability.task,
        )
        if not authority.is_valid():
            return False
        if not _receipt_is_valid(
            capability.receipt,
            now_ns=now_ns,
            require_active=require_active,
        ):
            return False
        if not _is_exact_text(capability.capability_id):
            return False
        if capability.issued_at_ns != capability.receipt.issued_at_ns:
            return False
        if capability.expires_at_ns != capability.receipt.expires_at_ns:
            return False
        if capability.status is not capability.receipt.status:
            return False
        if capability.revoked_at_ns != capability.receipt.revoked_at_ns:
            return False
        if capability.status is AuthorityStatus.ACTIVE:
            if capability.revoked_at_ns is not None:
                return False
            if require_active and now_ns >= capability.expires_at_ns:
                return False
        elif capability.status is AuthorityStatus.REVOKED:
            if require_active or type(capability.revoked_at_ns) is not int:
                return False
            if capability.revoked_at_ns < capability.issued_at_ns:
                return False
        else:
            return False
        expected = _canonical_json_bytes(_capability_payload(capability))
        return (
            isinstance(capability.canonical_bytes, bytes)
            and capability.canonical_bytes == expected
            and _is_digest(capability.capability_digest)
            and _sha256_bytes(expected) == capability.capability_digest
        )
    except Exception:
        return False


def _revoke_receipt(
    receipt: DirectUserIntentReceipt,
    revoked_at_ns: int,
) -> DirectUserIntentReceipt:
    return _make_receipt(
        receipt_id=receipt.receipt_id,
        session_id=receipt.session_id,
        turn_id=receipt.turn_id,
        platform=receipt.platform,
        source=receipt.source,
        source_identity_digest=receipt.source_identity_digest,
        user_message_digest=receipt.user_message_digest,
        issued_at_ns=receipt.issued_at_ns,
        expires_at_ns=receipt.expires_at_ns,
        status=AuthorityStatus.REVOKED,
        revoked_at_ns=revoked_at_ns,
    )


def _revoke_capability(
    capability: IntentCapability,
    revoked_at_ns: int,
) -> IntentCapability:
    revoked = replace(
        capability,
        receipt=_revoke_receipt(capability.receipt, revoked_at_ns),
        status=AuthorityStatus.REVOKED,
        revoked_at_ns=revoked_at_ns,
        canonical_bytes=b"",
        capability_digest="",
    )
    canonical = _canonical_json_bytes(_capability_payload(revoked))
    return replace(
        revoked,
        canonical_bytes=canonical,
        capability_digest=_sha256_bytes(canonical),
    )


class _IngressLease:
    __slots__ = ("_active", "_consumed", "_lock")

    def __init__(self) -> None:
        self._active = True
        self._consumed = False
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            if not self._active or self._consumed:
                return False
            self._consumed = True
            return True

    def close(self) -> None:
        with self._lock:
            self._active = False


@dataclass(frozen=True, slots=True)
class _IngressClaim:
    origin: InputProvenance
    session_id: str
    platform: str
    source_identity_digest: str
    user_message_digest: str
    lease: _IngressLease


_CURRENT_INPUT: ContextVar[Optional[_IngressClaim]] = ContextVar(
    "hermes_current_input_provenance", default=None
)


def _origin_matches_platform(origin: InputProvenance, platform: str) -> bool:
    normalized = platform.strip().lower()
    if origin is InputProvenance.DIRECT_USER_CLI:
        return normalized in {"cli", "local"}
    if origin is InputProvenance.DIRECT_USER_API:
        return normalized in {"api", "api_server"}
    if origin is InputProvenance.DIRECT_USER_ACP:
        return normalized == "acp"
    if origin is InputProvenance.DIRECT_USER_GATEWAY:
        return normalized in TRUSTED_DIRECT_USER_GATEWAY_PLATFORMS
    return False


@contextmanager
def bind_trusted_input(
    *,
    origin: InputProvenance,
    session_id: str,
    platform: str,
    source_identity: str,
    clean_user_message: Any,
) -> Iterator[None]:
    """Bind one ingress-classified input for exactly one agent turn.

    This function is for trusted Hermes ingress code only.  It accepts no
    receipt, turn id, timestamp, expiry, status, or capability bytes.  Direct
    provenance is sealed into a single-use lease; malformed or non-user input
    still runs normally but cannot mint authority.
    """

    safe_origin = (
        origin if isinstance(origin, InputProvenance) else InputProvenance.UNCLASSIFIED
    )
    try:
        source_digest = digest_text(source_identity)
        message_digest = digest_clean_message(clean_user_message)
    except Exception:
        safe_origin = InputProvenance.UNCLASSIFIED
        source_digest = ""
        message_digest = ""

    if safe_origin in DIRECT_USER_PROVENANCE and not (
        _is_exact_text(session_id)
        and _is_exact_text(platform)
        and _is_exact_text(source_identity)
        and _origin_matches_platform(safe_origin, platform)
        and _is_digest(source_digest)
        and _is_digest(message_digest)
    ):
        safe_origin = InputProvenance.UNCLASSIFIED

    claim = _IngressClaim(
        origin=safe_origin,
        session_id=session_id if isinstance(session_id, str) else "",
        platform=platform if isinstance(platform, str) else "",
        source_identity_digest=source_digest,
        user_message_digest=message_digest,
        lease=_IngressLease(),
    )
    token = _CURRENT_INPUT.set(claim)
    try:
        yield
    finally:
        claim.lease.close()
        _CURRENT_INPUT.reset(token)


class IntentCapabilityLedger:
    """Turn-local, externally read-only ledger of one exact capability."""

    __slots__ = ("_closed", "_issued", "_lock", "_receipt", "_records")

    def __init__(self, receipt: Optional[DirectUserIntentReceipt]) -> None:
        self._receipt = receipt
        self._records: dict[str, IntentCapability] = {}
        self._closed = False
        self._issued = False
        self._lock = threading.RLock()

    def exact_lookup(self, lookup: CapabilityLookup) -> Optional[IntentCapability]:
        """Return a capability only when every binding is an exact live match."""

        try:
            if not isinstance(lookup, CapabilityLookup):
                return None
            if not _valid_schema(lookup.schema_version):
                return None
            if _CURRENT_LEDGER.get() is not self:
                return None
            now_ns = time.time_ns()
            with self._lock:
                if self._closed:
                    return None
                capability = self._records.get(lookup.capability_id)
                if capability is None:
                    return None
                if CapabilityLookup.from_capability(capability) != lookup:
                    return None
                if not _capability_is_valid(capability, now_ns=now_ns):
                    return None
                if self._receipt != capability.receipt:
                    return None
                return capability
        except Exception:
            return None

    def revalidate(
        self,
        capability: IntentCapability,
        lookup: CapabilityLookup,
    ) -> bool:
        """Revalidate an exact record without trusting caller-provided bytes."""

        try:
            stored = self.exact_lookup(lookup)
            return stored is not None and stored == capability
        except Exception:
            return False

    def _issue(self, authority: OneGoalTaskAuthority) -> IntentCapability:
        with self._lock:
            now_ns = time.time_ns()
            if _CURRENT_LEDGER.get() is not self or self._closed:
                raise AuthorityDenied("no active turn capability ledger")
            if self._receipt is None or not _receipt_is_valid(
                self._receipt, now_ns=now_ns
            ):
                raise AuthorityDenied("no live direct-user intent receipt")
            if (
                not isinstance(authority, OneGoalTaskAuthority)
                or not authority.is_valid()
            ):
                raise AuthorityDenied("task authority is malformed or not one-goal")
            if self._issued or self._records:
                raise AuthorityDenied(
                    "this turn already issued its exact-task capability"
                )
            capability = _make_capability(self._receipt, authority)
            if not _capability_is_valid(capability, now_ns=now_ns):
                raise AuthorityDenied("capability failed canonical validation")
            self._records[capability.capability_id] = capability
            self._issued = True
            return capability

    def _close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                revoked_at_ns = time.time_ns()
                self._records = {
                    key: _revoke_capability(value, revoked_at_ns)
                    for key, value in self._records.items()
                }
                if self._receipt is not None:
                    self._receipt = _revoke_receipt(self._receipt, revoked_at_ns)
            except Exception:
                self._records = {}
                self._receipt = None
            finally:
                # Cleanup must fail closed even if a corrupted private record
                # cannot be rendered into its immutable revoked form.
                self._closed = True

    def _active_receipt(self) -> Optional[DirectUserIntentReceipt]:
        try:
            if _CURRENT_LEDGER.get() is not self:
                return None
            with self._lock:
                if self._closed or self._receipt is None:
                    return None
                if not _receipt_is_valid(self._receipt, now_ns=time.time_ns()):
                    return None
                return self._receipt
        except Exception:
            return None


_CURRENT_LEDGER: ContextVar[Optional[IntentCapabilityLedger]] = ContextVar(
    "hermes_current_intent_capability_ledger", default=None
)
_DENIED_LEDGER = IntentCapabilityLedger(None)
_DENIED_LEDGER._closed = True


@contextmanager
def activate_turn_authority(
    *,
    session_id: str,
    turn_id: str,
    platform: str,
    clean_user_message: Any,
) -> Iterator[IntentCapabilityLedger]:
    """Consume the current ingress claim and activate a per-turn ledger."""

    receipt: Optional[DirectUserIntentReceipt] = None
    claim = _CURRENT_INPUT.get()
    try:
        message_digest = digest_clean_message(clean_user_message)
    except Exception:
        message_digest = ""

    try:
        if (
            claim is not None
            and claim.origin in DIRECT_USER_PROVENANCE
            and _is_exact_text(session_id)
            and _is_exact_text(turn_id)
            and _is_exact_text(platform)
            and claim.session_id == session_id
            and claim.platform == platform
            and claim.user_message_digest == message_digest
            and _origin_matches_platform(claim.origin, platform)
            and claim.lease.consume()
        ):
            issued_at_ns = time.time_ns()
            receipt = _make_receipt(
                receipt_id=uuid.uuid4().hex,
                session_id=session_id,
                turn_id=turn_id,
                platform=platform,
                source=claim.origin,
                source_identity_digest=claim.source_identity_digest,
                user_message_digest=message_digest,
                issued_at_ns=issued_at_ns,
                expires_at_ns=issued_at_ns + CAPABILITY_TTL_NS,
            )
            if not _receipt_is_valid(receipt, now_ns=issued_at_ns):
                receipt = None
    except Exception:
        receipt = None

    ledger = IntentCapabilityLedger(receipt)
    token = _CURRENT_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        try:
            ledger._close()
        finally:
            _CURRENT_LEDGER.reset(token)


def current_intent_capability_ledger() -> IntentCapabilityLedger:
    """Return the active read-only ledger, or a permanently denying ledger."""

    ledger = _CURRENT_LEDGER.get()
    if ledger is None or ledger._closed:
        return _DENIED_LEDGER
    return ledger


def current_direct_user_receipt() -> Optional[DirectUserIntentReceipt]:
    """Return the live receipt for this exact turn, never a stale copy."""

    return current_intent_capability_ledger()._active_receipt()


def issue_intent_capability(authority: OneGoalTaskAuthority) -> IntentCapability:
    """Issue this direct-user turn's sole exact-task capability.

    Receipt/provenance/session/turn/time/status/expiry are intentionally not
    parameters.  They can only come from the active runtime scope.
    """

    return current_intent_capability_ledger()._issue(authority)


def audit_intent_capability(capability: IntentCapability) -> CapabilityAuditEvidence:
    """Build non-secret audit evidence (ids, digests, times, state, names)."""

    if not _capability_is_valid(
        capability,
        now_ns=time.time_ns(),
        require_active=False,
    ):
        raise AuthorityDenied("cannot audit a malformed capability")
    return CapabilityAuditEvidence(
        capability_id=capability.capability_id,
        capability_digest=capability.capability_digest,
        receipt_id=capability.receipt.receipt_id,
        receipt_digest=capability.receipt.receipt_digest,
        session_id=capability.receipt.session_id,
        turn_id=capability.receipt.turn_id,
        platform=capability.receipt.platform,
        source=capability.receipt.source.value,
        source_identity_digest=capability.receipt.source_identity_digest,
        user_message_digest=capability.receipt.user_message_digest,
        parent_objective_id=capability.parent_objective.objective_id,
        parent_objective_digest=capability.parent_objective.content_digest,
        approved_addendum_id=capability.approved_addendum.addendum_id,
        approved_addendum_digest=capability.approved_addendum.content_digest,
        goal_digest=digest_text(capability.goal),
        project_id=capability.task.project_id,
        component=capability.task.component,
        task_id=capability.task.task_id,
        task_hash=capability.task.task_hash,
        scope_digest=capability.task.scope.content_digest,
        exclusions_digest=capability.task.scope.exclusions_digest,
        issued_at_ns=capability.issued_at_ns,
        expires_at_ns=capability.expires_at_ns,
        revoked_at_ns=capability.revoked_at_ns,
        status=capability.status.value,
        constraint_names=tuple(item.name for item in capability.task.constraints),
    )


__all__ = [
    "ApprovedAddendum",
    "AuthorityConstraint",
    "AuthorityDenied",
    "AuthorityStatus",
    "CapabilityAuditEvidence",
    "CapabilityLookup",
    "DirectUserIntentReceipt",
    "ExactScope",
    "InputProvenance",
    "IntentCapability",
    "IntentCapabilityLedger",
    "OneGoalTaskAuthority",
    "ParentObjective",
    "TaskBinding",
    "TRUSTED_DIRECT_USER_GATEWAY_PLATFORMS",
    "activate_turn_authority",
    "audit_intent_capability",
    "bind_trusted_input",
    "current_direct_user_receipt",
    "current_intent_capability_ledger",
    "digest_clean_message",
    "digest_text",
    "issue_intent_capability",
]
