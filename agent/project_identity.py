"""Read-only trusted Project identity resolution for memory routing.

The Project Controller registry under the canonical Hermes root is the
authoritative source.  The per-profile ``projects.db`` remains a compatibility
fallback only when the Controller registry is absent or valid but has no match.
This module deliberately accepts trusted path candidates only; it never reads
conversation content or accepts a caller-selected registry root.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_MAX_PORTFOLIO_BYTES = 2 * 1024 * 1024
_MAX_PROJECT_CARD_BYTES = 2 * 1024 * 1024
_IDENTITY_KEYS = (
    "project_id",
    "project_slug",
    "project_name",
    "project_source",
    "project_match",
)


class _RegistryInvalid(Exception):
    """An authoritative registry exists but cannot safely establish scope."""


def _load_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        if path.stat().st_size > max_bytes:
            raise _RegistryInvalid("file_too_large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except _RegistryInvalid:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _RegistryInvalid(type(exc).__name__) from exc
    if not isinstance(value, dict):
        raise _RegistryInvalid("not_an_object")
    return value


def _real_directory(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser()
        if not path.is_absolute() or not path.is_dir():
            return None
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _candidate_directories(*values: Any) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for value in values:
        candidate = _real_directory(value)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _validated_project_roots(
    *,
    controller_root: Path,
    slug: str,
    indexed: dict[str, Any],
) -> tuple[str, tuple[tuple[str, Path], ...]]:
    card_dir = controller_root / "projects" / "coding" / slug
    try:
        coding_dir = (controller_root / "projects" / "coding").resolve(strict=True)
        resolved_card_dir = card_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _RegistryInvalid("invalid_project_card_path") from exc
    if resolved_card_dir.parent != coding_dir or resolved_card_dir.name != slug:
        raise _RegistryInvalid("invalid_project_card_path")
    card = _load_json_object(
        resolved_card_dir / "project.json", max_bytes=_MAX_PROJECT_CARD_BYTES
    )
    portfolio = card.get("portfolio")
    approved_clients = card.get("approved_clients")
    explicit_identity = str(
        card.get("project_slug") or card.get("project_id") or card.get("id") or ""
    ).strip()
    if (
        not isinstance(portfolio, dict)
        or portfolio.get("schema_version") != 2
        or portfolio.get("role") != "stable_parent"
        or card.get("portfolio_role") != "stable_parent"
        or card.get("status") != "active"
        or card.get("project_type") != "code"
        or approved_clients != ["codex"]
        or card.get("codex_enabled") is not True
        or (explicit_identity and explicit_identity != slug)
    ):
        raise _RegistryInvalid("invalid_project_contract")

    repository = _real_directory(card.get("repository"))
    indexed_repository = _real_directory(indexed.get("repository"))
    if repository is None or indexed_repository is None or repository != indexed_repository:
        raise _RegistryInvalid("repository_identity_mismatch")

    expected_brief = resolved_card_dir / "PROJECT.md"
    indexed_brief = Path(str(indexed.get("project_brief_path") or "")).expanduser()
    card_brief = Path(str(card.get("project_brief_path") or "")).expanduser()
    try:
        if (
            not indexed_brief.is_absolute()
            or not card_brief.is_absolute()
            or not expected_brief.is_file()
            or indexed_brief.resolve(strict=False) != expected_brief
            or card_brief.resolve(strict=False) != expected_brief
        ):
            raise _RegistryInvalid("project_brief_identity_mismatch")
    except (OSError, RuntimeError) as exc:
        raise _RegistryInvalid("project_brief_identity_mismatch") from exc

    roots: list[tuple[str, Path]] = [("canonical_repository", repository)]
    workspaces = card.get("execution_workspaces")
    if workspaces is None:
        workspaces = []
    if not isinstance(workspaces, list):
        raise _RegistryInvalid("invalid_execution_workspaces")
    for workspace_value in workspaces:
        workspace = _real_directory(workspace_value)
        if workspace is None:
            continue
        if workspace not in (root for _, root in roots):
            roots.append(("execution_workspace", workspace))

    project_name = str(card.get("display_name") or slug).strip()[:256] or slug
    return project_name, tuple(roots)


def _controller_identity(candidates: tuple[Path, ...]) -> tuple[str, dict[str, str]]:
    """Return ``(registry_state, identity)``.

    ``registry_state`` is ``absent``, ``valid``, or ``invalid``.  An invalid
    authoritative registry must fail closed and therefore suppress the generic
    per-profile fallback.
    """
    from hermes_constants import get_default_hermes_root

    controller_root = get_default_hermes_root().resolve(strict=False)
    portfolio_path = controller_root / "projects" / "coding" / "portfolio.json"
    if not portfolio_path.exists():
        return "absent", {}
    try:
        coding_dir = (controller_root / "projects" / "coding").resolve(strict=True)
        if portfolio_path.resolve(strict=True).parent != coding_dir:
            raise _RegistryInvalid("invalid_portfolio_path")
        portfolio = _load_json_object(
            portfolio_path, max_bytes=_MAX_PORTFOLIO_BYTES
        )
        stable_parents = portfolio.get("stable_parents")
        if portfolio.get("schema_version") != 2 or not isinstance(stable_parents, dict):
            raise _RegistryInvalid("invalid_portfolio_contract")

        matches: list[dict[str, str]] = []
        invalid_card_seen = False
        for slug in sorted(stable_parents):
            indexed = stable_parents[slug]
            if (
                not isinstance(slug, str)
                or not _PROJECT_SLUG_RE.fullmatch(slug)
                or not isinstance(indexed, dict)
            ):
                raise _RegistryInvalid("invalid_stable_parent_index")
            try:
                project_name, roots = _validated_project_roots(
                    controller_root=controller_root,
                    slug=slug,
                    indexed=indexed,
                )
            except _RegistryInvalid:
                invalid_card_seen = True
                continue
            for candidate in candidates:
                for classification, root in roots:
                    if _contains(root, candidate):
                        matches.append({
                            "project_id": slug,
                            "project_slug": slug,
                            "project_name": project_name,
                            "project_source": "controller_registry",
                            "project_match": classification,
                        })
                        break
                else:
                    continue
                break

        unique = {
            (match["project_id"], match["project_match"]): match
            for match in matches
        }
        project_ids = {project_id for project_id, _ in unique}
        if len(project_ids) > 1:
            raise _RegistryInvalid("ambiguous_project_match")
        if unique:
            # Prefer the canonical repository classification when both a
            # session path and a workspace path identify the same parent.
            return "valid", sorted(
                unique.values(),
                key=lambda item: item["project_match"] != "canonical_repository",
            )[0]
        if invalid_card_seen:
            raise _RegistryInvalid("invalid_indexed_project_card")
        return "valid", {}
    except _RegistryInvalid as exc:
        logger.warning(
            "Project Controller registry ignored for memory routing: %s",
            str(exc)[:96],
        )
        return "invalid", {}


def _projects_db_identity(candidates: tuple[Path, ...]) -> dict[str, str]:
    """Resolve the lower-priority per-profile compatibility Project store."""
    try:
        from hermes_cli import projects_db
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "projects.db"
        if not db_path.is_file():
            return {}
        uri = f"{db_path.resolve(strict=True).as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for candidate in candidates:
                project = projects_db.project_for_path(conn, str(candidate))
                if project is not None:
                    return {
                        "project_id": str(project.id)[:128],
                        "project_slug": str(project.slug)[:128],
                        "project_name": str(project.name)[:256],
                        "project_source": "projects_db",
                        "project_match": "projects_db",
                    }
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(
            "Profile Project fallback ignored for memory routing: %s",
            type(exc).__name__,
        )
    return {}


def resolve_project_identity(
    *,
    session_cwd: str = "",
    git_repo_root: str = "",
    runtime_cwd: str = "",
) -> dict[str, str]:
    """Resolve bounded Project identity from trusted current path metadata.

    Controller identity is scope context only.  It grants no write,
    deployment, credential, or current-state authority.
    """
    candidates = _candidate_directories(session_cwd, git_repo_root, runtime_cwd)
    if not candidates:
        return {}
    registry_state, identity = _controller_identity(candidates)
    if identity:
        return {key: identity[key] for key in _IDENTITY_KEYS}
    if registry_state == "invalid":
        return {
            "project_source": "controller_registry_invalid",
            "project_match": "invalid",
        }
    fallback = _projects_db_identity(candidates)
    return {key: fallback[key] for key in _IDENTITY_KEYS} if fallback else {}
