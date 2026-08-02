"""Behavior tests for trusted Controller Project identity resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.project_identity import resolve_project_identity


PROJECT = "hermes-agent-engineering"


def _registry_documents(
    root: Path, repository: Path, workspace: Path
) -> tuple[dict, dict, Path]:
    card_dir = root / "projects" / "coding" / PROJECT
    card_dir.mkdir(parents=True)
    brief_path = card_dir / "PROJECT.md"
    brief_path.write_text("# Test Project\n", encoding="utf-8")
    portfolio = {
        "schema_version": 2,
        "stable_parents": {
            PROJECT: {
                "repository": str(repository),
                "project_brief_path": str(brief_path),
            }
        },
    }
    card = {
        "approved_clients": ["codex"],
        "clients": {},
        "codex_enabled": True,
        "display_name": "Hermes Agent Engineering",
        "execution_workspaces": [str(workspace)],
        "portfolio": {"role": "stable_parent", "schema_version": 2},
        "portfolio_role": "stable_parent",
        "project_brief_path": str(brief_path),
        "project_type": "code",
        "repository": str(repository),
        "status": "active",
    }
    return portfolio, card, card_dir


def _write_registry(
    root: Path,
    repository: Path,
    workspace: Path,
    *,
    portfolio_mutation=None,
    card_mutation=None,
    malformed: str = "",
) -> None:
    portfolio, card, card_dir = _registry_documents(root, repository, workspace)
    if portfolio_mutation:
        portfolio_mutation(portfolio)
    if card_mutation:
        card_mutation(card)
    portfolio_path = root / "projects" / "coding" / "portfolio.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    if malformed == "portfolio":
        portfolio_path.write_text("{not-json", encoding="utf-8")
    else:
        portfolio_path.write_text(json.dumps(portfolio), encoding="utf-8")
    card_path = card_dir / "project.json"
    if malformed == "card":
        card_path.write_text("{not-json", encoding="utf-8")
    else:
        card_path.write_text(json.dumps(card), encoding="utf-8")


def _set_profile_home(monkeypatch, root: Path) -> Path:
    profile_home = root / "profiles" / "coding"
    profile_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    return profile_home


def test_controller_registry_maps_repository_and_execution_workspace(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes-root"
    repository = tmp_path / "canonical-repository"
    workspace = tmp_path / "execution-workspace"
    repository.mkdir()
    workspace.mkdir()
    _write_registry(root, repository, workspace)
    profile_home = _set_profile_home(monkeypatch, root)

    repository_identity = resolve_project_identity(session_cwd=str(repository))
    workspace_identity = resolve_project_identity(runtime_cwd=str(workspace))

    assert not (profile_home / "projects.db").exists()
    assert repository_identity == {
        "project_id": PROJECT,
        "project_slug": PROJECT,
        "project_name": "Hermes Agent Engineering",
        "project_source": "controller_registry",
        "project_match": "canonical_repository",
    }
    assert workspace_identity["project_slug"] == PROJECT
    assert workspace_identity["project_source"] == "controller_registry"
    assert workspace_identity["project_match"] == "execution_workspace"


def test_profile_local_home_uses_only_canonical_root_registry(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    _write_registry(root, repository, workspace)
    profile_home = _set_profile_home(monkeypatch, root)

    # A conflicting registry below the profile must never be consulted.
    local_coding = profile_home / "projects" / "coding"
    local_coding.mkdir(parents=True)
    (local_coding / "portfolio.json").write_text("{not-json", encoding="utf-8")

    identity = resolve_project_identity(git_repo_root=str(repository))

    assert identity["project_slug"] == PROJECT
    assert identity["project_source"] == "controller_registry"


@pytest.mark.parametrize(
    ("case", "portfolio_mutation", "card_mutation", "malformed"),
    [
        (
            "unindexed",
            lambda value: value.update(stable_parents={}),
            None,
            "",
        ),
        (
            "portfolio-schema-v1",
            lambda value: value.update(schema_version=1),
            None,
            "",
        ),
        (
            "card-schema-v1",
            None,
            lambda value: value["portfolio"].update(schema_version=1),
            "",
        ),
        (
            "inactive",
            None,
            lambda value: value.update(status="inactive"),
            "",
        ),
        (
            "alias-card",
            None,
            lambda value: value.update(portfolio_role="alias"),
            "",
        ),
        (
            "non-codex-clients",
            None,
            lambda value: value.update(approved_clients=["codex", "claude"]),
            "",
        ),
        ("malformed-portfolio", None, None, "portfolio"),
        ("malformed-card", None, None, "card"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_invalid_or_unindexed_controller_cards_do_not_establish_identity(
    tmp_path,
    monkeypatch,
    case,
    portfolio_mutation,
    card_mutation,
    malformed,
):
    del case
    root = tmp_path / "root"
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    _write_registry(
        root,
        repository,
        workspace,
        portfolio_mutation=portfolio_mutation,
        card_mutation=card_mutation,
        malformed=malformed,
    )
    _set_profile_home(monkeypatch, root)

    identity = resolve_project_identity(session_cwd=str(repository))
    assert "project_id" not in identity
    assert "project_slug" not in identity


def test_path_boundary_and_symlink_escape_do_not_match(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    repository = tmp_path / "repo"
    sibling = tmp_path / "repo-sibling"
    workspace = tmp_path / "workspace"
    escaped = tmp_path / "outside"
    for path in (repository, sibling, workspace, escaped):
        path.mkdir()
    _write_registry(root, repository, workspace)
    _set_profile_home(monkeypatch, root)
    escape_link = workspace / "escape"
    try:
        escape_link.symlink_to(escaped, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    assert resolve_project_identity(session_cwd=str(sibling)) == {}
    assert resolve_project_identity(session_cwd=str(escape_link)) == {}


def test_controller_identity_precedes_conflicting_profile_project_fallback(
    tmp_path, monkeypatch
):
    from hermes_cli import projects_db

    root = tmp_path / "root"
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    fallback_only = tmp_path / "fallback-only"
    for path in (repository, workspace, fallback_only):
        path.mkdir()
    _write_registry(root, repository, workspace)
    profile_home = _set_profile_home(monkeypatch, root)
    with projects_db.connect_closing(db_path=profile_home / "projects.db") as conn:
        projects_db.create_project(
            conn,
            name="Conflicting Generic Project",
            slug="wrong-project",
            folders=[str(repository)],
        )
        projects_db.create_project(
            conn,
            name="Fallback Project",
            slug="fallback-project",
            folders=[str(fallback_only)],
        )

    authoritative = resolve_project_identity(runtime_cwd=str(repository))
    fallback = resolve_project_identity(runtime_cwd=str(fallback_only))

    assert authoritative["project_slug"] == PROJECT
    assert authoritative["project_source"] == "controller_registry"
    assert fallback["project_slug"] == "fallback-project"
    assert fallback["project_source"] == "projects_db"


def test_invalid_controller_registry_suppresses_profile_project_fallback(
    tmp_path, monkeypatch
):
    from hermes_cli import projects_db

    root = tmp_path / "root"
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    _write_registry(root, repository, workspace, malformed="portfolio")
    profile_home = _set_profile_home(monkeypatch, root)
    with projects_db.connect_closing(db_path=profile_home / "projects.db") as conn:
        projects_db.create_project(
            conn,
            name="Must Not Override Invalid Authority",
            slug="generic-project",
            folders=[str(repository)],
        )

    assert resolve_project_identity(runtime_cwd=str(repository)) == {
        "project_source": "controller_registry_invalid",
        "project_match": "invalid",
    }
