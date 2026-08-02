"""Tests for Hindsight's declared config surface."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    get_provider_config_schema,
)


def test_hindsight_is_declared():
    provider = get_provider_config_schema("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
        "recall_tags",
        "route_policy",
        "recall_routes",
        "recall_max_tokens",
        "recall_max_results",
        "recall_min_scores",
        "recall_skip_low_signal_queries",
        "recall_low_signal_min_chars",
        "recall_domain_signal_keywords",
    }


def test_fields_are_all_inline():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    assert {field.key for field in provider.inline_fields()} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
        "recall_tags",
        "route_policy",
    }


def test_route_controls_are_grouped_outside_the_compact_panel():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    route_fields = {
        field.key: field
        for field in provider.fields
        if field.group == "Recall routing"
    }
    assert route_fields
    assert all(not field.inline for field in route_fields.values())


def test_route_field_kinds_preserve_structured_config():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None
    fields = {field.key: field for field in provider.fields}

    assert fields["recall_routes"].kind == KIND_JSON
    assert fields["recall_min_scores"].kind == KIND_JSON
    assert fields["recall_skip_low_signal_queries"].kind == KIND_BOOL
    assert fields["recall_max_tokens"].kind == KIND_NUMBER
    assert fields["recall_max_results"].kind == KIND_NUMBER
    assert fields["recall_low_signal_min_chars"].kind == KIND_NUMBER


def test_mode_gating_is_expressed_as_select_options():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # local_embedded is intentionally unsupported on desktop.
    assert "local_embedded" not in mode.allowed_values()


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"


def test_recall_tags_is_declared_as_comma_separated_text():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    recall_tags = next(field for field in provider.fields if field.key == "recall_tags")
    assert recall_tags.kind == KIND_TEXT
    assert "comma-separated" in recall_tags.description.lower()
