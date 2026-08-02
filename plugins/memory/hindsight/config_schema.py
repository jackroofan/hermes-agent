"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode",
            label="Mode",
            kind=KIND_SELECT,
            default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption(
                    "cloud",
                    "Cloud",
                    "Hindsight Cloud API (lightweight, just needs an API key)",
                ),
                ProviderFieldOption(
                    "local_external",
                    "Local External",
                    "Connect to an existing Hindsight instance",
                ),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key",
            label="API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key",
            inline=True,
        ),
        ProviderField(
            key="api_url",
            label="API URL",
            kind=KIND_TEXT,
            default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",),
            env_fallbacks=("HINDSIGHT_API_URL",),
            inline=True,
        ),
        ProviderField(
            key="bank_id",
            label="Bank ID",
            kind=KIND_TEXT,
            default="hermes",
            aliases=("bankId",),
            inline=True,
        ),
        ProviderField(
            key="recall_budget",
            label="Recall budget",
            kind=KIND_SELECT,
            default="mid",
            aliases=("budget",),
            options=(
                ProviderFieldOption("low", "low"),
                ProviderFieldOption("mid", "mid"),
                ProviderFieldOption("high", "high"),
            ),
            inline=True,
        ),
        ProviderField(
            key="recall_tags",
            label="Recall tags",
            kind=KIND_TEXT,
            description="Comma-separated tags to filter when searching memories.",
            placeholder="project:hermes, kind:note",
            inline=True,
        ),
        ProviderField(
            key="route_policy",
            label="Route policy",
            kind=KIND_TEXT,
            default="",
            description=(
                "Optional JSON for trusted project routes and the bounded "
                "general fallback. Each route has one canonical project and "
                "may declare project_alias_tags for legacy recall plus "
                "retain_project_alias_tags when compatibility retains must "
                "carry an alias. Project selection uses host metadata only."
            ),
            inline=True,
        ),
        ProviderField(
            key="recall_routes",
            label="Recall routes",
            kind=KIND_JSON,
            description=(
                "Compatibility routes keyed or matched by exact chat identity, "
                "with keyword fallback only for non-Project scope. Routes may "
                "set tags, exclusions, prefixes, caps, score floors, priority "
                "tags, retain tags, and automatic-recall controls."
            ),
            placeholder='{"chat-id": {"tags": ["profile:work"], "max_results": 2}}',
            group="Recall routing",
        ),
        ProviderField(
            key="recall_max_tokens",
            label="Recall token cap",
            kind=KIND_NUMBER,
            default="4096",
            description="Maximum tokens requested by recall and reflect operations.",
            group="Recall routing",
        ),
        ProviderField(
            key="recall_max_results",
            label="Recall result cap",
            kind=KIND_NUMBER,
            default="0",
            description="Global post-merge result cap; zero is unlimited and routes may override it.",
            group="Recall routing",
        ),
        ProviderField(
            key="recall_min_scores",
            label="Recall score floors",
            kind=KIND_JSON,
            description="Optional semantic, keyword, reranker, and final score floors.",
            placeholder='{"final": 0.6}',
            group="Recall routing",
        ),
        ProviderField(
            key="recall_skip_low_signal_queries",
            label="Skip low-signal recall",
            kind=KIND_BOOL,
            default="false",
            description="Suppress automatic recall for acknowledgement-like turns.",
            group="Recall routing",
        ),
        ProviderField(
            key="recall_low_signal_min_chars",
            label="Low-signal length",
            kind=KIND_NUMBER,
            default="0",
            description="Queries at least this long bypass low-signal suppression.",
            group="Recall routing",
        ),
        ProviderField(
            key="recall_domain_signal_keywords",
            label="Recall signal keywords",
            kind=KIND_TEXT,
            description="Comma-separated domain keywords that always allow automatic recall.",
            placeholder="memory, incident, 记忆",
            group="Recall routing",
        ),
    ),
)
