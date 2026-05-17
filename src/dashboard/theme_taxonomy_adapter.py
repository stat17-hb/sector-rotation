"""Dashboard adapter for projecting theme taxonomy onto runtime sector data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data_sources.theme_taxonomy import (
    ExecutionMapping,
    ThemeTaxonomy,
    ThemeTaxonomyEntry,
    load_theme_taxonomy_config,
)


@dataclass(frozen=True)
class TaxonomySectorContext:
    sector_code: str
    sector_name: str
    mapping_id: str
    taxonomy_label: str
    base_labels: tuple[str, ...]
    cross_labels: tuple[str, ...]
    theme_labels: tuple[str, ...]
    legacy_regime: str
    score_role: str
    action_policy: str
    export_series_alias: str


@dataclass(frozen=True)
class TaxonomyDashboardModel:
    market: str
    taxonomy_version: int
    sector_contexts: tuple[TaxonomySectorContext, ...]
    diagnostics: tuple[str, ...]

    @property
    def coverage_complete(self) -> bool:
        return not self.diagnostics

    @property
    def sector_count(self) -> int:
        return len(self.sector_contexts)

    def by_sector_code(self) -> dict[str, TaxonomySectorContext]:
        return {context.sector_code: context for context in self.sector_contexts}


def _sector_entries(sector_map: Mapping[str, Any] | None) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for regime, regime_payload in dict(sector_map or {}).get("regimes", {}).items():
        for raw_sector in list(dict(regime_payload or {}).get("sectors", [])):
            sector = dict(raw_sector or {})
            code = str(sector.get("code", "")).strip()
            name = str(sector.get("name", "")).strip()
            if code:
                entries.append({"code": code, "name": name or code, "regime": str(regime)})
    return entries


def _tag_label_lookup(taxonomy: ThemeTaxonomy) -> dict[str, str]:
    return {
        f"{axis.axis_id}/{child.tag_id}": child.name
        for axis in (
            *taxonomy.classification_axes.base_industries,
            *taxonomy.classification_axes.cross_themes,
        )
        for child in axis.children
    }


def _theme_label_lookup(themes: Sequence[ThemeTaxonomyEntry]) -> dict[str, str]:
    return {theme.theme_id: theme.name for theme in themes}


def _mapping_by_runtime_code(taxonomy: ThemeTaxonomy, *, market: str) -> dict[str, ExecutionMapping]:
    normalized_market = str(market or "").strip().upper()
    lookup: dict[str, ExecutionMapping] = {}
    for mapping in taxonomy.execution_mappings:
        for runtime_ref in mapping.runtime_index_refs:
            if runtime_ref.market == normalized_market:
                lookup[runtime_ref.index_code] = mapping
    return lookup


def _first_legacy_regime(mapping: ExecutionMapping) -> str:
    return mapping.legacy_sector_refs[0].regime if mapping.legacy_sector_refs else ""


def build_taxonomy_dashboard_model(
    *,
    taxonomy: ThemeTaxonomy | None = None,
    sector_map: Mapping[str, Any] | None = None,
    market: str = "KR",
) -> TaxonomyDashboardModel | None:
    """Project static taxonomy metadata onto the current dashboard sector map."""
    normalized_market = str(market or "KR").strip().upper() or "KR"
    if normalized_market != "KR":
        return None
    taxonomy = taxonomy or load_theme_taxonomy_config()
    tag_labels = _tag_label_lookup(taxonomy)
    theme_labels = _theme_label_lookup(taxonomy.themes)
    mappings = _mapping_by_runtime_code(taxonomy, market=normalized_market)
    diagnostics: list[str] = []
    contexts: list[TaxonomySectorContext] = []
    for sector in _sector_entries(sector_map):
        code = sector["code"]
        mapping = mappings.get(code)
        if mapping is None:
            diagnostics.append(f"unmapped sector_map.yml sector: {code} {sector['name']}")
            continue
        contexts.append(
            TaxonomySectorContext(
                sector_code=code,
                sector_name=sector["name"],
                mapping_id=mapping.mapping_id,
                taxonomy_label=mapping.label,
                base_labels=tuple(tag_labels.get(ref, ref) for ref in mapping.base_tag_refs),
                cross_labels=tuple(tag_labels.get(ref, ref) for ref in mapping.cross_tag_refs),
                theme_labels=tuple(theme_labels.get(ref, ref) for ref in mapping.theme_refs),
                legacy_regime=sector.get("regime") or _first_legacy_regime(mapping),
                score_role=mapping.scoring_policy.score_role,
                action_policy=mapping.scoring_policy.action_policy,
                export_series_alias=mapping.export_series_alias,
            )
        )
    return TaxonomyDashboardModel(
        market=taxonomy.market,
        taxonomy_version=taxonomy.taxonomy_version,
        sector_contexts=tuple(contexts),
        diagnostics=tuple(diagnostics),
    )


def taxonomy_context_for_sector(
    taxonomy_context: TaxonomyDashboardModel | None,
    sector_code: str,
) -> TaxonomySectorContext | None:
    if taxonomy_context is None:
        return None
    return taxonomy_context.by_sector_code().get(str(sector_code or "").strip())


def taxonomy_summary_cards(taxonomy_context: TaxonomyDashboardModel | None) -> list[dict[str, str]]:
    if taxonomy_context is None:
        return []
    cross_count = sum(1 for context in taxonomy_context.sector_contexts if context.cross_labels)
    return [
        {"label": "Taxonomy", "value": f"v{taxonomy_context.taxonomy_version}"},
        {"label": "런타임 매핑", "value": f"{taxonomy_context.sector_count}개"},
        {"label": "크로스테마", "value": f"{cross_count}개 섹터"},
        {"label": "커버리지", "value": "완료" if taxonomy_context.coverage_complete else "확인 필요"},
    ]


def enrich_lookup_sector_rows(
    rows: Sequence[Mapping[str, Any]],
    taxonomy_context: TaxonomyDashboardModel | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    lookup = taxonomy_context.by_sector_code() if taxonomy_context is not None else {}
    for row in rows:
        item = dict(row)
        context = lookup.get(str(item.get("sector_code", "")).strip())
        if context is not None:
            item["taxonomy_label"] = context.taxonomy_label
            item["taxonomy_base_labels"] = list(context.base_labels)
            item["taxonomy_cross_labels"] = list(context.cross_labels)
            item["taxonomy_theme_labels"] = list(context.theme_labels)
        enriched.append(item)
    return enriched
