"""KR theme taxonomy metadata.

This module deliberately owns taxonomy metadata only. It must not import
warehouse, pykrx, or dashboard modules because taxonomy loading is a static
config validation path, not a price/proxy refresh path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

import yaml


THEME_TAXONOMY_CONFIG_PATH = Path("config/theme_taxonomy.yml")
THEME_AUTHORITY_BASIS_TYPES: tuple[str, ...] = (
    "ETF_PRODUCT_THEME_INDEX",
    "ETF_COMPARISON_INDEX",
    "ASSET_MANAGER_CATEGORY",
    "OFFICIAL_INDEX_OR_INDUSTRY_ANCHOR",
)
PRODUCT_AUTHORITY_BASIS_TYPES: tuple[str, ...] = (
    "ETF_PRODUCT_THEME_INDEX",
    "ETF_COMPARISON_INDEX",
    "ASSET_MANAGER_CATEGORY",
)
OFFICIAL_ANCHOR_BASIS_TYPE = "OFFICIAL_INDEX_OR_INDUSTRY_ANCHOR"
THEME_SOURCE_ROLES: tuple[str, ...] = (
    "OFFICIAL_ISSUER_PRODUCT_PAGE",
    "OFFICIAL_ISSUER_FACTSHEET",
    "OFFICIAL_PROSPECTUS",
    "OFFICIAL_INDEX_PROVIDER_PAGE",
    "OFFICIAL_EXCHANGE_PAGE",
    "AGGREGATOR_REFERENCE",
)
AGGREGATOR_SOURCE_ROLE = "AGGREGATOR_REFERENCE"
AGGREGATOR_PRIMARY_HOSTS: tuple[str, ...] = (
    "k-etf.com",
    "www.k-etf.com",
)
VERIFICATION_STATUSES: tuple[str, ...] = (
    "verified",
    "needs_review",
)
EXECUTION_MAPPING_ROLES: tuple[str, ...] = (
    "legacy_sector_bridge",
    "benchmark_reference",
)
EXECUTION_SCORE_ROLES: tuple[str, ...] = (
    "driver",
    "reference",
    "bridge",
)


@dataclass(frozen=True)
class ThemeSourceReference:
    role: str
    url: str
    label: str


@dataclass(frozen=True)
class ThemeAuthority:
    provider: str
    basis_type: str
    index_name: str
    source_url: str
    source_role: str
    supporting_urls: tuple[ThemeSourceReference, ...]
    evidence_note: str


@dataclass(frozen=True)
class ThemeProduct:
    code: str
    name: str
    issuer: str
    product_type: str
    benchmark_index: str
    comparison_index: str
    source_url: str
    source_role: str
    supporting_urls: tuple[ThemeSourceReference, ...]
    evidence_note: str


@dataclass(frozen=True)
class ThemeAnchor:
    provider: str
    label: str
    source_url: str
    source_role: str
    supporting_urls: tuple[ThemeSourceReference, ...]
    evidence_note: str


@dataclass(frozen=True)
class ExposureRule:
    primary_metric: str
    pure_play: str
    relevant: str
    satellite: str


@dataclass(frozen=True)
class TaxonomyHistoryEntry:
    version: int
    effective_from: str
    note: str


@dataclass(frozen=True)
class ClassificationTag:
    tag_id: str
    name: str
    aliases: tuple[str, ...]
    inclusion_rule: str
    exposure_rule: ExposureRule
    history: tuple[TaxonomyHistoryEntry, ...]


@dataclass(frozen=True)
class ClassificationAxis:
    axis_id: str
    name: str
    assignment: str
    children: tuple[ClassificationTag, ...]


@dataclass(frozen=True)
class ClassificationAxes:
    base_industries: tuple[ClassificationAxis, ...]
    cross_themes: tuple[ClassificationAxis, ...]


@dataclass(frozen=True)
class TaxonomyVerification:
    last_verified_at: date
    verification_status: str
    stale_after_days: int

    def is_stale(self, as_of: date | str) -> bool:
        if isinstance(as_of, str):
            as_of_date = _parse_iso_date(as_of, "as_of")
        else:
            as_of_date = as_of
        return (as_of_date - self.last_verified_at).days > self.stale_after_days


@dataclass(frozen=True)
class ThemeMapping:
    theme_id: str
    base_tag_refs: tuple[str, ...]
    cross_tag_refs: tuple[str, ...]
    mapping_rule: str


@dataclass(frozen=True)
class RuntimeIndexRef:
    market: str
    index_code: str
    index_name: str
    source: str


@dataclass(frozen=True)
class LegacySectorRef:
    source: str
    regime: str
    code: str
    name: str


@dataclass(frozen=True)
class ExecutionScoringPolicy:
    score_role: str
    action_policy: str


@dataclass(frozen=True)
class ExecutionFlowJoin:
    sector_code: str
    sector_name: str


@dataclass(frozen=True)
class ExecutionRepresentativeEtf:
    code: str
    name: str


@dataclass(frozen=True)
class ExecutionMapping:
    mapping_id: str
    runtime_role: str
    label: str
    base_tag_refs: tuple[str, ...]
    cross_tag_refs: tuple[str, ...]
    theme_refs: tuple[str, ...]
    runtime_index_refs: tuple[RuntimeIndexRef, ...]
    legacy_sector_refs: tuple[LegacySectorRef, ...]
    scoring_policy: ExecutionScoringPolicy
    flow_join: ExecutionFlowJoin | None
    representative_etfs: tuple[ExecutionRepresentativeEtf, ...]
    export_series_alias: str


@dataclass(frozen=True)
class ThemeTaxonomyEntry:
    theme_id: str
    name: str
    aliases: tuple[str, ...]
    primary_authority: ThemeAuthority
    products: tuple[ThemeProduct, ...]
    secondary_anchors: tuple[ThemeAnchor, ...]


@dataclass(frozen=True)
class ThemeTaxonomy:
    market: str
    taxonomy_version: int
    authority_priority: tuple[str, ...]
    verification: TaxonomyVerification
    classification_axes: ClassificationAxes
    execution_mappings: tuple[ExecutionMapping, ...]
    theme_mappings: tuple[ThemeMapping, ...]
    themes: tuple[ThemeTaxonomyEntry, ...]


def _coerce_text(value: object) -> str:
    return str(value or "").strip()


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_text(value: object, context: str) -> str:
    text = _coerce_text(value)
    if not text:
        raise ValueError(f"{context} is required")
    return text


def _parse_iso_date(value: object, context: str) -> date:
    text = _require_text(value, context)
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        raise ValueError(f"{context} must be an ISO YYYY-MM-DD date")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO YYYY-MM-DD date") from exc


def _has_source_evidence(item: Mapping[str, object]) -> bool:
    return bool(_coerce_text(item.get("source_url")) or _coerce_text(item.get("evidence_note")))


def _url_host(value: str) -> str:
    return (urlparse(value).netloc or "").lower()


def _is_aggregator_url(value: str) -> bool:
    host = _url_host(value)
    return host in AGGREGATOR_PRIMARY_HOSTS or host.endswith(".k-etf.com")


def _parse_source_role(value: object, context: str, *, allow_aggregator: bool) -> str:
    role = _require_text(value, context)
    if role not in THEME_SOURCE_ROLES:
        raise ValueError(f"{context} has invalid source_role: {role}")
    if not allow_aggregator and role == AGGREGATOR_SOURCE_ROLE:
        raise ValueError(f"{context} cannot use AGGREGATOR_REFERENCE as primary source_role")
    return role


def _validate_primary_source_url(source_url: str, *, context: str) -> None:
    if not source_url:
        raise ValueError(f"{context} requires source_url")
    if _is_aggregator_url(source_url):
        raise ValueError(f"{context} cannot use k-etf.com as primary source_url")


def _parse_supporting_urls(value: object, *, context: str) -> tuple[ThemeSourceReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context}.supporting_urls must be a list")
    references: list[ThemeSourceReference] = []
    seen_urls: set[str] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"{context}.supporting_urls {index}")
        role = _parse_source_role(
            item.get("role"),
            f"{context}.supporting_urls {index}.role",
            allow_aggregator=True,
        )
        url = _require_text(item.get("url"), f"{context}.supporting_urls {index}.url")
        if url in seen_urls:
            raise ValueError(f"{context}.supporting_urls contains duplicate url: {url}")
        seen_urls.add(url)
        references.append(
            ThemeSourceReference(
                role=role,
                url=url,
                label=_coerce_text(item.get("label")),
            )
        )
    return tuple(references)


def _load_yaml(path: Path) -> Mapping[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(payload, Mapping):
        raise ValueError("theme taxonomy config must be a mapping")
    return payload


def _parse_authority_priority(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw_priority = payload.get("authority_priority")
    if not isinstance(raw_priority, list) or not raw_priority:
        raise ValueError("authority_priority must be a non-empty list")
    priority = tuple(_require_text(item, "authority_priority item") for item in raw_priority)
    if len(set(priority)) != len(priority):
        raise ValueError("authority_priority contains duplicate basis types")
    unknown = [item for item in priority if item not in THEME_AUTHORITY_BASIS_TYPES]
    if unknown:
        raise ValueError(f"invalid authority_priority basis type: {unknown[0]}")
    if OFFICIAL_ANCHOR_BASIS_TYPE not in priority:
        raise ValueError("authority_priority must include OFFICIAL_INDEX_OR_INDUSTRY_ANCHOR")
    missing_product = [item for item in PRODUCT_AUTHORITY_BASIS_TYPES if item not in priority]
    if missing_product:
        raise ValueError(f"authority_priority missing product authority basis type: {missing_product[0]}")
    official_rank = priority.index(OFFICIAL_ANCHOR_BASIS_TYPE)
    highest_product_rank = max(priority.index(item) for item in PRODUCT_AUTHORITY_BASIS_TYPES)
    if official_rank < highest_product_rank:
        raise ValueError("official industry anchors cannot outrank product/theme-index authority")
    return priority


def _parse_aliases(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    aliases: list[str] = []
    for item in value:
        text = _coerce_text(item)
        if text:
            aliases.append(text)
    return tuple(aliases)


def _parse_primary_authority(value: object, *, theme_id: str) -> ThemeAuthority:
    item = _require_mapping(value, f"theme {theme_id} primary_authority")
    if not _has_source_evidence(item):
        raise ValueError(f"theme {theme_id} primary_authority requires source_url or evidence_note")
    basis_type = _require_text(item.get("basis_type"), f"theme {theme_id} primary_authority.basis_type")
    if basis_type not in THEME_AUTHORITY_BASIS_TYPES:
        raise ValueError(f"theme {theme_id} has invalid primary_authority.basis_type: {basis_type}")
    source_role = _parse_source_role(
        item.get("source_role"),
        f"theme {theme_id} primary_authority.source_role",
        allow_aggregator=False,
    )
    source_url = _coerce_text(item.get("source_url"))
    _validate_primary_source_url(source_url, context=f"theme {theme_id} primary_authority")
    return ThemeAuthority(
        provider=_require_text(item.get("provider"), f"theme {theme_id} primary_authority.provider"),
        basis_type=basis_type,
        index_name=_coerce_text(item.get("index_name")),
        source_url=source_url,
        source_role=source_role,
        supporting_urls=_parse_supporting_urls(
            item.get("supporting_urls"),
            context=f"theme {theme_id} primary_authority",
        ),
        evidence_note=_coerce_text(item.get("evidence_note")),
    )


def _parse_products(value: object, *, theme_id: str) -> tuple[ThemeProduct, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"theme {theme_id} requires at least one product")
    products: list[ThemeProduct] = []
    seen_codes: set[str] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"theme {theme_id} product {index}")
        if not _has_source_evidence(item):
            raise ValueError(f"theme {theme_id} product {index} requires source_url or evidence_note")
        code_value = item.get("code")
        if not isinstance(code_value, str):
            raise ValueError(f"theme {theme_id} product {index} code must be a string")
        code = _require_text(code_value, f"theme {theme_id} product {index}.code")
        if code in seen_codes:
            raise ValueError(f"theme {theme_id} contains duplicate product code: {code}")
        seen_codes.add(code)
        source_role = _parse_source_role(
            item.get("source_role"),
            f"theme {theme_id} product {index}.source_role",
            allow_aggregator=False,
        )
        source_url = _coerce_text(item.get("source_url"))
        _validate_primary_source_url(source_url, context=f"theme {theme_id} product {index}")
        products.append(
            ThemeProduct(
                code=code,
                name=_require_text(item.get("name"), f"theme {theme_id} product {index}.name"),
                issuer=_coerce_text(item.get("issuer")),
                product_type=_require_text(item.get("product_type"), f"theme {theme_id} product {index}.product_type"),
                benchmark_index=_coerce_text(item.get("benchmark_index")),
                comparison_index=_coerce_text(item.get("comparison_index")),
                source_url=source_url,
                source_role=source_role,
                supporting_urls=_parse_supporting_urls(
                    item.get("supporting_urls"),
                    context=f"theme {theme_id} product {index}",
                ),
                evidence_note=_coerce_text(item.get("evidence_note")),
            )
        )
    return tuple(products)


def _parse_secondary_anchors(value: object, *, theme_id: str) -> tuple[ThemeAnchor, ...]:
    if not isinstance(value, list):
        return ()
    anchors: list[ThemeAnchor] = []
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"theme {theme_id} secondary_anchor {index}")
        if not _has_source_evidence(item):
            raise ValueError(f"theme {theme_id} secondary_anchor {index} requires source_url or evidence_note")
        source_url = _coerce_text(item.get("source_url"))
        source_role = ""
        if source_url or item.get("source_role") is not None:
            source_role = _parse_source_role(
                item.get("source_role"),
                f"theme {theme_id} secondary_anchor {index}.source_role",
                allow_aggregator=True,
            )
        anchors.append(
            ThemeAnchor(
                provider=_require_text(item.get("provider"), f"theme {theme_id} secondary_anchor {index}.provider"),
                label=_require_text(item.get("label"), f"theme {theme_id} secondary_anchor {index}.label"),
                source_url=source_url,
                source_role=source_role,
                supporting_urls=_parse_supporting_urls(
                    item.get("supporting_urls"),
                    context=f"theme {theme_id} secondary_anchor {index}",
                ),
                evidence_note=_coerce_text(item.get("evidence_note")),
            )
        )
    return tuple(anchors)


def _parse_exposure_rule(value: object, *, context: str) -> ExposureRule:
    item = _require_mapping(value, f"{context}.exposure_rule")
    return ExposureRule(
        primary_metric=_require_text(item.get("primary_metric"), f"{context}.exposure_rule.primary_metric"),
        pure_play=_require_text(item.get("pure_play"), f"{context}.exposure_rule.pure_play"),
        relevant=_require_text(item.get("relevant"), f"{context}.exposure_rule.relevant"),
        satellite=_require_text(item.get("satellite"), f"{context}.exposure_rule.satellite"),
    )


def _parse_history(value: object, *, context: str) -> tuple[TaxonomyHistoryEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.history must be a non-empty list")
    history: list[TaxonomyHistoryEntry] = []
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"{context}.history {index}")
        version_raw = item.get("version")
        try:
            version = int(version_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.history {index}.version must be an integer") from exc
        if version < 1:
            raise ValueError(f"{context}.history {index}.version must be positive")
        history.append(
            TaxonomyHistoryEntry(
                version=version,
                effective_from=_require_text(item.get("effective_from"), f"{context}.history {index}.effective_from"),
                note=_require_text(item.get("note"), f"{context}.history {index}.note"),
            )
        )
    return tuple(history)


def _parse_classification_tag(value: object, *, context: str) -> ClassificationTag:
    item = _require_mapping(value, context)
    tag_id = _require_text(item.get("tag_id"), f"{context}.tag_id")
    return ClassificationTag(
        tag_id=tag_id,
        name=_require_text(item.get("name"), f"{context}.name"),
        aliases=_parse_aliases(item.get("aliases")),
        inclusion_rule=_require_text(item.get("inclusion_rule"), f"{context}.inclusion_rule"),
        exposure_rule=_parse_exposure_rule(item.get("exposure_rule"), context=context),
        history=_parse_history(item.get("history"), context=context),
    )


def _parse_classification_axis(value: object, *, context: str, expected_assignment: str) -> ClassificationAxis:
    item = _require_mapping(value, context)
    axis_id = _require_text(item.get("axis_id"), f"{context}.axis_id")
    assignment = _require_text(item.get("assignment"), f"{context}.assignment")
    if assignment != expected_assignment:
        raise ValueError(f"{context}.assignment must be {expected_assignment}")
    raw_children = item.get("children")
    if not isinstance(raw_children, list) or not raw_children:
        raise ValueError(f"{context}.children must be a non-empty list")
    children: list[ClassificationTag] = []
    seen_child_ids: set[str] = set()
    for index, raw_child in enumerate(raw_children):
        child = _parse_classification_tag(raw_child, context=f"{context}.children {index}")
        if child.tag_id in seen_child_ids:
            raise ValueError(f"{context} contains duplicate tag_id: {child.tag_id}")
        seen_child_ids.add(child.tag_id)
        children.append(child)
    return ClassificationAxis(
        axis_id=axis_id,
        name=_require_text(item.get("name"), f"{context}.name"),
        assignment=assignment,
        children=tuple(children),
    )


def _parse_axis_group(value: object, *, group_name: str, expected_assignment: str) -> tuple[ClassificationAxis, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"classification_axes.{group_name} must be a list")
    axes: list[ClassificationAxis] = []
    seen_axis_ids: set[str] = set()
    for index, raw_axis in enumerate(value):
        axis = _parse_classification_axis(
            raw_axis,
            context=f"classification_axes.{group_name} {index}",
            expected_assignment=expected_assignment,
        )
        if axis.axis_id in seen_axis_ids:
            raise ValueError(f"classification_axes.{group_name} contains duplicate axis_id: {axis.axis_id}")
        seen_axis_ids.add(axis.axis_id)
        axes.append(axis)
    return tuple(axes)


def _parse_classification_axes(value: object) -> ClassificationAxes:
    if value is None:
        return ClassificationAxes(base_industries=(), cross_themes=())
    item = _require_mapping(value, "classification_axes")
    return ClassificationAxes(
        base_industries=_parse_axis_group(
            item.get("base_industries"),
            group_name="base_industries",
            expected_assignment="single",
        ),
        cross_themes=_parse_axis_group(
            item.get("cross_themes"),
            group_name="cross_themes",
            expected_assignment="multiple",
        ),
    )


def _parse_verification(value: object) -> TaxonomyVerification:
    item = _require_mapping(value, "verification")
    verification_status = _require_text(item.get("verification_status"), "verification.verification_status")
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(f"verification.verification_status must be one of {VERIFICATION_STATUSES}")
    stale_after_days_raw = item.get("stale_after_days")
    try:
        stale_after_days = int(stale_after_days_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("verification.stale_after_days must be a positive integer") from exc
    if stale_after_days < 1:
        raise ValueError("verification.stale_after_days must be a positive integer")
    return TaxonomyVerification(
        last_verified_at=_parse_iso_date(item.get("last_verified_at"), "verification.last_verified_at"),
        verification_status=verification_status,
        stale_after_days=stale_after_days,
    )


def _collect_tag_refs(axes: tuple[ClassificationAxis, ...]) -> set[str]:
    return {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in axes
        for child in axis.children
    }


def _parse_tag_ref_list(
    value: object,
    *,
    context: str,
    valid_refs: set[str],
    require_non_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    refs = tuple(_require_text(item, f"{context} item") for item in value)
    if require_non_empty and not refs:
        raise ValueError(f"{context} must be non-empty")
    for ref in refs:
        if "/" not in ref:
            raise ValueError(f"{context} must use axis_id/tag_id refs: {ref}")
        if ref not in valid_refs:
            raise ValueError(f"{context} references unknown tag ref: {ref}")
    return refs


def _parse_theme_ref_list(
    value: object,
    *,
    context: str,
    theme_ids: set[str],
    require_non_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    refs = tuple(_require_text(item, f"{context} item") for item in value)
    if require_non_empty and not refs:
        raise ValueError(f"{context} must be non-empty")
    for ref in refs:
        if ref not in theme_ids:
            raise ValueError(f"{context} references unknown theme_id: {ref}")
    return refs


def _parse_runtime_index_refs(value: object, *, context: str) -> tuple[RuntimeIndexRef, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.runtime_index_refs must be a non-empty list")
    refs: list[RuntimeIndexRef] = []
    seen_codes: set[tuple[str, str]] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"{context}.runtime_index_refs {index}")
        market = _require_text(item.get("market"), f"{context}.runtime_index_refs {index}.market").upper()
        code = _require_text(item.get("index_code"), f"{context}.runtime_index_refs {index}.index_code")
        key = (market, code)
        if key in seen_codes:
            raise ValueError(f"{context}.runtime_index_refs contains duplicate index_code: {market}/{code}")
        seen_codes.add(key)
        refs.append(
            RuntimeIndexRef(
                market=market,
                index_code=code,
                index_name=_require_text(item.get("index_name"), f"{context}.runtime_index_refs {index}.index_name"),
                source=_require_text(item.get("source"), f"{context}.runtime_index_refs {index}.source"),
            )
        )
    return tuple(refs)


def _parse_legacy_sector_refs(value: object, *, context: str) -> tuple[LegacySectorRef, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.legacy_sector_refs must be a non-empty list")
    refs: list[LegacySectorRef] = []
    seen_codes: set[tuple[str, str]] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"{context}.legacy_sector_refs {index}")
        source = _require_text(item.get("source"), f"{context}.legacy_sector_refs {index}.source")
        code = _require_text(item.get("code"), f"{context}.legacy_sector_refs {index}.code")
        key = (source, code)
        if key in seen_codes:
            raise ValueError(f"{context}.legacy_sector_refs contains duplicate code: {source}/{code}")
        seen_codes.add(key)
        refs.append(
            LegacySectorRef(
                source=source,
                regime=_require_text(item.get("regime"), f"{context}.legacy_sector_refs {index}.regime"),
                code=code,
                name=_require_text(item.get("name"), f"{context}.legacy_sector_refs {index}.name"),
            )
        )
    return tuple(refs)


def _parse_execution_scoring_policy(value: object, *, context: str) -> ExecutionScoringPolicy:
    item = _require_mapping(value, f"{context}.scoring_policy")
    score_role = _require_text(item.get("score_role"), f"{context}.scoring_policy.score_role")
    if score_role not in EXECUTION_SCORE_ROLES:
        raise ValueError(f"{context}.scoring_policy.score_role has invalid value: {score_role}")
    return ExecutionScoringPolicy(
        score_role=score_role,
        action_policy=_require_text(item.get("action_policy"), f"{context}.scoring_policy.action_policy"),
    )


def _parse_execution_flow_join(value: object, *, context: str) -> ExecutionFlowJoin | None:
    if value is None:
        return None
    item = _require_mapping(value, f"{context}.flow_join")
    return ExecutionFlowJoin(
        sector_code=_require_text(item.get("sector_code"), f"{context}.flow_join.sector_code"),
        sector_name=_require_text(item.get("sector_name"), f"{context}.flow_join.sector_name"),
    )


def _parse_execution_etfs(value: object, *, context: str) -> tuple[ExecutionRepresentativeEtf, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context}.representative_etfs must be a list")
    refs: list[ExecutionRepresentativeEtf] = []
    seen_codes: set[str] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"{context}.representative_etfs {index}")
        code = _require_text(item.get("code"), f"{context}.representative_etfs {index}.code")
        if code in seen_codes:
            raise ValueError(f"{context}.representative_etfs contains duplicate code: {code}")
        seen_codes.add(code)
        refs.append(
            ExecutionRepresentativeEtf(
                code=code,
                name=_require_text(item.get("name"), f"{context}.representative_etfs {index}.name"),
            )
        )
    return tuple(refs)


def _parse_execution_mappings(
    value: object,
    *,
    classification_axes: ClassificationAxes,
    themes: tuple[ThemeTaxonomyEntry, ...],
) -> tuple[ExecutionMapping, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("execution_mappings must be a list")
    base_refs = _collect_tag_refs(classification_axes.base_industries)
    cross_refs = _collect_tag_refs(classification_axes.cross_themes)
    theme_ids = {theme.theme_id for theme in themes}
    mappings: list[ExecutionMapping] = []
    seen_mapping_ids: set[str] = set()
    seen_runtime_refs: set[tuple[str, str]] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"execution_mappings {index}")
        mapping_id = _require_text(item.get("mapping_id"), f"execution_mappings {index}.mapping_id")
        if mapping_id in seen_mapping_ids:
            raise ValueError(f"execution_mappings contains duplicate mapping_id: {mapping_id}")
        seen_mapping_ids.add(mapping_id)
        runtime_role = _require_text(item.get("runtime_role"), f"execution_mappings {mapping_id}.runtime_role")
        if runtime_role not in EXECUTION_MAPPING_ROLES:
            raise ValueError(f"execution_mappings {mapping_id}.runtime_role has invalid value: {runtime_role}")
        runtime_refs = _parse_runtime_index_refs(item.get("runtime_index_refs"), context=f"execution_mappings {mapping_id}")
        for runtime_ref in runtime_refs:
            key = (runtime_ref.market, runtime_ref.index_code)
            if key in seen_runtime_refs:
                raise ValueError(f"execution_mappings contains duplicate runtime index ref: {runtime_ref.market}/{runtime_ref.index_code}")
            seen_runtime_refs.add(key)
        mappings.append(
            ExecutionMapping(
                mapping_id=mapping_id,
                runtime_role=runtime_role,
                label=_require_text(item.get("label"), f"execution_mappings {mapping_id}.label"),
                base_tag_refs=_parse_tag_ref_list(
                    item.get("base_tag_refs"),
                    context=f"execution_mappings {mapping_id}.base_tag_refs",
                    valid_refs=base_refs,
                    require_non_empty=runtime_role == "legacy_sector_bridge",
                ),
                cross_tag_refs=_parse_tag_ref_list(
                    item.get("cross_tag_refs"),
                    context=f"execution_mappings {mapping_id}.cross_tag_refs",
                    valid_refs=cross_refs,
                    require_non_empty=False,
                ),
                theme_refs=_parse_theme_ref_list(
                    item.get("theme_refs"),
                    context=f"execution_mappings {mapping_id}.theme_refs",
                    theme_ids=theme_ids,
                    require_non_empty=False,
                ),
                runtime_index_refs=runtime_refs,
                legacy_sector_refs=_parse_legacy_sector_refs(
                    item.get("legacy_sector_refs"),
                    context=f"execution_mappings {mapping_id}",
                ),
                scoring_policy=_parse_execution_scoring_policy(
                    item.get("scoring_policy"),
                    context=f"execution_mappings {mapping_id}",
                ),
                flow_join=_parse_execution_flow_join(item.get("flow_join"), context=f"execution_mappings {mapping_id}"),
                representative_etfs=_parse_execution_etfs(
                    item.get("representative_etfs"),
                    context=f"execution_mappings {mapping_id}",
                ),
                export_series_alias=_coerce_text(item.get("export_series_alias")),
            )
        )
    return tuple(mappings)


def _parse_theme_mappings(
    value: object,
    *,
    themes: tuple[ThemeTaxonomyEntry, ...],
    classification_axes: ClassificationAxes,
    require_mappings: bool,
) -> tuple[ThemeMapping, ...]:
    if value is None:
        if require_mappings:
            raise ValueError("theme_mappings is required")
        return ()
    if not isinstance(value, list):
        raise ValueError("theme_mappings must be a list")
    theme_ids = {theme.theme_id for theme in themes}
    base_refs = _collect_tag_refs(classification_axes.base_industries)
    cross_refs = _collect_tag_refs(classification_axes.cross_themes)
    mappings: list[ThemeMapping] = []
    seen_theme_ids: set[str] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"theme_mappings {index}")
        theme_id = _require_text(item.get("theme_id"), f"theme_mappings {index}.theme_id")
        if theme_id not in theme_ids:
            raise ValueError(f"theme_mappings {index}.theme_id references unknown theme_id: {theme_id}")
        if theme_id in seen_theme_ids:
            raise ValueError(f"theme_mappings contains duplicate theme_id: {theme_id}")
        seen_theme_ids.add(theme_id)
        mappings.append(
            ThemeMapping(
                theme_id=theme_id,
                base_tag_refs=_parse_tag_ref_list(
                    item.get("base_tag_refs"),
                    context=f"theme_mappings {theme_id}.base_tag_refs",
                    valid_refs=base_refs,
                    require_non_empty=True,
                ),
                cross_tag_refs=_parse_tag_ref_list(
                    item.get("cross_tag_refs"),
                    context=f"theme_mappings {theme_id}.cross_tag_refs",
                    valid_refs=cross_refs,
                    require_non_empty=False,
                ),
                mapping_rule=_require_text(item.get("mapping_rule"), f"theme_mappings {theme_id}.mapping_rule"),
            )
        )
    missing_theme_ids = sorted(theme_ids - seen_theme_ids)
    if missing_theme_ids:
        raise ValueError(f"theme_mappings missing theme_id: {missing_theme_ids[0]}")
    extra_theme_ids = sorted(seen_theme_ids - theme_ids)
    if extra_theme_ids:
        raise ValueError(f"theme_mappings contains unknown theme_id: {extra_theme_ids[0]}")
    return tuple(mappings)


def _parse_themes(value: object) -> tuple[ThemeTaxonomyEntry, ...]:
    if not isinstance(value, list):
        raise ValueError("theme taxonomy config must contain a themes list")
    themes: list[ThemeTaxonomyEntry] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(value):
        item = _require_mapping(raw_item, f"theme {index}")
        theme_id = _require_text(item.get("theme_id"), f"theme {index}.theme_id")
        if theme_id in seen_ids:
            raise ValueError(f"duplicate theme_id: {theme_id}")
        seen_ids.add(theme_id)
        themes.append(
            ThemeTaxonomyEntry(
                theme_id=theme_id,
                name=_require_text(item.get("name"), f"theme {theme_id}.name"),
                aliases=_parse_aliases(item.get("aliases")),
                primary_authority=_parse_primary_authority(item.get("primary_authority"), theme_id=theme_id),
                products=_parse_products(item.get("products"), theme_id=theme_id),
                secondary_anchors=_parse_secondary_anchors(item.get("secondary_anchors"), theme_id=theme_id),
            )
        )
    return tuple(themes)


def load_theme_taxonomy_config(path: Path = THEME_TAXONOMY_CONFIG_PATH) -> ThemeTaxonomy:
    """Load and validate KR theme taxonomy definitions."""
    payload = _load_yaml(path)
    version_raw = payload.get("taxonomy_version", 1)
    try:
        taxonomy_version = int(version_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("taxonomy_version must be an integer") from exc
    if taxonomy_version < 1:
        raise ValueError("taxonomy_version must be positive")
    classification_axes = _parse_classification_axes(payload.get("classification_axes"))
    themes = _parse_themes(payload.get("themes"))
    return ThemeTaxonomy(
        market=_coerce_text(payload.get("market")) or "KR",
        taxonomy_version=taxonomy_version,
        authority_priority=_parse_authority_priority(payload),
        verification=_parse_verification(payload.get("verification")),
        classification_axes=classification_axes,
        execution_mappings=_parse_execution_mappings(
            payload.get("execution_mappings"),
            classification_axes=classification_axes,
            themes=themes,
        ),
        theme_mappings=_parse_theme_mappings(
            payload.get("theme_mappings"),
            themes=themes,
            classification_axes=classification_axes,
            require_mappings=taxonomy_version >= 2,
        ),
        themes=themes,
    )


def build_theme_authority_basis_map(taxonomy: ThemeTaxonomy) -> dict[str, str]:
    """Return the effective authority basis for each theme.

    Secondary anchors are intentionally ignored here. They explain a theme, but
    they cannot override the product/theme-index primary authority selected by
    the taxonomy.
    """
    return {
        theme.theme_id: theme.primary_authority.basis_type
        for theme in taxonomy.themes
    }
