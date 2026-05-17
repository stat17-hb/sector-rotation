from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pytest

import src.data_sources.theme_taxonomy as taxonomy


CURRENT_THEME_IDS = {
    "power_infra",
    "shipbuilding",
    "nuclear",
    "robotics",
    "defense",
    "aerospace_uam",
    "beauty_cosmetics",
}
EXPANDED_THEME_IDS = {
    "semiconductors",
    "steel",
    "banks",
    "securities",
    "insurance",
    "secondary_battery",
    "automobiles",
    "healthcare",
    "energy_chemicals",
    "consumer_discretionary",
    "consumer_staples",
    "information_technology",
    "climate_transition",
}
REQUIRED_BASE_AXES = {
    "financials",
    "industrials",
    "information_technology",
    "materials_energy",
    "healthcare",
    "consumer",
    "mobility",
    "communication_services",
    "infrastructure_utilities",
    "holdings_conglomerates",
    "real_estate_reits",
}
REQUIRED_CROSS_THEME_AXES = {
    "ai_infrastructure",
    "energy_transition",
    "defense_space",
    "digital_transformation",
    "k_consumption",
    "capital_market_reform",
    "ai_applications",
}
REQUIRED_TAG_REFS = {
    "financials/banks",
    "industrials/power_equipment",
    "information_technology/software_platforms",
    "consumer/games_content",
    "communication_services/telecom_services",
    "infrastructure_utilities/utilities",
    "holdings_conglomerates/holding_companies",
    "real_estate_reits/reits",
    "ai_infrastructure/ai_power_cooling",
    "energy_transition/climate_transition_solutions",
    "defense_space/k_defense_exports",
    "defense_space/naval_defense_mro",
    "digital_transformation/network_infrastructure",
    "k_consumption/k_beauty",
    "k_consumption/k_content_entertainment",
    "k_consumption/k_food",
    "k_consumption/k_tourism_duty_free",
    "capital_market_reform/value_up_governance",
    "capital_market_reform/holding_company_discount",
    "ai_applications/ai_software_services",
}


def _write_minimal_config(path: Path, *, extra: str = "", priority: str | None = None) -> None:
    authority_priority = priority or """
authority_priority:
  - "ETF_PRODUCT_THEME_INDEX"
  - "ETF_COMPARISON_INDEX"
  - "ASSET_MANAGER_CATEGORY"
  - "OFFICIAL_INDEX_OR_INDUSTRY_ANCHOR"
"""
    path.write_text(
        f"""
market: "KR"
taxonomy_version: 1
verification:
  last_verified_at: "2026-05-17"
  verification_status: "verified"
  stale_after_days: 180
{authority_priority}
themes:
  - theme_id: "shipbuilding"
    name: "조선"
    primary_authority:
      provider: "FnGuide"
      basis_type: "ETF_PRODUCT_THEME_INDEX"
      index_name: "FnGuide 조선해운 지수"
      source_url: "https://wcomp.fnguide.com/etp/etfSnapshot?cmp_cd=441540"
      source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"
      supporting_urls:
        - role: "AGGREGATOR_REFERENCE"
          url: "https://www.k-etf.com/etf/441540"
          label: "K-ETF"
    products:
      - code: "0141S0"
        name: "SOL 조선기자재"
        product_type: "ETF"
        benchmark_index: "FnGuide 조선기자재 지수(PR)"
        source_url: "https://wcomp.fnguide.com/etp/etfSnapshot?cmp_cd=0141S0"
        source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"
        supporting_urls:
          - role: "AGGREGATOR_REFERENCE"
            url: "https://www.k-etf.com/etf/0141S0"
            label: "K-ETF"
    secondary_anchors:
      - provider: "WICS/FICS"
        label: "조선 industry anchor"
        evidence_note: "Industry anchor only."
{extra}
""",
        encoding="utf-8",
    )


def _write_default_config_copy(path: Path) -> None:
    path.write_text(Path("config/theme_taxonomy.yml").read_text(encoding="utf-8"), encoding="utf-8")


def _all_tag_refs(loaded: taxonomy.ThemeTaxonomy) -> set[str]:
    return {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in (
            *loaded.classification_axes.base_industries,
            *loaded.classification_axes.cross_themes,
        )
        for child in axis.children
    }


def test_load_theme_taxonomy_config_preserves_order_and_string_codes():
    loaded = taxonomy.load_theme_taxonomy_config()

    assert loaded.market == "KR"
    assert loaded.authority_priority[-1] == taxonomy.OFFICIAL_ANCHOR_BASIS_TYPE
    assert [theme.theme_id for theme in loaded.themes[:2]] == ["power_infra", "shipbuilding"]
    shipbuilding = next(theme for theme in loaded.themes if theme.theme_id == "shipbuilding")
    assert shipbuilding.products[0].code == "441540"
    assert isinstance(shipbuilding.products[0].code, str)
    assert shipbuilding.primary_authority.source_role == "OFFICIAL_INDEX_PROVIDER_PAGE"
    assert shipbuilding.primary_authority.supporting_urls[0].role == "AGGREGATOR_REFERENCE"


def test_theme_taxonomy_preserves_current_theme_lens_ids_and_adds_new_theme():
    loaded = taxonomy.load_theme_taxonomy_config()
    theme_ids = {theme.theme_id for theme in loaded.themes}

    assert CURRENT_THEME_IDS.issubset(theme_ids)
    assert EXPANDED_THEME_IDS.issubset(theme_ids)


def test_theme_taxonomy_loads_dual_axis_classification_layer():
    loaded = taxonomy.load_theme_taxonomy_config()
    base_axes = loaded.classification_axes.base_industries
    cross_axes = loaded.classification_axes.cross_themes

    assert {axis.axis_id for axis in base_axes} == REQUIRED_BASE_AXES
    assert {axis.axis_id for axis in cross_axes} == REQUIRED_CROSS_THEME_AXES
    assert {axis.assignment for axis in base_axes} == {"single"}
    assert {axis.assignment for axis in cross_axes} == {"multiple"}

    tag_refs = {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in (*base_axes, *cross_axes)
        for child in axis.children
    }
    assert REQUIRED_TAG_REFS.issubset(tag_refs)

    k_beauty = next(
        child
        for axis in cross_axes
        if axis.axis_id == "k_consumption"
        for child in axis.children
        if child.tag_id == "k_beauty"
    )
    assert "화장품/K-뷰티" in k_beauty.aliases
    assert k_beauty.history[0].version == 1
    assert k_beauty.history[0].effective_from == "2026-05-17"


def test_theme_taxonomy_exposes_verification_metadata_and_staleness():
    loaded = taxonomy.load_theme_taxonomy_config()

    assert loaded.verification.last_verified_at == date(2026, 5, 17)
    assert loaded.verification.verification_status == "verified"
    assert loaded.verification.stale_after_days == 180
    assert not loaded.verification.is_stale(date(2026, 11, 13))
    assert loaded.verification.is_stale("2026-11-14")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('verification_status: "verified"', 'verification_status: "draft"', "verification_status"),
        ('last_verified_at: "2026-05-17"', 'last_verified_at: "20260517"', "ISO YYYY-MM-DD"),
        ("stale_after_days: 180", "stale_after_days: 0", "positive integer"),
    ],
)
def test_theme_taxonomy_rejects_invalid_verification_metadata(tmp_path, old, new, match):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_default_config_copy(config_path)
    config_path.write_text(config_path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_maps_all_themes_to_classification_axes():
    loaded = taxonomy.load_theme_taxonomy_config()
    theme_ids = {theme.theme_id for theme in loaded.themes}
    mapping_ids = {mapping.theme_id for mapping in loaded.theme_mappings}
    base_refs = {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in loaded.classification_axes.base_industries
        for child in axis.children
    }
    cross_refs = {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in loaded.classification_axes.cross_themes
        for child in axis.children
    }

    assert mapping_ids == theme_ids
    assert len(loaded.theme_mappings) == 20
    assert all(mapping.base_tag_refs for mapping in loaded.theme_mappings)
    assert all(mapping.cross_tag_refs is not None for mapping in loaded.theme_mappings)
    assert any(mapping.cross_tag_refs == () for mapping in loaded.theme_mappings)
    for mapping in loaded.theme_mappings:
        assert set(mapping.base_tag_refs).issubset(base_refs)
        assert set(mapping.cross_tag_refs).issubset(cross_refs)
        assert mapping.mapping_rule


def test_theme_taxonomy_loads_execution_mappings_without_sector_map_dependency():
    loaded = taxonomy.load_theme_taxonomy_config()
    mappings = {mapping.runtime_index_refs[0].index_code: mapping for mapping in loaded.execution_mappings}

    assert set(mappings) == {"1155", "1168", "5042", "5044", "5045", "1157", "1165", "1170", "5046", "5048", "5049"}
    assert mappings["5044"].label == "KRX 반도체"
    assert mappings["5044"].base_tag_refs == ("information_technology/semiconductors",)
    assert mappings["5044"].cross_tag_refs == ("ai_infrastructure/ai_semiconductors",)
    assert mappings["5044"].theme_refs == ("semiconductors",)
    assert mappings["5044"].scoring_policy.action_policy == "project_existing_signal"


def test_theme_taxonomy_rejects_unknown_execution_taxonomy_ref(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_default_config_copy(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        '- "information_technology/semiconductors"',
        '- "information_technology/unknown"',
        1,
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown tag ref"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_v2_requires_theme_mappings(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_default_config_copy(config_path)
    raw = config_path.read_text(encoding="utf-8")
    without_mappings = raw[: raw.index("\ntheme_mappings:")] + raw[raw.index("\nthemes:") :]
    config_path.write_text(without_mappings, encoding="utf-8")

    with pytest.raises(ValueError, match="theme_mappings is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('theme_id: "power_infra"', 'theme_id: "unknown_theme"', "unknown theme_id"),
        ('base_tag_refs:\n      - "industrials/power_equipment"', "base_tag_refs: []", "base_tag_refs must be non-empty"),
        ('- "industrials/power_equipment"', '- "industrials/unknown_tag"', "unknown tag ref"),
        ('- "ai_infrastructure/ai_power_cooling"', '- "ai_infrastructure/unknown_tag"', "unknown tag ref"),
        ('theme_id: "power_infra"', 'theme_id: "shipbuilding"', "duplicate theme_id"),
        (
            '    mapping_rule: "Apply the AI power/cooling overlay only when data-center power, cooling, transformer, cable, or grid CAPEX exposure is verified."\n',
            "",
            "mapping_rule is required",
        ),
    ],
)
def test_theme_taxonomy_rejects_invalid_theme_mappings(tmp_path, old, new, match):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_default_config_copy(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(old, new, 1)
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_preserves_feedback_critical_overlay_semantics():
    loaded = taxonomy.load_theme_taxonomy_config()
    tags = {
        f"{axis.axis_id}/{child.tag_id}": child
        for axis in loaded.classification_axes.cross_themes
        for child in axis.children
    }

    assert "digital_transformation/network_5g" not in _all_tag_refs(loaded)
    network = tags["digital_transformation/network_infrastructure"]
    assert {"5G", "6G", "통신장비", "데이터센터 네트워크"}.issubset(set(network.aliases))
    assert "데이터센터 네트워크" in network.inclusion_rule

    value_up = tags["capital_market_reform/value_up_governance"]
    assert {"밸류업", "주주환원", "저PBR", "지배구조 개선"}.issubset(set(value_up.aliases))
    for term in ("주주환원", "저PBR", "지배구조"):
        assert term in value_up.inclusion_rule


def test_theme_taxonomy_requires_duplicate_free_theme_ids(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
  - theme_id: "shipbuilding"
    name: "Duplicate"
    primary_authority:
      provider: "FnGuide"
      basis_type: "ETF_PRODUCT_THEME_INDEX"
      source_url: "https://example.com"
      source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"
    products:
      - code: "999999"
        name: "Duplicate ETF"
        product_type: "ETF"
        source_url: "https://example.com"
        source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"
""",
    )

    with pytest.raises(ValueError, match="duplicate theme_id"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_missing_theme_identity(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace('theme_id: "shipbuilding"', 'theme_id: ""')
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="theme 0.theme_id is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_missing_source_evidence(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        '      source_url: "https://wcomp.fnguide.com/etp/etfSnapshot?cmp_cd=441540"\n',
        "",
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="primary_authority requires source_url or evidence_note"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_invalid_basis_type(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        'basis_type: "ETF_PRODUCT_THEME_INDEX"',
        'basis_type: "BROKER_THEME_MAP"',
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid primary_authority.basis_type"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_invalid_source_role(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        'source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"',
        'source_role: "BLOG_POST"',
        1,
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid source_role"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_aggregator_as_primary_source_role(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        'source_role: "OFFICIAL_INDEX_PROVIDER_PAGE"',
        'source_role: "AGGREGATOR_REFERENCE"',
        1,
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot use AGGREGATOR_REFERENCE as primary"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_ketf_primary_authority_url(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        'source_url: "https://wcomp.fnguide.com/etp/etfSnapshot?cmp_cd=441540"',
        'source_url: "https://www.k-etf.com/etf/441540"',
        1,
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot use k-etf.com as primary source_url"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_ketf_primary_product_url_even_with_supporting_urls(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        '        source_url: "https://wcomp.fnguide.com/etp/etfSnapshot?cmp_cd=0141S0"',
        '        source_url: "https://www.k-etf.com/etf/0141S0"',
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot use k-etf.com as primary source_url"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_malformed_supporting_urls(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace(
        '        - role: "AGGREGATOR_REFERENCE"\n          url: "https://www.k-etf.com/etf/441540"\n          label: "K-ETF"\n',
        '        - role: "AGGREGATOR_REFERENCE"\n          label: "K-ETF"\n',
    )
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="supporting_urls 0.url is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_official_anchor_priority_over_product_authority(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        priority="""
authority_priority:
  - "OFFICIAL_INDEX_OR_INDUSTRY_ANCHOR"
  - "ETF_PRODUCT_THEME_INDEX"
  - "ETF_COMPARISON_INDEX"
  - "ASSET_MANAGER_CATEGORY"
""",
    )

    with pytest.raises(ValueError, match="official industry anchors cannot outrank"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_duplicate_classification_axis_ids(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  base_industries:
    - axis_id: "financials"
      name: "금융"
      assignment: "single"
      children:
        - tag_id: "banks"
          name: "은행"
          inclusion_rule: "은행업 중심"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
    - axis_id: "financials"
      name: "금융 duplicate"
      assignment: "single"
      children:
        - tag_id: "insurance"
          name: "보험"
          inclusion_rule: "보험업 중심"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
""",
    )

    with pytest.raises(ValueError, match="duplicate axis_id"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_duplicate_classification_tag_ids_within_axis(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  cross_themes:
    - axis_id: "energy_transition"
      name: "에너지전환"
      assignment: "multiple"
      children:
        - tag_id: "nuclear"
          name: "원자력"
          inclusion_rule: "원전 노출"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
        - tag_id: "nuclear"
          name: "원전 duplicate"
          inclusion_rule: "원전 노출"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
""",
    )

    with pytest.raises(ValueError, match="duplicate tag_id"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_allows_same_tag_id_across_different_axes(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  base_industries:
    - axis_id: "information_technology"
      name: "정보기술"
      assignment: "single"
      children:
        - tag_id: "semiconductors"
          name: "반도체"
          inclusion_rule: "반도체 산업"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
  cross_themes:
    - axis_id: "ai_infrastructure"
      name: "AI 인프라"
      assignment: "multiple"
      children:
        - tag_id: "semiconductors"
          name: "AI반도체"
          inclusion_rule: "AI 반도체 테마"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
""",
    )

    loaded = taxonomy.load_theme_taxonomy_config(config_path)
    refs = {
        f"{axis.axis_id}/{child.tag_id}"
        for axis in (
            *loaded.classification_axes.base_industries,
            *loaded.classification_axes.cross_themes,
        )
        for child in axis.children
    }
    assert refs == {"information_technology/semiconductors", "ai_infrastructure/semiconductors"}


def test_theme_taxonomy_rejects_invalid_classification_assignment(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  base_industries:
    - axis_id: "financials"
      name: "금융"
      assignment: "multiple"
      children:
        - tag_id: "banks"
          name: "은행"
          inclusion_rule: "은행업 중심"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
""",
    )

    with pytest.raises(ValueError, match="assignment must be single"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_classification_tag_missing_metadata(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  base_industries:
    - axis_id: "financials"
      name: "금융"
      assignment: "single"
      children:
        - tag_id: "banks"
          name: "은행"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
              note: "seed"
""",
    )

    with pytest.raises(ValueError, match="inclusion_rule is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_classification_tag_missing_history_fields(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(
        config_path,
        extra="""
classification_axes:
  base_industries:
    - axis_id: "financials"
      name: "금융"
      assignment: "single"
      children:
        - tag_id: "banks"
          name: "은행"
          inclusion_rule: "은행업 중심"
          exposure_rule:
            primary_metric: "매출"
            pure_play: ">=50%"
            relevant: "20-50%"
            satellite: "5-20%"
          history:
            - version: 1
              effective_from: "2026-05-17"
""",
    )

    with pytest.raises(ValueError, match="history 0.note is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_malformed_product_entries(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace('        product_type: "ETF"\n', "")
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="product 0.product_type is required"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_theme_taxonomy_rejects_numeric_product_codes(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    raw = config_path.read_text(encoding="utf-8").replace('code: "0141S0"', "code: 141")
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="code must be a string"):
        taxonomy.load_theme_taxonomy_config(config_path)


def test_secondary_anchors_do_not_override_primary_authority(tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)

    loaded = taxonomy.load_theme_taxonomy_config(config_path)
    basis_map = taxonomy.build_theme_authority_basis_map(loaded)

    assert basis_map == {"shipbuilding": "ETF_PRODUCT_THEME_INDEX"}
    assert loaded.themes[0].secondary_anchors[0].provider == "WICS/FICS"


def test_theme_taxonomy_loader_is_static_metadata_only(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_taxonomy.yml"
    _write_minimal_config(config_path)
    sys.modules.pop("pykrx", None)
    warehouse_module = sys.modules.get("src.data_sources.warehouse")
    sys.modules.pop("src.data_sources.warehouse", None)

    try:
        loaded = taxonomy.load_theme_taxonomy_config(config_path)

        assert [theme.theme_id for theme in loaded.themes] == ["shipbuilding"]
        assert "pykrx" not in sys.modules
        assert "src.data_sources.warehouse" not in sys.modules
    finally:
        if warehouse_module is not None:
            sys.modules["src.data_sources.warehouse"] = warehouse_module
