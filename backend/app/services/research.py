"""
backend/app/services/research.py
--------------------------------
Service that dynamically reads research reports, CSV tables, metrics,
and JSON artifacts at runtime without hardcoded values.
Powering all /api/research/* endpoints.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any
import yaml

from ..config import settings

logger = logging.getLogger("wtc.research")


def _read_csv(filepath: Path) -> list[dict[str, Any]]:
    """Helper to read CSV into list of string/float dicts."""
    if not filepath.exists():
        logger.warning("CSV file not found: %s", filepath)
        return []
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {}
            for k, v in row.items():
                if v is None:
                    parsed_row[k] = None
                    continue
                v_str = str(v).strip()
                # Attempt float parsing
                try:
                    if "." in v_str or "e" in v_str.lower():
                        parsed_row[k] = float(v_str)
                    elif v_str.isdigit() or (v_str.startswith("-") and v_str[1:].isdigit()):
                        parsed_row[k] = int(v_str)
                    else:
                        parsed_row[k] = v_str
                except ValueError:
                    parsed_row[k] = v_str
            rows.append(parsed_row)
    return rows


def _read_json(filepath: Path) -> dict[str, Any]:
    """Helper to read JSON file."""
    if not filepath.exists():
        logger.warning("JSON file not found: %s", filepath)
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(filepath: Path) -> str:
    """Helper to read markdown/text file."""
    if not filepath.exists():
        logger.warning("Text file not found: %s", filepath)
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_research_overview() -> dict[str, Any]:
    """Overview table and summary metrics from reports/model_comparison.csv."""
    comp_csv = settings.REPO_ROOT_PATH / "reports" / "model_comparison.csv"
    comp_rows = _read_csv(comp_csv)
    comp_md = _read_text(settings.REPO_ROOT_PATH / "reports" / "model_comparison.md")

    return {
        "title": "Welcome to Civilization — Research & Modeling Benchmark",
        "description": "Multi-label toxicity moderation benchmark on the Jigsaw Toxic Comment dataset.",
        "model_comparison_table": comp_rows,
        "comparison_markdown": comp_md,
        "detoxify_note": "Detoxify was trained by its authors (Unitary AI) on the raw Jigsaw 2018 competition train set prior to near-duplicate deduplication, which partially overlaps this project's test split. Consequently, Detoxify's metrics serve as an external pretrained upper-bound baseline rather than a direct fair in-domain comparison.",
    }


def get_eda_data() -> dict[str, Any]:
    """EDA statistics, findings, and figure gallery."""
    findings_json = settings.REPO_ROOT_PATH / "reports" / "eda" / "eda_findings.json"
    conclusions_md = settings.REPO_ROOT_PATH / "reports" / "eda" / "eda_conclusions.md"

    # List all figure files in reports/eda/figures
    figures = []
    fig_dir = settings.EDA_FIGURES_PATH
    if fig_dir.exists():
        for p in sorted(fig_dir.glob("*.png")):
            figures.append({
                "filename": p.name,
                "url": f"/static/eda/{p.name}",
                "name": p.stem.replace("_", " ").title(),
            })

    return {
        "findings": _read_json(findings_json),
        "conclusions_markdown": _read_text(conclusions_md),
        "figures": figures,
    }


def get_splits_data() -> dict[str, Any]:
    """Split manifest, near-duplicate group distribution, and verification stats."""
    manifest_file = settings.REPO_ROOT_PATH / "data" / "splits" / "split_manifest.json"
    manifest = _read_json(manifest_file)
    data_dict_md = _read_text(settings.REPO_ROOT_PATH / "DATA_DICTIONARY.md")

    return {
        "manifest": manifest,
        "data_dictionary_markdown": data_dict_md,
        "methodology": {
            "strategy": "Multilabel-stratified 80/10/10 split on near-duplicate groups (dup_group_v2)",
            "leakage_prevention": "MinHash LSH Jaccard similarity >= 0.8 on character 5-grams ensures 0% duplicate leakage across splits.",
            "casing": "Preserved without lowercasing in processed text to retain uppercase emotional intensity features.",
        },
    }


def get_models_config_data() -> dict[str, Any]:
    """Architecture specifications, hyperparameter tuning, and metadata for all models."""
    models_yaml_file = settings.REPO_ROOT_PATH / "configs" / "models.yaml"
    models_cfg = {}
    if models_yaml_file.exists():
        with open(models_yaml_file, "r", encoding="utf-8") as f:
            models_cfg = yaml.safe_load(f)

    # Collect meta.json for each model
    metas = {}
    for m in ["1_tfidf_lr", "2_bigru_attn", "2b_bigru_attn_v2", "3_distilroberta", "detoxify"]:
        meta_file = settings.REPO_ROOT_PATH / "models" / m / "meta.json"
        metas[m] = _read_json(meta_file)

    # Blend and tuning details
    blend_summary = _read_json(settings.REPO_ROOT_PATH / "reports" / "3_distilroberta" / "ensemble_blend_summary.json")
    tuning_csv = settings.REPO_ROOT_PATH / "reports" / "3_distilroberta" / "tables" / "tuning_trials.csv"

    return {
        "config_yaml": models_cfg,
        "metadata_by_model": metas,
        "blend_summary": blend_summary,
        "tuning_trials": _read_csv(tuning_csv),
    }


def get_results_data() -> dict[str, Any]:
    """Test split per-label metrics, bootstrap confidence intervals, and visualization links."""
    per_label_csv = settings.REPO_ROOT_PATH / "reports" / "model_comparison_per_label.csv"
    overall_csv = settings.REPO_ROOT_PATH / "reports" / "model_comparison.csv"

    # Per-model metrics.json
    metrics_by_model = {}
    for m in ["1_tfidf_lr", "2_bigru_attn", "2b_bigru_attn_v2", "3_distilroberta", "detoxify"]:
        m_file = settings.REPO_ROOT_PATH / "reports" / m / "metrics.json"
        metrics_by_model[m] = _read_json(m_file)

    # Visualizations per model
    vis_by_model = {}
    vis_dir = settings.VISUALIZATIONS_PATH
    if vis_dir.exists():
        for m_dir in vis_dir.iterdir():
            if m_dir.is_dir():
                m_name = m_dir.name
                vis_by_model[m_name] = [
                    {"filename": f.name, "url": f"/static/vis/{m_name}/{f.name}", "title": f.stem.replace("_", " ").title()}
                    for f in sorted(m_dir.glob("*.png"))
                ]

    return {
        "overall_metrics": _read_csv(overall_csv),
        "per_label_metrics": _read_csv(per_label_csv),
        "detailed_metrics": metrics_by_model,
        "visualizations": vis_by_model,
    }


def get_comparisons_data() -> dict[str, Any]:
    """Paired bootstrap significance testing tables vs Model 2b and Detoxify."""
    vs_2b_csv = settings.REPO_ROOT_PATH / "reports" / "3_distilroberta" / "tables" / "paired_bootstrap_vs_2b_bigru_attn_v2.csv"
    vs_detox_csv = settings.REPO_ROOT_PATH / "reports" / "3_distilroberta" / "tables" / "paired_bootstrap_vs_detoxify.csv"

    return {
        "paired_bootstrap_vs_model2b": _read_csv(vs_2b_csv),
        "paired_bootstrap_vs_detoxify": _read_csv(vs_detox_csv),
    }


def get_imbalance_data() -> dict[str, Any]:
    """Dataset imbalance statistics and their direct translation into asymmetric moderation policy."""
    manifest_file = settings.REPO_ROOT_PATH / "data" / "splits" / "split_manifest.json"
    manifest = _read_json(manifest_file)
    policy_file = settings.POLICY_PATH
    policy_cfg = {}
    if policy_file.exists():
        with open(policy_file, "r", encoding="utf-8") as f:
            policy_cfg = yaml.safe_load(f)

    return {
        "split_manifest": manifest,
        "policy_configuration": policy_cfg,
        "asymmetric_rationale": (
            "Severe class imbalance (threat at 0.30% positive, identity_hate at 0.88%, severe_toxic at 0.99%) "
            "causes models trained with class weighting to output inflated probabilities with F1 scores bounded at ~0.50-0.60. "
            "To prevent rampant false deletions, the moderation policy enforces: (1) piecewise linear threshold normalization, "
            "(2) high-precision calibrated remove cuts derived from validation data, (3) multi-model corroboration for automated deletion, "
            "and (4) a lower-bound watch zone [0.30, 0.50) directing questionable comments to human moderator review."
        ),
    }


def get_limitations_data() -> dict[str, Any]:
    """Model limitations and evaluation constraints."""
    m3_report = _read_text(settings.REPO_ROOT_PATH / "reports" / "3_distilroberta" / "model_report.md")
    detox_report = _read_text(settings.REPO_ROOT_PATH / "reports" / "detoxify" / "model_report.md")

    # Extract limitations sections if present
    def extract_lims(md_text: str) -> str:
        if "## 11. Limitations" in md_text:
            return md_text.split("## 11. Limitations")[-1].strip()
        elif "## Limitations" in md_text:
            return md_text.split("## Limitations")[-1].strip()
        return md_text

    return {
        "distilroberta_limitations_markdown": extract_lims(m3_report),
        "detoxify_limitations_markdown": extract_lims(detox_report),
        "detoxify_footnote": (
            "Detoxify was trained by Unitary AI on the original 2018 Kaggle Jigsaw competition training set. "
            "Because that dataset predates this repository's strict near-duplicate deduplication (MinHash LSH Jaccard 0.8), "
            "its evaluation split partially overlaps our curated test split. Hence Detoxify represents an external pretrained "
            "reference point, not a fair zero-leakage comparison."
        ),
    }
