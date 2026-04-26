#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute six legacy structural metrics from outputs_constrained_4dim/*.xml:
1) well_formed
2) schema_valid
3) content_validity_ratio
4) hallucination_rate
5) panel_compliant
6) multi_character_ratio
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(STAGE2_ROOT / "evaluate_llm"))

from stage_2_Narrative_Capability_under_Controlled_Complex.evaluation_llm.config_4dim import DIMENSIONS, load_element_library

POSITION_TAGS = ("left", "middle", "right")
# Evaluate only core content elements (exclude name/appearance/skin/hair/lips/balloon).
CORE_TYPES = {"setting", "position", "id", "pose", "facing"}
LEGACY_SINGLE_STORIES = {"2", "3", "5", "6", "7"}
LEGACY_MULTI_STORIES = {"1_2_3", "2_3", "5_6", "5_6_7", "5_7"}
GLOBAL_ELEMENTS_FILE = STAGE2_ROOT / "spec_story_elements" / "all_stories_elements.json"


def t(elem: ET.Element | None) -> str:
    return (elem.text or "").strip() if elem is not None else ""


def build_allowed_sets(element_library: dict[str, Any]) -> dict[str, set[str]]:
    allowed = {
        "id": set(),
        "name": set(),
        "appearance": set(),
        "skin": set(),
        "hair": set(),
        "lips": set(),
        "pose": set(element_library.get("poses", [])),
        "setting": set(element_library.get("settings", [])),
        "facing": set(element_library.get("facings", [])),
        "position": set(element_library.get("positions", [])),
        "balloon_status": set(element_library.get("balloon_status", [])),
    }
    for cid, cdata in element_library.get("characters", {}).items():
        allowed["id"].add(str(cid))
        for v in (cdata.get("variants", {}) or {}).values():
            if not isinstance(v, dict):
                continue
            for key in ("name", "appearance", "skin", "hair", "lips"):
                val = (v.get(key) or "").strip()
                if val:
                    allowed[key].add(val)
    return allowed


def load_global_element_library() -> dict[str, Any]:
    if not GLOBAL_ELEMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing global element file: {GLOBAL_ELEMENTS_FILE}")
    return json.loads(GLOBAL_ELEMENTS_FILE.read_text(encoding="utf-8"))


def validate_schema_and_panels(root: ET.Element, schema_mode: str) -> tuple[bool, list[str], int]:
    errors: list[str] = []
    if root.tag != "comic":
        return False, ["Root tag must be <comic>"], 0

    scenes = root.find("scenes")
    if scenes is None:
        return False, ["Missing <scenes>"], 0
    scene = scenes.find("scene")
    if scene is None:
        return False, ["Missing <scene> under <scenes>"], 0

    panels = scene.findall("panel")
    panel_count = len(panels)
    if panel_count == 0:
        return False, ["No <panel> under <scene>"], 0

    for idx, panel in enumerate(panels, start=1):
        positions = [c for c in list(panel) if c.tag in POSITION_TAGS]
        if not positions:
            errors.append(f"Panel {idx}: missing position container")

        if not t(panel.find("setting")):
            errors.append(f"Panel {idx}: missing <setting>")

        for pos in positions:
            for fig in pos.findall("figure"):
                if not t(fig.find("pose")):
                    errors.append(f"Panel {idx}: figure missing <pose>")
                if not t(fig.find("facing")):
                    errors.append(f"Panel {idx}: figure missing <facing>")
                fid = t(fig.find("id"))
                name = t(fig.find("name"))
                app = t(fig.find("appearance"))

                if schema_mode == "strict":
                    # Strict mode: id/name/appearance must be either all empty (non-character) or all present.
                    if fid or name or app:
                        if not (fid and name and app):
                            errors.append(
                                f"Panel {idx}: figure id/name/appearance must be all non-empty"
                            )
                else:
                    # Lenient mode: id and name should appear as a pair.
                    if fid or name:
                        if not (fid and name):
                            errors.append(f"Panel {idx}: figure id/name should appear together")

            if schema_mode == "strict":
                for balloon in pos.findall("balloon"):
                    status = (balloon.get("status") or "").strip()
                    if not status:
                        errors.append(f"Panel {idx}: balloon missing status")
                    if not t(balloon.find("content")):
                        errors.append(f"Panel {idx}: balloon missing <content>")

    return len(errors) == 0, errors, panel_count


def collect_used_elements(root: ET.Element) -> list[tuple[str, str, str]]:
    """
    Return tuples in the shape of (etype, value, location).
    Example location: Scene1_Panel4_left, used to trace hallucination details.
    """
    used: list[tuple[str, str, str]] = []
    for panel_idx, panel in enumerate(root.findall(".//panel"), start=1):
        for child in list(panel):
            if child.tag == "setting":
                v = t(child)
                if v:
                    used.append(("setting", v, f"Scene1_Panel{panel_idx}"))
                continue

            if child.tag not in POSITION_TAGS:
                continue

            used.append(("position", child.tag, f"Scene1_Panel{panel_idx}_{child.tag}"))

            for fig in child.findall("figure"):
                for k in ("id", "name", "appearance", "skin", "hair", "lips", "pose", "facing"):
                    v = t(fig.find(k))
                    if v:
                        used.append((k, v, f"Scene1_Panel{panel_idx}_{child.tag}"))

            for balloon in child.findall("balloon"):
                status = (balloon.get("status") or "").strip()
                if status:
                    used.append(("balloon_status", status, f"Scene1_Panel{panel_idx}_{child.tag}"))
    return used


def evaluate_content(
    root: ET.Element, allowed: dict[str, set[str]], content_types: set[str]
) -> dict[str, Any]:
    used = collect_used_elements(root)
    used = [x for x in used if x[0] in content_types]
    total = len(used)
    if total == 0:
        return {
            "content_validity_ratio": 0.0,
            "hallucination_rate": 0.0,
            "invalid_elements": [],
            "valid_elements": 0,
            "hallucinated_elements": [],
            "total_elements": 0,
        }

    valid = 0
    invalid_elements: list[list[Any]] = []
    hallucinated_elements: list[list[Any]] = []
    for etype, val, loc in used:
        if val in allowed.get(etype, set()):
            valid += 1
        else:
            invalid_elements.append([val, etype, loc])
            hallucinated_elements.append([etype, val, loc])

    hallucinated = total - valid
    return {
        "content_validity_ratio": valid / total,
        "hallucination_rate": hallucinated / total,
        "invalid_elements": invalid_elements,
        "valid_elements": valid,
        "hallucinated_elements": hallucinated_elements,
        "total_elements": total,
    }


def evaluate_character_attr_consistency(
    root: ET.Element, element_library: dict[str, Any]
) -> dict[str, Any]:
    """
    Character attribute consistency check (legacy-compatible behavior):
    for each figure with an id, verify appearance/skin/hair against
    any variant of that character id.
    """
    characters = element_library.get("characters", {}) or {}
    attrs = ("appearance", "skin", "hair")
    total = 0
    valid = 0
    invalid_elements: list[list[Any]] = []
    hallucinated_elements: list[list[Any]] = []

    for panel_idx, panel in enumerate(root.findall(".//panel"), start=1):
        for pos in POSITION_TAGS:
            pos_elem = panel.find(pos)
            if pos_elem is None:
                continue
            for fig in pos_elem.findall("figure"):
                cid = t(fig.find("id"))
                if not cid:
                    continue
                variants = (characters.get(cid, {}) or {}).get("variants", {}) or {}
                allowed_by_attr = {a: set() for a in attrs}
                for v in variants.values():
                    if not isinstance(v, dict):
                        continue
                    for a in attrs:
                        vv = (v.get(a) or "").strip()
                        if vv:
                            allowed_by_attr[a].add(vv)

                loc = f"Scene1_Panel{panel_idx}_{pos}"
                for a in attrs:
                    val = t(fig.find(a))
                    if not val:
                        continue
                    total += 1
                    if val in allowed_by_attr[a]:
                        valid += 1
                    else:
                        invalid_elements.append([val, a, loc])
                        hallucinated_elements.append([a, val, loc])

    return {
        "total_elements": total,
        "valid_elements": valid,
        "hallucinated_elements_count": total - valid,
        "invalid_elements": invalid_elements,
        "hallucinated_elements": hallucinated_elements,
    }


def parse_story_label(story_dir_name: str) -> str:
    return story_dir_name.replace("story_", "", 1)


def compute_multi_character_ratio(root: ET.Element, dimension: str, mcr_mode: str) -> float | None:
    # This metric is only defined for multi-character dimensions (dim3/dim4).
    if dimension not in {"dim3_multi_char_single_story", "dim4_multi_char_multi_story"}:
        return None
    panels = root.findall(".//panel")
    if not panels:
        return None

    multi_char_panels = 0
    for panel in panels:
        n_fig = 0
        ids: set[str] = set()
        for pos in POSITION_TAGS:
            pos_elem = panel.find(pos)
            if pos_elem is None:
                continue
            figures = pos_elem.findall("figure")
            n_fig += len(figures)
            for fig in figures:
                cid = t(fig.find("id"))
                if cid:
                    ids.add(cid)
        if mcr_mode == "distinct_ids":
            is_multi = len(ids) >= 2
        else:
            is_multi = n_fig >= 2
        if is_multi:
            multi_char_panels += 1
    return multi_char_panels / len(panels)


def evaluate_one_file(
    xml_file: Path,
    output_base: Path,
    library_scope: str,
    content_types: set[str],
    schema_mode: str,
    mcr_mode: str,
    panel_min_override: int | None,
    enable_attr_consistency: bool,
) -> dict[str, Any]:
    rel = xml_file.relative_to(output_base)
    parts = rel.parts  # model/dimension/story_x/genN.xml
    model = parts[0] if len(parts) > 0 else "unknown_model"
    dimension = parts[1] if len(parts) > 1 else "unknown_dimension"
    story_dir = parts[2] if len(parts) > 2 else "story_unknown"
    story_label = parse_story_label(story_dir)

    result: dict[str, Any] = {
        "file": str(rel),
        "well_formed": False,
        "schema_valid": False,
        "content_validity_ratio": 0.0,
        "hallucination_rate": 0.0,
        "panel_compliant": False,
        "multi_character_ratio": None,
        "errors": [],
        "details": {
            "invalid_elements": [],
            "total_elements": 0,
            "valid_elements": 0,
            "hallucinated_elements": [],
            "panel_count": 0,
        },
        "model": model,
        "dimension": dimension,
        "story_label": story_label,
    }

    try:
        root = ET.parse(xml_file).getroot()
        result["well_formed"] = True
    except Exception as e:
        result["errors"].append(f"XML parse error: {e}")
        return result

    schema_ok, schema_errors, panel_count = validate_schema_and_panels(root, schema_mode=schema_mode)
    result["schema_valid"] = schema_ok
    result["errors"].extend(schema_errors)
    result["details"]["panel_count"] = panel_count

    # panel_compliant: whether panel count falls within the dimension range.
    panel_range = DIMENSIONS.get(dimension, {}).get("panel_range")
    if panel_range and panel_count > 0:
        lo, hi = panel_range
        if panel_min_override is not None:
            lo = panel_min_override
        result["panel_compliant"] = lo <= panel_count <= hi

    # multi_character_ratio
    result["multi_character_ratio"] = compute_multi_character_ratio(
        root, dimension, mcr_mode=mcr_mode
    )

    # content validity + hallucination
    try:
        if library_scope == "global":
            lib = load_global_element_library()
        else:
            lib = load_element_library(story_label)
        if lib is None:
            result["errors"].append(f"Element library not found for story_label={story_label}")
            return result
        allowed = build_allowed_sets(lib)
        content = evaluate_content(root, allowed, content_types=content_types)
        if enable_attr_consistency:
            extra = evaluate_character_attr_consistency(root, lib)
            content["total_elements"] += extra["total_elements"]
            content["valid_elements"] += extra["valid_elements"]
            content["invalid_elements"].extend(extra["invalid_elements"])
            content["hallucinated_elements"].extend(extra["hallucinated_elements"])
            if content["total_elements"] > 0:
                hall = content["total_elements"] - content["valid_elements"]
                content["content_validity_ratio"] = content["valid_elements"] / content["total_elements"]
                content["hallucination_rate"] = hall / content["total_elements"]

        result["content_validity_ratio"] = content["content_validity_ratio"]
        result["hallucination_rate"] = content["hallucination_rate"]
        result["details"]["invalid_elements"] = content["invalid_elements"]
        result["details"]["total_elements"] = content["total_elements"]
        result["details"]["valid_elements"] = content["valid_elements"]
        result["details"]["hallucinated_elements"] = content["hallucinated_elements"]
    except Exception as e:
        result["errors"].append(f"Content evaluation error: {e}")

    return result


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dimension"])].append(r)

    summary: dict[str, dict[str, Any]] = {}
    for (model, dimension), rs in grouped.items():
        total = len(rs)
        wf = [1.0 if bool(x["well_formed"]) else 0.0 for x in rs]
        sv = [1.0 if bool(x["schema_valid"]) else 0.0 for x in rs]
        pc = [1.0 if bool(x["panel_compliant"]) else 0.0 for x in rs]
        cv = [float(x["content_validity_ratio"]) for x in rs]
        hr = [float(x["hallucination_rate"]) for x in rs]
        mcr = [float(x["multi_character_ratio"]) for x in rs if x.get("multi_character_ratio") is not None]

        key = f"{model}_{dimension}"
        summary[key] = {
            "model": model,
            "dimension": dimension,
            "total_files": total,
            "well_formed_count": int(sum(wf)),
            "schema_valid_count": int(sum(sv)),
            "panel_compliant_count": int(sum(pc)),
            "well_formed_rate": safe_mean(wf),
            "schema_validity_rate": safe_mean(sv),
            "panel_compliance_rate": safe_mean(pc),
            "avg_content_validity_ratio": safe_mean(cv),
            "avg_hallucination_rate": safe_mean(hr),
            "avg_multi_character_ratio": safe_mean(mcr),
        }
    return summary


def is_legacy_story_label(story_label: str) -> bool:
    if "_" in story_label:
        return story_label in LEGACY_MULTI_STORIES
    return story_label in LEGACY_SINGLE_STORIES


def collect_xml_files(
    output_dir: Path,
    allowed_models: list[str] | None,
    use_legacy_story_filter: bool,
) -> list[Path]:
    files = sorted(output_dir.rglob("*.xml"))
    if not allowed_models:
        return files
    allow = set(allowed_models)
    selected = []
    for p in files:
        rel = p.relative_to(output_dir)
        model = rel.parts[0] if rel.parts else ""
        if model not in allow:
            continue
        if use_legacy_story_filter:
            rel = p.relative_to(output_dir)
            story_dir = rel.parts[2] if len(rel.parts) > 2 else ""
            story_label = parse_story_label(story_dir)
            if not is_legacy_story_label(story_label):
                continue
        selected.append(p)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute legacy structural metrics (detailed + summary) from output XML files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGE2_ROOT / "outputs_constrained_4dim",
        help="Input XML output directory (default: stage2_root/outputs_constrained_4dim).",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Result directory (default: this script directory).",
    )
    parser.add_argument(
        "--allowed-models",
        nargs="*",
        default=None,
        help="Optional: evaluate only these model folders.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional: manually set timestamp (YYYYMMDD_HHMMSS) for reproducible naming.",
    )
    parser.add_argument(
        "--disable-legacy-story-filter",
        action="store_true",
        help="Disable legacy story-combination filtering (enabled by default for 20260302 compatibility).",
    )
    parser.add_argument(
        "--library-scope",
        choices=["story", "global"],
        default="story",
        help="Element-library scope for content validity checks (default: story).",
    )
    parser.add_argument(
        "--schema-mode",
        choices=["lenient", "strict"],
        default="lenient",
        help="Schema strictness level (default: lenient).",
    )
    parser.add_argument(
        "--mcr-mode",
        choices=["figures", "distinct_ids"],
        default="distinct_ids",
        help="Decision mode for multi_character_ratio (default: distinct_ids).",
    )
    parser.add_argument(
        "--content-types",
        type=str,
        default="setting,id,pose",
        help="Comma-separated element types used for content validity checks.",
    )
    parser.add_argument(
        "--panel-min-override",
        type=int,
        default=None,
        help="Optional: override the minimum compliant panel count globally (e.g., 5).",
    )
    parser.add_argument(
        "--enable-attr-consistency-check",
        action="store_true",
        help="Enable ID-based appearance/skin/hair consistency checking (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    xml_files = collect_xml_files(
        output_dir=output_dir,
        allowed_models=args.allowed_models,
        use_legacy_story_filter=not args.disable_legacy_story_filter,
    )
    if not xml_files:
        raise FileNotFoundError(f"No xml files found under {output_dir}")

    print(f"Output dir : {output_dir}")
    print(f"XML files  : {len(xml_files)}")
    content_types = {x.strip() for x in args.content_types.split(",") if x.strip()}
    print(
        f"Config     : library_scope={args.library_scope}, "
        f"schema_mode={args.schema_mode}, mcr_mode={args.mcr_mode}, "
        f"content_types={sorted(content_types)}, "
        f"panel_min_override={args.panel_min_override}, "
        f"attr_consistency={args.enable_attr_consistency_check}"
    )

    rows: list[dict[str, Any]] = []
    for idx, xml in enumerate(xml_files, start=1):
        print(f"[{idx}/{len(xml_files)}] {xml.relative_to(output_dir)}")
        rows.append(
            evaluate_one_file(
                xml_file=xml,
                output_base=output_dir,
                library_scope=args.library_scope,
                content_types=content_types,
                schema_mode=args.schema_mode,
                mcr_mode=args.mcr_mode,
                panel_min_override=args.panel_min_override,
                enable_attr_consistency=args.enable_attr_consistency_check,
            )
        )

    summary = build_summary(rows)
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_path = result_dir / f"detailed_results_{ts}.json"
    summary_path = result_dir / f"summary_{ts}.json"

    detailed_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Detailed: {detailed_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()

