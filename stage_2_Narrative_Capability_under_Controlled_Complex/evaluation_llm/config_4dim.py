#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation_config_4dim - shared module for 4-dimension evaluation.

Provides load_element_library(story_label) and DIMENSIONS for scripts such as
evaluate_llm_judge_local_metrics and evaluate_diversity_metrics.
Data source: spec_story_elements/story_*.json (aligned with constrained_generation_4dim).
"""

import json
from pathlib import Path

SPEC_ELEMENTS_DIR = Path("spec_story_elements")

# 4-dimension experiment configuration (aligned with constrained_generation_4dim and output layout)
DIMENSIONS = {
    "dim1_single_char_single_story": {"char_count": 1, "story_count": "single", "panel_range": (6, 12)},
    "dim2_single_char_multi_story": {"char_count": 1, "story_count": "multi", "panel_range": (6, 12)},
    "dim3_multi_char_single_story": {"char_count": 2, "story_count": "single", "panel_range": (6, 12)},
    "dim4_multi_char_multi_story": {"char_count": 2, "story_count": "multi", "panel_range": (6, 12)},
}


def load_element_library(story_label: str):
    """
    Load the element library by story_label.
    story_label: "1" means single-story, "1_2" means merged multi-story.
    Data source: spec_story_elements/story_<id>.json.
    """
    story_ids = [int(s) for s in str(story_label).strip().split("_")]

    if len(story_ids) == 1:
        story_file = SPEC_ELEMENTS_DIR / f"story_{story_ids[0]}.json"
        if story_file.exists():
            with open(story_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # Multi-story: merge libraries
    merged = {
        "characters": {},
        "poses": set(),
        "settings": set(),
        "facings": set(),
        "positions": set(),
        "balloon_status": set(),
    }

    for story_id in story_ids:
        story_file = SPEC_ELEMENTS_DIR / f"story_{story_id}.json"
        if not story_file.exists():
            continue
        with open(story_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for char_id, char_data in data.get("characters", {}).items():
            if char_id not in merged["characters"]:
                merged["characters"][char_id] = char_data
            else:
                existing = merged["characters"][char_id].get("variants", {})
                existing.update(char_data.get("variants", {}))
                merged["characters"][char_id]["variants"] = existing

        merged["poses"].update(data.get("poses", []))
        merged["settings"].update(data.get("settings", []))
        merged["facings"].update(data.get("facings", []))
        merged["positions"].update(data.get("positions", []))
        merged["balloon_status"].update(data.get("balloon_status", []))

    return {
        "characters": merged["characters"],
        "poses": sorted(merged["poses"]),
        "settings": sorted(merged["settings"]),
        "facings": sorted(merged["facings"]),
        "positions": sorted(merged["positions"]),
        "balloon_status": sorted(merged["balloon_status"]),
    }
