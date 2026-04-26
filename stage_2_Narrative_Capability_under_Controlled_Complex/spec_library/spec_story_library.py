#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any


POSITION_TAGS = ("left", "middle", "right")


def _txt(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _attr(node: ET.Element | None, name: str) -> str:
    if node is None:
        return ""
    value = node.get(name)
    return value.strip() if value else ""


def _unique_sorted(values: set[str]) -> list[str]:
    return sorted(v for v in values if v != "")


def extract_story_elements(scene: ET.Element) -> dict[str, Any]:
    poses: set[str] = set()
    settings: set[str] = set()
    facings: set[str] = set()
    positions: set[str] = set()
    balloon_status: set[str] = set()

    characters: "OrderedDict[str, Any]" = OrderedDict()

    for panel in scene.findall("panel"):
        # setting can appear directly under panel (content panels)
        st = _txt(panel, "setting")
        if st:
            settings.add(st)

        # setting can also appear inside cover-like embedding structure
        embedding_panel = panel.find("embedding/panel")
        emb_setting = _txt(embedding_panel, "setting")
        if emb_setting:
            settings.add(emb_setting)

        for pos_tag in POSITION_TAGS:
            pos_node = panel.find(pos_tag)
            if pos_node is None:
                continue
            positions.add(pos_tag)

            for balloon in pos_node.findall("balloon"):
                status = _attr(balloon, "status")
                if status:
                    balloon_status.add(status)

            for figure in pos_node.findall("figure"):
                pose = _txt(figure, "pose")
                if pose:
                    poses.add(pose)

                facing = _txt(figure, "facing")
                if facing:
                    facings.add(facing)

                cid = _txt(figure, "id")
                cname = _txt(figure, "name")
                if not cid or not cname:
                    # prop figures do not have id/name; ignore for character library
                    continue

                # Keep the first observed variant as "default".
                # Missing attributes remain as empty strings to match existing outputs.
                if cid not in characters:
                    characters[cid] = OrderedDict(
                        [
                            ("id", cid),
                            ("name", cname),
                            (
                                "variants",
                                OrderedDict(
                                    [
                                        (
                                            "default",
                                            OrderedDict(
                                                [
                                                    ("appearance", _txt(figure, "appearance")),
                                                    ("skin", _txt(figure, "skin")),
                                                    ("hair", _txt(figure, "hair")),
                                                    ("lips", _txt(figure, "lips")),
                                                ]
                                            ),
                                        )
                                    ]
                                ),
                            ),
                        ]
                    )

    return OrderedDict(
        [
            ("characters", characters),
            ("poses", _unique_sorted(poses)),
            ("settings", _unique_sorted(settings)),
            ("facings", _unique_sorted(facings)),
            ("positions", _unique_sorted(positions)),
            ("balloon_status", _unique_sorted(balloon_status)),
        ]
    )


def build_all_stories(spec_path: Path) -> "OrderedDict[str, Any]":
    root = ET.parse(spec_path).getroot()
    scenes_node = root.find("scenes")
    if scenes_node is None:
        raise ValueError("Invalid specification: missing <scenes> root node")

    all_stories: "OrderedDict[str, Any]" = OrderedDict()
    scenes = list(scenes_node.findall("scene"))
    for idx, scene in enumerate(scenes):
        story_key = f"story_{idx}"
        story_elements = extract_story_elements(scene)
        all_stories[story_key] = OrderedDict(
            [
                ("story_index", idx),
                ("story_label", idx),
                *list(story_elements.items()),
            ]
        )
    return all_stories


def _write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-story element libraries from specification.xml."
    )
    parser.add_argument(
        "--spec",
        default=str(Path(__file__).resolve().parents[2] / "specification.xml"),
        help="Path to specification.xml",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory to write story_*.json and all_stories_library.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON outputs (default: skip if exists).",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stories = build_all_stories(spec_path)

    # Write aggregated file
    _write_json(out_dir / "all_stories_library.json", all_stories, overwrite=args.overwrite)

    # Write per-story files
    for story_key, payload in all_stories.items():
        idx = payload["story_index"]
        story_path = out_dir / f"story_{idx}.json"
        per_story = OrderedDict(
            [
                ("characters", payload["characters"]),
                ("poses", payload["poses"]),
                ("settings", payload["settings"]),
                ("facings", payload["facings"]),
                ("positions", payload["positions"]),
                ("balloon_status", payload["balloon_status"]),
            ]
        )
        _write_json(story_path, per_story, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

