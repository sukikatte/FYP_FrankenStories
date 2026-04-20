#!/usr/bin/env python3
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path


def _text(node, tag):
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value if value else None


def _collect_from_spec(spec_path: Path):
    root = ET.parse(spec_path).getroot()

    poses = set()
    settings = set()
    facings = set()
    positions = set()
    balloon_status = set()
    border = set()
    framing = set()

    # collect characters as specification，blank part will be set as "none"
    name_by_id = {}
    combos_by_id = defaultdict(list)
    combo_seen = defaultdict(set)

    for panel in root.findall(".//panel"):
        st = _text(panel, "setting")
        if st:
            settings.add(st)

        bd = _text(panel, "border")
        if bd:
            border.add(bd)

        embedding_panel = panel.find("embedding/panel")
        emb_setting = _text(embedding_panel, "setting") if embedding_panel is not None else None
        if emb_setting:
            settings.add(emb_setting)

        emb_framing = _text(embedding_panel, "framing") if embedding_panel is not None else None
        if emb_framing:
            framing.add(emb_framing)

        for pos_tag in ("left", "middle", "right"):
            pos_node = panel.find(pos_tag)
            if pos_node is None:
                continue
            positions.add(pos_tag)

            for balloon in pos_node.findall("balloon"):
                status = balloon.get("status")
                if status is not None:
                    status = status.strip()
                    if status:
                        balloon_status.add(status)

            for figure in pos_node.findall("figure"):
                pose = _text(figure, "pose")
                if pose:
                    poses.add(pose)

                facing = _text(figure, "facing")
                if facing:
                    facings.add(facing)

                cid = _text(figure, "id")
                cname = _text(figure, "name")
                appearance = _text(figure, "appearance")
                skin = _text(figure, "skin")
                hair = _text(figure, "hair")
                lips = _text(figure, "lips")

                # remove duplicates
                if not cid or not cname:
                    continue
                name_by_id[cid] = cname
                combo = (
                    appearance or "none",
                    skin or "none",
                    hair or "none",
                    lips or "none",
                )
                if combo not in combo_seen[cid]:
                    combo_seen[cid].add(combo)
                    combos_by_id[cid].append(combo)

    return {
        "name_by_id": name_by_id,
        "combos_by_id": combos_by_id,
        "poses": sorted(poses),
        "settings": sorted(settings),
        "facings": sorted(facings),
        "positions": sorted(positions),
        "balloon_status": sorted(balloon_status),
        "border": sorted(border),
        "framing": sorted(framing),
    }


def _build_characters(data):
    characters = OrderedDict()
    for cid in sorted(data["combos_by_id"].keys()):
        combos = data["combos_by_id"][cid]
        if not combos:
            continue

        variants = OrderedDict()
        for i, combo in enumerate(combos):
            name = "default" if i == 0 else f"variant_{i}"
            appearance, skin, hair, lips = combo
            variants[name] = OrderedDict(
                [
                    ("appearance", appearance),
                    ("skin", skin),
                    ("hair", hair),
                    ("lips", lips),
                ]
            )

        characters[cid] = OrderedDict(
            [
                ("id", cid),
                ("name", data["name_by_id"].get(cid, cid)),
                ("variants", variants),
            ]
        )

    return characters


def build_element_library(spec_path: Path):
    data = _collect_from_spec(spec_path)
    characters = _build_characters(data)

    result = OrderedDict()
    result["characters"] = characters
    result["poses"] = data["poses"]
    result["settings"] = data["settings"]
    result["facings"] = data["facings"]
    result["positions"] = data["positions"]
    result["balloon_status"] = data["balloon_status"]
    result["border"] = data["border"]
    result["framing"] = data["framing"]
    return result


def main():
    spec_path = Path("../specification.xml")
    output_path = Path("element_library_output.json")
    library = build_element_library(spec_path)
    output_path.write_text(json.dumps(library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
