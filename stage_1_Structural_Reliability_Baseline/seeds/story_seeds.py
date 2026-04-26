#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Stage-One narrative seeds with official GPT-4.1.

What this script does:
1) Load all settings and poses from element_library.json
2) Ask GPT-4.1 to generate 20 narrative seeds
3) Save the generated seeds to a file under seeds/

Output file:
  seeds/story_seeds_gpt41_generated.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI



# Config (fill API key here)
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""  # Optional; leave empty for official endpoint
MODEL_ID = "gpt-4.1"

ROOT = Path(__file__).resolve().parent
ELEMENT_LIBRARY_FILE = ROOT / "element_library.json"
OUTPUT_FILE = ROOT / "seeds" / "story_seeds_generated.json"


def build_client() -> OpenAI:
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set OPENAI_API_KEY in this file or environment."
        )
    if OPENAI_BASE_URL:
        return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=api_key)


def load_element_library(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing element library: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_values(raw: Any) -> List[str]:
    """Convert nested/list/dict element values into a deduplicated string list."""
    vals: List[str] = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, (str, int, float)):
                s = str(x).strip()
                if s:
                    vals.append(s)
            elif isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, (str, int, float)):
                        s = str(v).strip()
                        if s:
                            vals.append(s)
    elif isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                vals.extend(normalize_values(v))
            elif isinstance(v, (str, int, float)):
                s = str(v).strip()
                if s:
                    vals.append(s)
    elif isinstance(raw, (str, int, float)):
        s = str(raw).strip()
        if s:
            vals.append(s)

    dedup: List[str] = []
    seen = set()
    for v in vals:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def extract_settings_and_poses(element_library: Dict[str, Any]) -> Dict[str, List[str]]:
    settings = normalize_values(element_library.get("settings", []))
    poses = normalize_values(element_library.get("poses", []))
    return {"settings": settings, "poses": poses}


def build_prompt(settings: List[str], poses: List[str]) -> str:
    return f"""
You are designing narrative seeds for constrained comic generation.

Context requirements (must satisfy):
- A shared set of 20 narrative seeds is required so that both Method A and Method B operate on identical thematic inputs.
- Narrative seeds are minimal textual prompts that describe a situation/event.
- Seeds must serve as high-level narrative triggers.
- Seeds must NOT specify characters, exact poses, or detailed visual composition.
- Models should infer characters, actions, and visual composition during generation.

Important constraints:
1) Generate exactly 20 seeds.
2) Each seed should be 1 sentence, concise, and open-ended.
3) Do not use character names.
4) Do not mention exact pose names from the provided pose list.
5) Avoid over-specifying camera/composition details.
6) Keep themes diverse across social, workplace, public, indoor/outdoor, and unusual-event contexts.
7) Ensure seeds are suitable for ComiXML-style constrained generation.

Reference resources (for domain relevance only; do NOT copy exact pose names into seeds):
- Available settings ({len(settings)}):
{json.dumps(settings, ensure_ascii=False)}

- Available poses ({len(poses)}):
{json.dumps(poses, ensure_ascii=False)}

Return STRICT JSON only with this schema:
{{
  "description": "Story seed collection for generating ComicXML format comic stories",
  "principles": {{
    "seed_definition": "minimal prompt describing a situation/event without fixing characters, poses, or detailed visuals",
    "shared_input": "same 20 seeds shared by Method A and Method B"
  }},
  "story_seeds": [
    {{
      "id": "seed_01",
      "theme": "..."
    }}
  ]
}}

No markdown. No extra text.
""".strip()


def call_gpt41(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You produce strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    # Try direct JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Model output is not valid JSON.")


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    seeds = payload.get("story_seeds")
    if not isinstance(seeds, list):
        raise ValueError("Missing 'story_seeds' list in model output.")
    if len(seeds) != 20:
        raise ValueError(f"Expected 20 seeds, got {len(seeds)}.")

    normalized = []
    for i, item in enumerate(seeds, start=1):
        if isinstance(item, str):
            theme = item.strip()
        elif isinstance(item, dict):
            theme = str(item.get("theme", "")).strip()
        else:
            theme = ""
        if not theme:
            raise ValueError(f"Seed {i} is empty.")
        normalized.append({"id": f"seed_{i:02d}", "theme": theme})

    return {
        "description": payload.get(
            "description", "Story seed collection for generating ComicXML format comic stories"
        ),
        "principles": payload.get(
            "principles",
            {
                "seed_definition": "minimal prompt describing a situation/event without fixing characters, poses, or detailed visuals",
                "shared_input": "same 20 seeds shared by Method A and Method B",
            },
        ),
        "story_seeds": normalized,
    }


def main() -> None:
    element_library = load_element_library(ELEMENT_LIBRARY_FILE)
    resources = extract_settings_and_poses(element_library)
    prompt = build_prompt(resources["settings"], resources["poses"])
    client = build_client()

    print(f"[INFO] settings: {len(resources['settings'])}, poses: {len(resources['poses'])}")
    print(f"[INFO] calling {MODEL_ID} ...")
    raw = call_gpt41(client, prompt)
    payload = parse_json_response(raw)
    final_data = validate_payload(payload)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

