#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import openai

# Config, fill keys here
OPENAI_API_KEY = ""
TOGETHER_API_KEY = ""
OPENAI_BASE_URL = ""
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

OPENAI_MODEL_GPT41 = "gpt-4.1"
OPENAI_MODEL_GPT4O = "gpt-4o"
OPENAI_MODEL_O1 = "o1"
# DeepSeek-R1-Distill-Qwen-7B is often not serverless on Together, need to deploy it first and then replace this value with your deployed endpoint/model id.
TOGETHER_MODEL_DEEPSEEK_R1_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TOGETHER_MODEL_DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3.1"
TOGETHER_MODEL_DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
TOGETHER_MODEL_LLAMA = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

ROOT = Path(__file__).resolve().parent
SPEC_STORY_ELEMENTS_DIR = ROOT / "spec_library"
TEMPLATE_FILE = ROOT.parent / "template.xml"
OUTPUT_BASE = ROOT / "outputs_constrained_4dim"

SLEEP_BETWEEN_CALLS = 1.0

# 4-dimension setup
DIMENSIONS: Dict[str, Dict] = {
    "dim1_single_char_single_story": {"char_count": 1, "story_count": "single"},
    "dim2_single_char_multi_story": {"char_count": 1, "story_count": "multi"},
    "dim3_multi_char_single_story": {"char_count": 2, "story_count": "single"},
    "dim4_multi_char_multi_story": {"char_count": 2, "story_count": "multi"},
}

MODELS = {
    # key: display name -> (output_dir_name, prompt_type, api_model_id)
    "gpt-4.1": ("gpt_4.1", "CoT", OPENAI_MODEL_GPT41),
    "gpt-4o": ("gpt_4o", "CoT", OPENAI_MODEL_GPT4O),
    "o1": ("gpt_o1", "Direct", OPENAI_MODEL_O1),
    "DeepSeek-R1-Distill-Qwen-7B": ("deepseek_7b", "CoT", TOGETHER_MODEL_DEEPSEEK_R1_QWEN),
    "DeepSeek-V3": ("deepseek_v3", "CoT", TOGETHER_MODEL_DEEPSEEK_V3),
    "DeepSeek-R1": ("deepseek_r1", "CoT", TOGETHER_MODEL_DEEPSEEK_R1),
    "llama-3.3-70B": ("llama_3_70b", "Direct", TOGETHER_MODEL_LLAMA),
}

TOGETHER_API_MODEL_IDS = frozenset(
    {
        TOGETHER_MODEL_DEEPSEEK_R1_QWEN,
        TOGETHER_MODEL_DEEPSEEK_V3,
        TOGETHER_MODEL_DEEPSEEK_R1,
        TOGETHER_MODEL_LLAMA,
    }
)


def build_clients() -> tuple[openai.OpenAI, openai.OpenAI]:
    openai_key = (OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY") or "").strip()
    together_key = (TOGETHER_API_KEY or os.environ.get("TOGETHER_API_KEY") or "").strip()
    openai_base = (OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL") or "").strip()
    together_base = (
        TOGETHER_BASE_URL
        or os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    ).strip()

    if not openai_key or not together_key:
        raise RuntimeError(
            "Please set OPENAI_API_KEY and TOGETHER_API_KEY at the top of this file "
            "or via environment variables."
        )

    if openai_base:
        openai_client = openai.OpenAI(api_key=openai_key, base_url=openai_base)
    else:
        openai_client = openai.OpenAI(api_key=openai_key)

    together_client = openai.OpenAI(
        api_key=together_key, base_url=together_base or "https://api.together.xyz/v1"
    )
    return openai_client, together_client


def pick_client(
    api_model: str,
    openai_client: openai.OpenAI,
    together_client: openai.OpenAI,
) -> openai.OpenAI:
    if api_model in TOGETHER_API_MODEL_IDS:
        return together_client
    return openai_client


def load_template() -> str:
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"template.xml not found at {TEMPLATE_FILE}")
    return TEMPLATE_FILE.read_text(encoding="utf-8")


def load_story_elements(story_id: int) -> Dict:
    story_file = SPEC_STORY_ELEMENTS_DIR / f"story_{story_id}.json"
    if not story_file.exists():
        raise FileNotFoundError(f"Story elements file not found: {story_file}")
    with story_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_story_elements(story_ids: List[int]) -> Dict:
    merged = {
        "characters": {},
        "poses": set(),
        "settings": set(),
        "facings": set(),
        "positions": set(),
        "balloon_status": set(),
    }

    for sid in story_ids:
        data = load_story_elements(sid)

        for char_id, char_data in data.get("characters", {}).items():
            if char_id not in merged["characters"]:
                merged["characters"][char_id] = char_data
            else:
                existing_variants = merged["characters"][char_id].get("variants", {})
                new_variants = char_data.get("variants", {})
                existing_variants.update(new_variants)
                merged["characters"][char_id]["variants"] = existing_variants

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


def chat_completion(
    client: openai.OpenAI,
    model_name: str,
    messages: List[dict],
    max_tokens: int = 2000,
    temperature: float = 0.4,
) -> str:
    params = {"model": model_name, "messages": messages, "temperature": temperature}
    if model_name.startswith("o1"):
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        if content:
            return content
        raise RuntimeError("API returned empty content.")
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from e


def build_user_prompt(
    model_name: str,
    dimension_id: str,
    story_label: str,
    elements: Dict,
    template_xml: str,
    char_count: int,
    story_count: str,
) -> str:
    dim_desc = {
        "dim1_single_char_single_story": "Dim1: Single character and single story.",
        "dim2_single_char_multi_story": "Dim2: Single character and multiple stories merged.",
        "dim3_multi_char_single_story": "Dim3: Multiple characters within a single story.",
        "dim4_multi_char_multi_story": "Dim4: Multiple characters and multiple stories merged.",
    }.get(dimension_id, dimension_id)

    if char_count == 1:
        char_constraint = (
            "- You MUST use only ONE main character ID from the element library\n"
            "  across the whole story (single-character setting).\n"
        )
    else:
        char_constraint = (
            "- You MUST use MULTIPLE distinct character IDs from the element library\n"
            "  across the story (multi-character setting), and most panels should\n"
            "  contain 2 characters.\n"
        )

    story_info = (
        f"You must create a ComicXML story for **{dim_desc}**\n"
        f"- Story source type: {story_count} (story_label = {story_label})\n"
        "- Panel count must be between 6 and 12 (inclusive).\n"
        f"{char_constraint}"
    )

    elements_json = json.dumps(elements, ensure_ascii=False, indent=2)

    prompt = f"""
You are an expert ComicXML generation assistant.

Your task:
- Generate a COMPLETE ComicXML for a comic story.
- STRICTLY follow the structure defined by the template.
- Use ONLY elements from the provided element library.
- Respect the character constraints above (single-character or multi-character).

{story_info}

Element library (permitted values only):
{elements_json}

ComicXML template (STRUCTURE GUIDE, not actual values):
{template_xml}

IMPORTANT CONSTRAINTS:
1. XML must be well-formed (proper tag closure and nesting).
2. XML must follow the template structure (no missing required elements).
3. Use only elements (characters, poses, settings, positions, facings, balloon_status) from the element library above.
4. You MAY reuse elements across panels, but do not invent new ones.
5. Panel count must be between 6 and 12.
6. For multi-character dimensions, most panels should contain multiple characters.

Output:
- Return ONLY the final <comic> XML, no extra explanation.
"""
    return prompt


def generate_story_for_dimension(
    openai_client: openai.OpenAI,
    together_client: openai.OpenAI,
    api_model_name: str,
    model_dir: str,
    dimension_id: str,
    story_labels: List[str],
    template_xml: str,
    gen_per_story: int = 2,
) -> None:
    dim_info = DIMENSIONS[dimension_id]
    char_count = dim_info["char_count"]
    story_count_type = dim_info["story_count"]

    for story_label in story_labels:
        story_ids = [int(s) for s in story_label.split("_")]
        if len(story_ids) == 1:
            elements = load_story_elements(story_ids[0])
        else:
            elements = merge_story_elements(story_ids)

        for gen_idx in range(1, gen_per_story + 1):
            print(
                f"  -> Generating: model={model_dir}, dim={dimension_id}, "
                f"story={story_label}, gen{gen_idx}"
            )
            user_prompt = build_user_prompt(
                api_model_name,
                dimension_id,
                story_label,
                elements,
                template_xml,
                char_count,
                story_count_type,
            )
            messages = [
                {"role": "system", "content": "You are a strict ComicXML generation assistant."},
                {"role": "user", "content": user_prompt},
            ]

            client = pick_client(api_model_name, openai_client, together_client)
            xml_text = chat_completion(client, api_model_name, messages)
            time.sleep(SLEEP_BETWEEN_CALLS)

            out_dir = OUTPUT_BASE / model_dir / dimension_id / f"story_{story_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"gen{gen_idx}.xml"
            out_path.write_text(xml_text, encoding="utf-8")
            print(f"     Saved: {out_path}")


def run_4dim_generation(
    selected_models: List[str],
    story_ids: List[int],
    mode: str,
    gen_per_story: int = 2,
) -> None:
    if not selected_models:
        selected_models = list(MODELS.keys())

    if not story_ids:
        raise ValueError("You must provide at least one story ID via --stories.")

    if mode not in {"single", "multi"}:
        mode = "single" if len(story_ids) == 1 else "multi"

    story_ids_sorted = sorted(story_ids)
    story_label = "_".join(str(sid) for sid in story_ids_sorted)
    template_xml = load_template()
    openai_client, together_client = build_clients()

    print("\n" + "=" * 80)
    print("4-Dimension Constrained Generation (official OpenAI + Together API)")
    print("=" * 80)
    print(f"Mode           : {mode} (story_ids = {story_ids_sorted}, label = {story_label})")
    print(f"Models to run  : {', '.join(selected_models)}")
    print(f"Generations/Dim: {gen_per_story}")
    print("=" * 80 + "\n")

    if mode == "single":
        if len(story_ids_sorted) != 1:
            raise ValueError("Single-story mode expects exactly ONE story ID.")
        dims_to_run = [
            "dim1_single_char_single_story",
            "dim3_multi_char_single_story",
        ]
    else:
        if len(story_ids_sorted) < 2:
            raise ValueError("Multi-story mode expects at least TWO story IDs.")
        dims_to_run = [
            "dim2_single_char_multi_story",
            "dim4_multi_char_multi_story",
        ]

    for display_name in selected_models:
        if display_name not in MODELS:
            print(f"[WARN] Unknown model name (skipped): {display_name}")
            continue

        model_dir, prompt_type, api_model_name = MODELS[display_name]

        print("\n" + "=" * 80)
        print(f"Running model: {display_name} (dir={model_dir}, prompt_type={prompt_type})")
        print("=" * 80)

        for dim_id in dims_to_run:
            print(f"\n[Dimension] {dim_id}")
            generate_story_for_dimension(
                openai_client=openai_client,
                together_client=together_client,
                api_model_name=api_model_name,
                model_dir=model_dir,
                dimension_id=dim_id,
                story_labels=[story_label],
                template_xml=template_xml,
                gen_per_story=gen_per_story,
            )

    print("\nAll generations completed.")


def run_interactive() -> None:
    print("\n" + "=" * 80)
    print("Interactive 4-Dimension Constrained Generation (official API)")
    print("=" * 80)
    print("Available models:")
    for idx, name in enumerate(MODELS.keys(), start=1):
        print(f"  {idx}. {name}")
    print("  all. Run all models")

    model_input = input(
        "\nSelect model(s) by number (e.g., 1,2) or 'all' [all]: "
    ).strip()
    if not model_input or model_input.lower() == "all":
        selected_models = list(MODELS.keys())
    else:
        selected_models = []
        for token in model_input.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                i = int(token)
                if 1 <= i <= len(MODELS):
                    selected_models.append(list(MODELS.keys())[i - 1])
                else:
                    raise ValueError(f"Invalid model index: {i}")
            elif token in MODELS:
                selected_models.append(token)
            else:
                raise ValueError(f"Invalid model selector: {token}")
        selected_models = list(dict.fromkeys(selected_models))

    stories_input = input("Enter story IDs separated by spaces (e.g., '5 6 7'): ").strip()
    if not stories_input:
        raise ValueError("You must provide at least one story ID.")
    story_ids = [int(x) for x in stories_input.split()]

    gen_input = input("Generations per story per dimension [default 2]: ").strip()
    gen_per_story = int(gen_input) if gen_input else 2

    run_4dim_generation(
        selected_models=selected_models,
        story_ids=story_ids,
        mode="",
        gen_per_story=gen_per_story,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_interactive()
        raise SystemExit(0)

    parser = argparse.ArgumentParser(
        description="Constrained ComicXML generation for 4 dimensions."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODELS.keys()),
        help="Models to run. Default: all models.",
    )
    parser.add_argument(
        "--stories",
        type=int,
        nargs="+",
        default=None,
        help="Story IDs to use as element library source, e.g. --stories 2 3 5.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["single", "multi"],
        help=(
            "Source mode: 'single' for single-story source (Dim1 & Dim3), "
            "'multi' for multi-story source (Dim2 & Dim4). "
            "If omitted, inferred from number of story IDs."
        ),
    )
    parser.add_argument(
        "--gen-per-story",
        type=int,
        default=2,
        help="Number of generations per story per dimension (default: 2).",
    )

    args = parser.parse_args()
    run_4dim_generation(
        selected_models=args.model,
        story_ids=args.stories or [],
        mode=args.mode or "",
        gen_per_story=args.gen_per_story,
    )

