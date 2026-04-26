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
SPEC_STORY_ELEMENTS_DIR = ROOT / "spec_story_elements"
TEMPLATE_FILE = ROOT / "template.xml"
OUTPUT_BASE = ROOT / "outputs_constrained"

SLEEP_BETWEEN_CALLS = 1.0

MODELS = {
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


def build_user_prompt(source_label: str, elements: Dict, template_xml: str) -> str:
    elements_json = json.dumps(elements, ensure_ascii=False, indent=2)
    return f"""
You are an expert ComicXML generation assistant.

Task:
- Generate ONE complete comic story in ComicXML.
- Use the given source element library only.
- Follow the template structure strictly.

Source IDs:
{source_label}

Element library (allowed values only):
{elements_json}

ComicXML template (structure guide, not actual values):
{template_xml}

IMPORTANT CONSTRAINTS:
1. XML must be well-formed (proper tag closure and nesting).
2. XML must follow the template structure (no missing required elements).
3. Use only elements (characters, poses, settings, positions, facings, balloon_status) from the element library above.
4. You may reuse elements across panels, but do not invent new ones.
5. Panel count must be between 6 and 12.

Output:
- Return ONLY the final <comic> XML, no extra explanation.
""".strip()


def run_generation(
    selected_models: List[str],
    source_ids: List[int],
    gen_per_source: int = 2,
) -> None:
    if not selected_models:
        selected_models = list(MODELS.keys())
    if not source_ids:
        raise ValueError("You must provide at least one source ID.")

    source_ids_sorted = sorted(source_ids)
    source_label = "_".join(str(sid) for sid in source_ids_sorted)
    template_xml = load_template()
    elements = (
        load_story_elements(source_ids_sorted[0])
        if len(source_ids_sorted) == 1
        else merge_story_elements(source_ids_sorted)
    )
    openai_client, together_client = build_clients()

    print("\n" + "=" * 80)
    print("Source-only Constrained Generation (official OpenAI + Together API)")
    print("=" * 80)
    print(f"Source IDs      : {source_ids_sorted} (label={source_label})")
    print(f"Models to run   : {', '.join(selected_models)}")
    print(f"Generations/model: {gen_per_source}")
    print("=" * 80 + "\n")

    prompt = build_user_prompt(source_label=source_label, elements=elements, template_xml=template_xml)
    messages = [
        {"role": "system", "content": "You are a strict ComicXML generation assistant."},
        {"role": "user", "content": prompt},
    ]

    for display_name in selected_models:
        if display_name not in MODELS:
            print(f"[WARN] Unknown model name (skipped): {display_name}")
            continue

        model_dir, prompt_type, api_model_name = MODELS[display_name]
        client = pick_client(api_model_name, openai_client, together_client)
        backend = "Together" if api_model_name in TOGETHER_API_MODEL_IDS else "OpenAI"

        print("\n" + "=" * 80)
        print(
            f"Running model: {display_name} "
            f"(dir={model_dir}, prompt_type={prompt_type}, backend={backend})"
        )
        print("=" * 80)

        for gen_idx in range(1, gen_per_source + 1):
            print(f"  -> Generating: source={source_label}, gen{gen_idx}")
            try:
                xml_text = chat_completion(client, api_model_name, messages)
            except RuntimeError as e:
                out_err = OUTPUT_BASE / model_dir / f"source_{source_label}" / f"gen{gen_idx}_error.txt"
                out_err.parent.mkdir(parents=True, exist_ok=True)
                out_err.write_text(str(e) + "\n", encoding="utf-8")
                print(f"     [ERROR] {e} — error saved to {out_err}")
                continue

            time.sleep(SLEEP_BETWEEN_CALLS)
            out_dir = OUTPUT_BASE / model_dir / f"source_{source_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"gen{gen_idx}.xml"
            out_path.write_text(xml_text, encoding="utf-8")
            print(f"     Saved: {out_path}")

    print("\nAll generations completed.")


def run_interactive() -> None:
    print("\n" + "=" * 80)
    print("Interactive Source-only Constrained Generation (official API)")
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

    source_input = input("Enter source IDs separated by spaces (e.g., '5 6 7'): ").strip()
    if not source_input:
        raise ValueError("You must provide at least one source ID.")
    source_ids = [int(x) for x in source_input.split()]

    gen_input = input("Generations per model [default 2]: ").strip()
    gen_per_source = int(gen_input) if gen_input else 2

    run_generation(
        selected_models=selected_models,
        source_ids=source_ids,
        gen_per_source=gen_per_source,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_interactive()
        raise SystemExit(0)

    parser = argparse.ArgumentParser(
        description="Source-only constrained ComicXML generation (official API)."
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
        "--source-ids",
        type=int,
        nargs="+",
        default=None,
        help="Source story IDs, e.g. --source-ids 2 3 5.",
    )
    parser.add_argument(
        "--gen-per-source",
        type=int,
        default=2,
        help="Number of generations per model (default: 2).",
    )

    args = parser.parse_args()
    run_generation(
        selected_models=args.model,
        source_ids=args.source_ids or [],
        gen_per_source=args.gen_per_source,
    )

