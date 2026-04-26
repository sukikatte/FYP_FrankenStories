#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# Paths / resources
ROOT = Path(__file__).resolve().parent
SEEDS_FILE = ROOT / "seeds" / "story_seeds.json"
ELEMENT_LIBRARY_FILE = ROOT / "library" / "element_library.json"
TEMPLATE_FILE = ROOT.parent / "template.xml"
OUTPUT_BASE = ROOT / "outputs"

SLEEP_BETWEEN_CALLS = 1.0
DEFAULT_GEN_PER_TOPIC = 1

# API config (fill with key)
OPENAI_API_KEY = ""
TOGETHER_API_KEY = ""
OPENAI_BASE_URL = ""  # optional
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# DeepSeek-R1-Distill-Qwen-7B is often not serverless on Together, need to deploy it first and then replace this value with your deployed endpoint/model id.
TOGETHER_MODEL_DEEPSEEK_R1_QWEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TOGETHER_MODEL_DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3.1"
TOGETHER_MODEL_LLAMA = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

load_dotenv(ROOT / ".env")  # optional fallback only


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    output_dir: str
    provider: str
    model_id: str
    prompt_style: str  # "CoT" | "Direct"


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "GPT-4.1": ModelConfig("GPT-4.1", "gpt_4.1", "openai", "gpt-4.1", "CoT"),
    "o1": ModelConfig("o1", "gpt_o1", "openai", "o1", "Direct"),
    "DeepSeek-R1-Qwen": ModelConfig(
        "DeepSeek-R1-Qwen", "deepseek_7b", "together", TOGETHER_MODEL_DEEPSEEK_R1_QWEN, "CoT"
    ),
    "DeepSeek-V3": ModelConfig("DeepSeek-V3", "deepseek_v3", "together", TOGETHER_MODEL_DEEPSEEK_V3, "Direct"),
    "Llama-3.3": ModelConfig(
        "Llama-3.3",
        "llama_3_70b",
        "together",
        TOGETHER_MODEL_LLAMA,
        "Direct",
    ),
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_resources():
    if not SEEDS_FILE.exists():
        raise FileNotFoundError(f"Missing seeds file: {SEEDS_FILE}")
    if not ELEMENT_LIBRARY_FILE.exists():
        raise FileNotFoundError(f"Missing element library: {ELEMENT_LIBRARY_FILE}")
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE_FILE}")

    seeds_raw = load_json(SEEDS_FILE)
    seeds: List[str] = []

    # Supported formats:
    # 1) ["seed1", "seed2", ...]
    # 2) {"story_seeds":[{"theme":"..."}, ...], ...}
    # 3) {"1":"...", "2":"..."} legacy mapping
    if isinstance(seeds_raw, list):
        for item in seeds_raw:
            if isinstance(item, str) and item.strip():
                seeds.append(item.strip())
            elif isinstance(item, dict):
                text = str(item.get("theme") or item.get("seed") or item.get("text") or "").strip()
                if text:
                    seeds.append(text)
    elif isinstance(seeds_raw, dict):
        if isinstance(seeds_raw.get("story_seeds"), list):
            for item in seeds_raw["story_seeds"]:
                if isinstance(item, str) and item.strip():
                    seeds.append(item.strip())
                elif isinstance(item, dict):
                    text = str(item.get("theme") or item.get("seed") or item.get("text") or "").strip()
                    if text:
                        seeds.append(text)
        else:
            for v in seeds_raw.values():
                if isinstance(v, str) and v.strip():
                    seeds.append(v.strip())
    else:
        raise ValueError("Unsupported seeds format in seeds/story_seeds.json")

    if not seeds:
        raise ValueError("No valid seed text found in seeds/story_seeds.json")

    element_library = load_json(ELEMENT_LIBRARY_FILE)
    template_xml = TEMPLATE_FILE.read_text(encoding="utf-8")
    return seeds, element_library, template_xml


def build_clients() -> Dict[str, OpenAI]:
    clients: Dict[str, OpenAI] = {}

    openai_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        openai_base = OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL")
        clients["openai"] = OpenAI(api_key=openai_key, base_url=openai_base) if openai_base else OpenAI(api_key=openai_key)

    together_key = TOGETHER_API_KEY or os.environ.get("TOGETHER_API_KEY")
    if together_key:
        together_base = TOGETHER_BASE_URL or os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        clients["together"] = OpenAI(api_key=together_key, base_url=together_base)

    return clients


def chat(client: OpenAI, model_id: str, messages: List[dict], max_tokens: int = 3000, temperature: float = 0.4) -> str:
    params = {"model": model_id, "messages": messages, "temperature": temperature}
    if model_id.startswith("o1"):
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**params)
            text = resp.choices[0].message.content or ""
            return text
        except Exception as e:
            if attempt == 2:
                raise
            print(f"[WARN] API call failed ({model_id}) attempt {attempt+1}/3: {e}")
            time.sleep(1.5 * (attempt + 1))
    return ""


def build_method_b_prompt(seed: str, element_library: dict, template_xml: str, prompt_style: str) -> str:
    cot_line = (
        "Reason internally if needed, but output only final ComiXML."
        if prompt_style == "CoT"
        else "Generate directly."
    )
    return f"""
You are a ComicXML generation assistant.

Given:
  - Seed story text
  - Full allowed element library
  - ComiXML template with required fields

Seed:
{seed}

Element Library:
{json.dumps(element_library, ensure_ascii=False)}

Template:
{template_xml}

Task:
Fill the template in one pass to produce final ComiXML.

Rules:
  - Use only library-allowed visual elements.
  - Ensure panel-count and schema-valid ComiXML output.
  - Keep panel count between 6 and 12.

Instruction:
{cot_line}

Output:
Return only the final <comic>...</comic> XML, with no extra text.
""".strip()


def build_method_a_single_call_prompt(seed: str, element_library: dict, template_xml: str, prompt_style: str) -> str:
    cot_line = (
        "Follow Stage 1 and Stage 2 internally, then output only final ComiXML."
        if prompt_style == "CoT"
        else "Follow the stages and output directly."
    )
    return f"""
You are a ComicXML generation assistant.

Use this staged workflow in ONE response:

Method A, Stage 1 (visual planning under constraints):
- Task: Plan panel-level visual elements before final ComiXML.
- Given:
    - Seed story text
    - Full allowed element library
    - ComiXML template with required fields
- Instructions:
    1) Select only allowed characters/poses/settings.
    2) Build a compact panel-level visual plan (6-12 panels).
    3) Keep the plan for Stage 2 in the same response.

Method A, Stage 2 (dialogue completion conditioned on Stage-1 plan):
- Task: Generate final ComiXML from the Stage-1 visual plan.
- Requirements:
    - Keep panel-level visual assignments from Stage 1 unchanged.
    - Add dialogue/text content consistent with the visual plan.
    - Use only library-allowed visual elements.
    - Ensure panel-count and schema-valid ComiXML output.
    - Keep panel count between 6 and 12.
    - Return final ComiXML only.

Seed:
{seed}

Element Library:
{json.dumps(element_library, ensure_ascii=False)}

Template:
{template_xml}

Instruction:
{cot_line}

Output:
Return only the final <comic>...</comic> XML, with no extra text.
""".strip()


def generate_method_b(client: OpenAI, cfg: ModelConfig, seed: str, element_library: dict, template_xml: str) -> str:
    user_prompt = build_method_b_prompt(seed, element_library, template_xml, cfg.prompt_style)
    messages = [
        {"role": "system", "content": "You are a strict ComicXML assistant."},
        {"role": "user", "content": user_prompt},
    ]
    return chat(client, cfg.model_id, messages, max_tokens=3200, temperature=0.25)


def generate_method_a(client: OpenAI, cfg: ModelConfig, seed: str, element_library: dict, template_xml: str) -> str:
    user_prompt = build_method_a_single_call_prompt(seed, element_library, template_xml, cfg.prompt_style)
    messages = [
        {"role": "system", "content": "You are a strict ComicXML assistant."},
        {"role": "user", "content": user_prompt},
    ]
    return chat(client, cfg.model_id, messages, max_tokens=3200, temperature=0.25)


def save_xml(xml_text: str, output_dir_name: str, method_name: str, topic_idx: int, gen_idx: int) -> Path:
    out_dir = OUTPUT_BASE / output_dir_name / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"topic{topic_idx}_gen{gen_idx}_{method_name}.xml"
    out_path.write_text(xml_text, encoding="utf-8")
    return out_path


def parse_selector(raw: str, options: List[str]) -> List[str]:
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return options[:]
    picked: List[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= len(options):
                picked.append(options[idx - 1])
            else:
                raise ValueError(f"Invalid index: {idx}")
        elif tok in options:
            picked.append(tok)
        else:
            raise ValueError(f"Invalid selector: {tok}")
    # 去重保序
    dedup = []
    seen = set()
    for p in picked:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def choose_interactively(available_models: List[str], seeds: List[str]):
    print("\n=== Stage One Official API Generation ===")
    print("Available models:")
    for i, m in enumerate(available_models, 1):
        print(f"  {i}. {m}")
    print("  all. all models")

    model_raw = input("\nChoose model(s) (e.g. 1,3 or all) [all]: ")
    models = parse_selector(model_raw, available_models)

    methods_options = ["A", "B"]
    method_raw = input("Choose method(s): A, B, or A,B [A,B]: ").strip()
    if not method_raw:
        methods = methods_options
    else:
        methods = parse_selector(method_raw, methods_options)

    gen_raw = input(f"Generations per topic [default {DEFAULT_GEN_PER_TOPIC}]: ").strip()
    gen_count = int(gen_raw) if gen_raw else DEFAULT_GEN_PER_TOPIC
    if gen_count < 1:
        raise ValueError("Generation count must be >= 1")

    print("\nAvailable seeds:")
    for i, s in enumerate(seeds, 1):
        preview = s.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  {i:>2}. {preview}")

    seed_raw = input("Seed indices (1-based, comma-separated) or all [all]: ").strip()
    seed_indices = None if (not seed_raw or seed_raw.lower() == "all") else [int(x.strip()) for x in seed_raw.split(",") if x.strip()]
    return models, methods, gen_count, seed_indices


def run(models: List[str], methods: List[str], gen_count: int, topic_indices: Optional[List[int]]):
    seeds, element_library, template_xml = load_resources()
    clients = build_clients()

    valid_models = []
    for m in models:
        cfg = MODEL_CONFIGS[m]
        if cfg.provider not in clients:
            print(f"[SKIP] {m}: missing credentials for provider '{cfg.provider}' in .env")
            continue
        valid_models.append(m)

    if not valid_models:
        raise RuntimeError("No runnable models selected (check .env API keys/base URLs).")

    if topic_indices is None:
        topic_indices = list(range(1, len(seeds) + 1))
    else:
        topic_indices = [i for i in topic_indices if 1 <= i <= len(seeds)]
        if not topic_indices:
            raise ValueError("No valid topic indices.")

    print("\nRun plan:")
    print(f"- Models : {', '.join(valid_models)}")
    print(f"- Methods: {', '.join(methods)}")
    print(f"- Topics : {len(topic_indices)} topics -> {topic_indices}")
    print(f"- Gen/topic/method: {gen_count}")
    print(f"- Output base: {OUTPUT_BASE}")

    for t_idx in topic_indices:
        seed = seeds[t_idx - 1]
        for g in range(1, gen_count + 1):
            for model_name in valid_models:
                cfg = MODEL_CONFIGS[model_name]
                client = clients[cfg.provider]
                if "B" in methods:
                    print(f"[{model_name}][MethodB] topic{t_idx} gen{g} ...")
                    xml_b = generate_method_b(client, cfg, seed, element_library, template_xml)
                    p = save_xml(xml_b, cfg.output_dir, "MethodB", t_idx, g)
                    print(f"  saved -> {p}")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                if "A" in methods:
                    print(f"[{model_name}][MethodA] topic{t_idx} gen{g} ...")
                    xml_a = generate_method_a(client, cfg, seed, element_library, template_xml)
                    p = save_xml(xml_a, cfg.output_dir, "MethodA", t_idx, g)
                    print(f"  saved -> {p}")
                    time.sleep(SLEEP_BETWEEN_CALLS)


def main():
    parser = argparse.ArgumentParser(description="Stage One generation with official APIs and interactive selection.")
    parser.add_argument("--non-interactive", action="store_true", help="Use CLI args only.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), help="Model names.")
    parser.add_argument("--methods", nargs="+", choices=["A", "B"], help="Methods to run.")
    parser.add_argument("--gen", type=int, default=DEFAULT_GEN_PER_TOPIC, help="Generations per topic.")
    parser.add_argument("--topics", nargs="+", type=int, help="1-based topic indices.")
    parser.add_argument("--seeds", nargs="+", type=int, help="1-based seed indices (alias of --topics).")
    args = parser.parse_args()

    available_models = list(MODEL_CONFIGS.keys())

    if args.non_interactive:
        models = args.models or available_models
        methods = args.methods or ["A", "B"]
        indices = args.seeds if args.seeds is not None else args.topics
        run(models=models, methods=methods, gen_count=args.gen, topic_indices=indices)
        return

    seeds, _, _ = load_resources()
    models, methods, gen_count, topic_indices = choose_interactively(available_models, seeds)
    run(models=models, methods=methods, gen_count=gen_count, topic_indices=topic_indices)


if __name__ == "__main__":
    main()

