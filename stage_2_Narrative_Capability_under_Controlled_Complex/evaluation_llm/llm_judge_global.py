#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import re
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent))

from config_4dim import load_element_library, DIMENSIONS

OUTPUT_DIR = Path("outputs_constrained_4dim")
EVAL_OUTPUT_DIR = Path("evaluation_results_global_continuity_entertainment")
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_MODELS = ["gpt_4", "gpt_4o", "deepseek_v3", "deepseek_r1", "llama_3_70b"]

# API configuration (official Claude API, direct call)
CLAUDE_API_KEY = ""
CLAUDE_MODEL = "claude-sonnet-4-6"

# API call configuration
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3


def get_claude_client():
    """Get official Claude API client."""
    if not CLAUDE_API_KEY:
        print("Warning: CLAUDE_API_KEY is empty. LLM Judge features will be disabled.")
        return None
    return Anthropic(api_key=CLAUDE_API_KEY)


def extract_panel_info(panel, panel_idx):
    """Extract panel information for LLM evaluation."""
    info = {
        "panel_index": panel_idx,
        "setting": None,
        "characters": [],
        "dialogue": None,
        "below_text": None,
    }

    # Setting
    setting = panel.find("setting")
    if setting is not None and setting.text:
        info["setting"] = setting.text.strip()

    # Characters and poses
    for pos in ["left", "middle", "right"]:
        pos_elem = panel.find(pos)
        if pos_elem is not None:
            figure = pos_elem.find("figure")
            if figure is not None:
                char_id = figure.find("id")
                pose = figure.find("pose")
                if char_id is not None and char_id.text:
                    char_info = {
                        "id": char_id.text.strip(),
                        "position": pos,
                        "pose": pose.text.strip() if pose is not None and pose.text else None,
                    }
                    info["characters"].append(char_info)

                balloon = pos_elem.find("balloon")
                if balloon is not None:
                    content = balloon.find("content")
                    if content is not None and content.text:
                        info["dialogue"] = content.text.strip()

    # Below text
    below = panel.find("below")
    if below is not None and below.text:
        info["below_text"] = below.text.strip()

    return info


def format_panel_for_llm(panel_info):
    """Format panel information as LLM-readable text."""
    lines = [f"Panel {panel_info['panel_index'] + 1}:"]

    if panel_info["setting"]:
        lines.append(f"  Setting: {panel_info['setting']}")

    if panel_info["characters"]:
        lines.append("  Characters:")
        for char in panel_info["characters"]:
            char_str = f"    - {char['id']}"
            if char["pose"]:
                char_str += f" (pose: {char['pose']})"
            if char["position"]:
                char_str += f" [position: {char['position']}]"
            lines.append(char_str)

    if panel_info["dialogue"]:
        lines.append(f"  Dialogue: \"{panel_info['dialogue']}\"")

    if panel_info["below_text"]:
        lines.append(f"  Narrative: {panel_info['below_text']}")

    return "\n".join(lines)


def call_claude_api(client, prompt, max_tokens=10):
    """Call official Claude API with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )

            time.sleep(SLEEP_BETWEEN_CALLS)
            texts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
            return "".join(texts).strip()

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2
                print(f"    [WARN] API call failed, retry in {wait_time}s... ({str(e)})")
                time.sleep(wait_time)
            else:
                print(f"    [ERROR] API call failed after maximum retries: {str(e)}")
                return None

    return None


def extract_score_from_response(response_text):
    """Extract a 1-5 score from model response text."""
    if not response_text:
        return None

    score_match = re.search(r"\b([1-5])\b", response_text)
    if score_match:
        return int(score_match.group(1))

    response_lower = response_text.lower()
    if "five" in response_lower or "excellent" in response_lower or "perfect" in response_lower:
        return 5
    if "four" in response_lower or "very good" in response_lower or "good" in response_lower:
        return 4
    if "three" in response_lower or "moderate" in response_lower or "average" in response_lower:
        return 3
    if "two" in response_lower or "poor" in response_lower:
        return 2
    if "one" in response_lower or "very poor" in response_lower or "terrible" in response_lower:
        return 1
    return None


def evaluate_global_continuity_llm(xml_content, client):
    """
    Evaluate global continuity with an LLM judge.
    Score range: 1-5 (1=globally fragmented, 5=globally coherent).
    """
    if not client:
        return {"score": None, "error": "Claude API not available"}

    try:
        root = ET.fromstring(xml_content)
        panels = root.findall(".//panel")

        if len(panels) < 2:
            return {"score": None, "error": "Not enough panels for global continuity evaluation"}

        panel_infos = [extract_panel_info(panel, idx) for idx, panel in enumerate(panels)]
        panel_texts = [format_panel_for_llm(panel_info) for panel_info in panel_infos]

        prompt = f"""Evaluate whether the story remains logically continuous at a GLOBAL level across all panels in this comic story.

Story panels:
{chr(10).join(panel_texts)}

Please evaluate the global continuity of the full story (not only adjacent transitions). Consider:
1. Whether earlier and later panels remain consistent in scene logic and story state
2. Whether character states, locations, and actions stay globally plausible over the full sequence
3. Whether the overall narrative arc (beginning-development-ending) is coherent from start to finish

Score from 1 to 5:
- 1: Globally fragmented and contradictory
- 2: Weak global continuity with multiple major inconsistencies
- 3: Moderately coherent overall with notable global gaps
- 4: Mostly coherent globally with minor inconsistencies
- 5: Strongly coherent global progression throughout

Respond with ONLY a single number (1-5) representing the overall global continuity score."""

        response_text = call_claude_api(client, prompt, max_tokens=10)
        if response_text is None:
            return {"score": None, "error": "API call failed"}

        score = extract_score_from_response(response_text)
        if score is None:
            return {"score": None, "error": f"Could not extract score from response: {response_text}"}

        return {
            "score": score,
            "max_score": 5,
            "normalized_score": score / 5.0,
            "raw_response": response_text,
        }
    except Exception as e:
        return {"score": None, "error": f"Evaluation error: {str(e)}"}


def evaluate_overall_entertainment_llm(xml_content, client):
    """
    Evaluate overall entertainment with an LLM judge.
    Score range: 1-5 (1=not entertaining, 5=highly entertaining).
    """
    if not client:
        return {"score": None, "error": "Claude API not available"}

    try:
        root = ET.fromstring(xml_content)
        panels = root.findall(".//panel")

        if len(panels) == 0:
            return {"score": None, "error": "No panels found"}

        panel_infos = [extract_panel_info(panel, idx) for idx, panel in enumerate(panels)]
        panel_texts = [format_panel_for_llm(panel_info) for panel_info in panel_infos]

        prompt = f"""Evaluate the overall entertainment value of this comic story.

Story panels:
{chr(10).join(panel_texts)}

Please evaluate overall entertainment by considering:
1. Humor, tension, or emotional engagement
2. Narrative interest and pacing across panels
3. Whether the story is enjoyable as a complete comic

Score from 1 to 5:
- 1: Not entertaining
- 2: Slightly entertaining
- 3: Moderately entertaining
- 4: Entertaining
- 5: Highly entertaining

Respond with ONLY a single number (1-5) representing the overall entertainment score."""

        response_text = call_claude_api(client, prompt, max_tokens=10)
        if response_text is None:
            return {"score": None, "error": "API call failed"}

        score = extract_score_from_response(response_text)
        if score is None:
            return {"score": None, "error": f"Could not extract score from response: {response_text}"}

        return {
            "score": score,
            "max_score": 5,
            "normalized_score": score / 5.0,
            "raw_response": response_text,
        }
    except Exception as e:
        return {"score": None, "error": f"Evaluation error: {str(e)}"}


def evaluate_file(xml_file, element_library, story_label, client):
    """Evaluate Global Continuity and Overall Entertainment for one file."""
    try:
        with open(xml_file, "r", encoding="utf-8") as f:
            xml_content = f.read()

        # 1. Global Continuity evaluation
        global_continuity_result = evaluate_global_continuity_llm(xml_content, client)

        # 2. Overall Entertainment evaluation
        overall_entertainment_result = evaluate_overall_entertainment_llm(xml_content, client)

        return {
            "file": str(xml_file),
            "global_continuity": global_continuity_result,
            "overall_entertainment": overall_entertainment_result,
        }
    except Exception as e:
        return {"file": str(xml_file), "error": f"Evaluation error: {str(e)}"}


def evaluate_all():
    """Evaluate all files."""
    client = get_claude_client()
    if not client:
        print("Warning: Claude API not available. LLM Judge features will be disabled.")
        print("Please set CLAUDE_API_KEY at the top of this file.")
        return []

    all_results = []
    total_files = 0
    evaluated_files = 0

    # Count total files first
    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_name not in TARGET_MODELS:
            continue
        for dimension_id in DIMENSIONS.keys():
            dim_dir = model_dir / dimension_id
            if dim_dir.exists():
                for story_dir in dim_dir.iterdir():
                    if story_dir.is_dir():
                        total_files += len(list(story_dir.glob("*.xml")))

    print(f"\n{'='*60}")
    print(f"Starting evaluation for {total_files} files")
    print(f"Estimated API calls: {total_files * 2} (2 per file: Global Continuity + Overall Entertainment)")
    print(f"Estimated time: {total_files * 2 * SLEEP_BETWEEN_CALLS / 60:.1f} minutes")
    print(f"{'='*60}\n")

    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if model_name not in TARGET_MODELS:
            continue

        print(f"\nEvaluating model: {model_name}")

        for dimension_id, _dimension_info in DIMENSIONS.items():
            dim_dir = model_dir / dimension_id
            if not dim_dir.exists():
                continue

            print(f"  Dimension: {dimension_id}")

            for story_dir in dim_dir.iterdir():
                if not story_dir.is_dir():
                    continue

                story_label = story_dir.name.replace("story_", "")

                element_library = load_element_library(story_label)
                if not element_library:
                    print(f"    [WARN] Element library not found for: {story_label}")
                    continue

                for xml_file in story_dir.glob("*.xml"):
                    evaluated_files += 1
                    print(f"    [{evaluated_files}/{total_files}] Evaluating: {xml_file.name}")
                    result = evaluate_file(xml_file, element_library, story_label, client)
                    result["model"] = model_name
                    result["dimension"] = dimension_id
                    result["story_label"] = story_label
                    all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_file = EVAL_OUTPUT_DIR / f"detailed_results_{timestamp}.json"
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    generate_summary(all_results, timestamp)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {detailed_file}")
    print(f"Total files evaluated: {len(all_results)}")
    print(f"{'='*60}")

    return all_results


def generate_summary(all_results, timestamp):
    """Generate summary statistics."""
    summary = {}

    for result in all_results:
        if "error" in result:
            continue

        model = result.get("model")
        dimension = result.get("dimension")
        key = f"{model}_{dimension}"

        if key not in summary:
            summary[key] = {
                "model": model,
                "dimension": dimension,
                "count": 0,
                "global_continuity": {"scores": [], "normalized_scores": []},
                "overall_entertainment": {"scores": [], "normalized_scores": []},
            }

        summary[key]["count"] += 1

        gc = result.get("global_continuity", {})
        if gc.get("score") is not None:
            summary[key]["global_continuity"]["scores"].append(gc.get("score", 0))
            summary[key]["global_continuity"]["normalized_scores"].append(gc.get("normalized_score", 0))

        oe = result.get("overall_entertainment", {})
        if oe.get("score") is not None:
            summary[key]["overall_entertainment"]["scores"].append(oe.get("score", 0))
            summary[key]["overall_entertainment"]["normalized_scores"].append(oe.get("normalized_score", 0))

    for key in summary:
        data = summary[key]
        gc = data["global_continuity"]
        oe = data["overall_entertainment"]

        gc["avg_score"] = sum(gc["scores"]) / len(gc["scores"]) if gc["scores"] else None
        gc["avg_normalized_score"] = (
            sum(gc["normalized_scores"]) / len(gc["normalized_scores"]) if gc["normalized_scores"] else None
        )

        oe["avg_score"] = sum(oe["scores"]) / len(oe["scores"]) if oe["scores"] else None
        oe["avg_normalized_score"] = (
            sum(oe["normalized_scores"]) / len(oe["normalized_scores"]) if oe["normalized_scores"] else None
        )

    summary_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(
            "Model,Dimension,Count,Global Continuity Score,Global Continuity Normalized,"
            "Overall Entertainment Score,Overall Entertainment Normalized\n"
        )
        for _key, data in sorted(summary.items()):
            model = data["model"]
            dimension = data["dimension"]
            count = data["count"]

            gc_score = data["global_continuity"].get("avg_score", "N/A")
            gc_norm = data["global_continuity"].get("avg_normalized_score", "N/A")
            oe_score = data["overall_entertainment"].get("avg_score", "N/A")
            oe_norm = data["overall_entertainment"].get("avg_normalized_score", "N/A")

            f.write(f"{model},{dimension},{count},{gc_score},{gc_norm},{oe_score},{oe_norm}\n")

    print(f"\nSummary saved to: {summary_file}")
    print(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    evaluate_all()

