#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continuity and Coherence evaluation script.
Evaluation dimensions:
1. Continuity: use Claude Sonnet 4.6 as an LLM judge to score how natural
   scene transitions are between adjacent panels (1-5).
2. Coherence: use Claude Sonnet 4.6 to score dialogue-pose/emotion alignment
   panel by panel (1-5).
"""

import xml.etree.ElementTree as ET
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import re
from anthropic import Anthropic

# Import shared evaluation helpers
sys.path.insert(0, str(Path(__file__).parent))

# Required dependency: evaluation_config_4dim
from config_4dim import load_element_library, DIMENSIONS

OUTPUT_DIR = Path("outputs_constrained_4dim")
EVAL_OUTPUT_DIR = Path("evaluation_results_continuity_coherence")
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_MODELS = ["gpt_4", "gpt_4o", "deepseek_v3", "deepseek_r1", "llama_3_70b"]

# API configuration (official Claude API, direct call)
CLAUDE_API_KEY = ""
CLAUDE_MODEL = "claude-sonnet-4-6"

# API call configuration
SLEEP_BETWEEN_CALLS = 1.0  # Delay between API calls (seconds)
MAX_RETRIES = 3  # Maximum retries


def get_claude_client():
    """Get official Claude API client."""
    if not CLAUDE_API_KEY:
        print("Warning: CLAUDE_API_KEY is empty. LLM Judge features will be disabled.")
        return None
    return Anthropic(api_key=CLAUDE_API_KEY)


def extract_panel_info(panel, panel_idx):
    """Extract panel information for LLM evaluation."""
    info = {
        'panel_index': panel_idx,
        'setting': None,
        'characters': [],
        'dialogue': None,
        'below_text': None
    }
    
    # Setting
    setting = panel.find('setting')
    if setting is not None and setting.text:
        info['setting'] = setting.text.strip()
    
    # Characters and poses
    for pos in ['left', 'middle', 'right']:
        pos_elem = panel.find(pos)
        if pos_elem is not None:
            figure = pos_elem.find('figure')
            if figure is not None:
                char_id = figure.find('id')
                pose = figure.find('pose')
                if char_id is not None and char_id.text:
                    char_info = {
                        'id': char_id.text.strip(),
                        'position': pos,
                        'pose': pose.text.strip() if pose is not None and pose.text else None
                    }
                    info['characters'].append(char_info)
                
                # Dialogue (balloon)
                balloon = pos_elem.find('balloon')
                if balloon is not None:
                    content = balloon.find('content')
                    if content is not None and content.text:
                        info['dialogue'] = content.text.strip()
    
    # Below text
    below = panel.find('below')
    if below is not None and below.text:
        info['below_text'] = below.text.strip()
    
    return info


def format_panel_for_llm(panel_info):
    """Format panel information as LLM-readable text."""
    lines = [f"Panel {panel_info['panel_index'] + 1}:"]
    
    if panel_info['setting']:
        lines.append(f"  Setting: {panel_info['setting']}")
    
    if panel_info['characters']:
        lines.append("  Characters:")
        for char in panel_info['characters']:
            char_str = f"    - {char['id']}"
            if char['pose']:
                char_str += f" (pose: {char['pose']})"
            if char['position']:
                char_str += f" [position: {char['position']}]"
            lines.append(char_str)
    
    if panel_info['dialogue']:
        lines.append(f"  Dialogue: \"{panel_info['dialogue']}\"")
    
    if panel_info['below_text']:
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
            
            time.sleep(SLEEP_BETWEEN_CALLS)  # Avoid rate limiting
            texts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
            return "".join(texts).strip()
        
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
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
    
    # Try extracting a digit directly
    score_match = re.search(r'\b([1-5])\b', response_text)
    if score_match:
        return int(score_match.group(1))
    
    # Fallback: infer score from wording
    response_lower = response_text.lower()
    if '5' in response_text or 'five' in response_lower or 'excellent' in response_lower or 'perfect' in response_lower:
        return 5
    elif '4' in response_text or 'four' in response_lower or 'good' in response_lower or 'very good' in response_lower:
        return 4
    elif '3' in response_text or 'three' in response_lower or 'moderate' in response_lower or 'average' in response_lower:
        return 3
    elif '2' in response_text or 'two' in response_lower or 'poor' in response_lower or 'bad' in response_lower:
        return 2
    elif '1' in response_text or 'one' in response_lower or 'very poor' in response_lower or 'terrible' in response_lower:
        return 1
    
    return None


def evaluate_continuity_llm(xml_content, client):
    """
    Evaluate scene continuity with an LLM judge.
    Score range: 1-5 (1=abrupt/inconsistent, 5=smooth/natural progression).
    """
    if not client:
        return {
            'score': None,
            'error': 'Claude API not available'
        }
    
    try:
        root = ET.fromstring(xml_content)
        panels = root.findall('.//panel')
        
        if len(panels) < 2:
            return {
                'score': None,
                'error': 'Not enough panels for continuity evaluation'
            }
        
        # Extract all panel information
        panel_infos = []
        for idx, panel in enumerate(panels):
            panel_info = extract_panel_info(panel, idx)
            panel_infos.append(panel_info)
        
        # Build prompt
        panel_texts = []
        for panel_info in panel_infos:
            panel_texts.append(format_panel_for_llm(panel_info))
        
        prompt = f"""Evaluate whether the scene transitions between panels are logically continuous in this comic story.

Story panels:
{chr(10).join(panel_texts)}

Please evaluate the continuity of scene transitions between adjacent panels. Consider:
1. Whether the settings/scenes flow naturally from one panel to the next
2. Whether character positions and movements make logical sense
3. Whether the narrative progression is smooth and coherent

Score from 1 to 5:
- 1: Abrupt and inconsistent transitions
- 2: Mostly inconsistent with some logical flow
- 3: Moderately consistent with some gaps
- 4: Mostly smooth with minor inconsistencies
- 5: Smooth and natural progression throughout

Respond with ONLY a single number (1-5) representing the overall continuity score."""

        # Call Claude API
        response_text = call_claude_api(client, prompt, max_tokens=10)
        
        if response_text is None:
            return {
                'score': None,
                'error': 'API call failed'
            }
        
        # Extract score
        score = extract_score_from_response(response_text)
        
        if score is None:
            return {
                'score': None,
                'error': f'Could not extract score from response: {response_text}'
            }
        
        return {
            'score': score,
            'max_score': 5,
            'normalized_score': score / 5.0,
            'raw_response': response_text
        }
    
    except Exception as e:
        return {
            'score': None,
            'error': f'Evaluation error: {str(e)}'
        }


def evaluate_coherence_llm(xml_content, client):
    """
    Evaluate dialogue-pose coherence panel by panel using an LLM judge.
    Each panel is scored 1-5, then averaged.
    """
    if not client:
        return {
            'avg_score': None,
            'panel_scores': [],
            'error': 'Claude API not available'
        }
    
    try:
        root = ET.fromstring(xml_content)
        panels = root.findall('.//panel')
        
        if len(panels) == 0:
            return {
                'avg_score': None,
                'panel_scores': [],
                'error': 'No panels found'
            }
        
        panel_scores = []
        panel_details = []
        
        # Evaluate panel by panel
        for idx, panel in enumerate(panels):
            panel_info = extract_panel_info(panel, idx)
            
            # Skip panels without dialogue or characters
            if not panel_info['dialogue'] or not panel_info['characters']:
                continue
            
            # Build prompt
            panel_text = format_panel_for_llm(panel_info)
            
            prompt = f"""Evaluate whether the dialogue aligns with the character's pose and emotion in this comic panel.

{panel_text}

Please evaluate the coherence between the dialogue and the character's pose/emotion. Consider:
1. Whether the dialogue content matches the character's pose (e.g., a "sad" pose should have sad dialogue)
2. Whether the emotional tone of the dialogue aligns with the character's pose
3. Whether the dialogue makes sense given the character's physical position and pose

Score from 1 to 5:
- 1: Dialogue and pose are completely inconsistent
- 2: Mostly inconsistent with some alignment
- 3: Moderately consistent with some gaps
- 4: Mostly consistent with minor inconsistencies
- 5: Dialogue perfectly aligns with pose and emotion

Respond with ONLY a single number (1-5) representing the coherence score for this panel."""

            # Call Claude API
            response_text = call_claude_api(client, prompt, max_tokens=10)
            
            if response_text is None:
                panel_details.append({
                    'panel_index': idx,
                    'score': None,
                    'error': 'API call failed'
                })
                continue
            
            # Extract score
            score = extract_score_from_response(response_text)
            
            if score is not None:
                panel_scores.append(score)
                panel_details.append({
                    'panel_index': idx,
                    'score': score,
                    'raw_response': response_text
                })
            else:
                panel_details.append({
                    'panel_index': idx,
                    'score': None,
                    'error': f'Could not extract score from response: {response_text}',
                    'raw_response': response_text
                })
        
        if not panel_scores:
            return {
                'avg_score': None,
                'panel_scores': panel_details,
                'error': 'No valid scores obtained'
            }
        
        avg_score = sum(panel_scores) / len(panel_scores)
        
        return {
            'avg_score': avg_score,
            'max_score': 5,
            'normalized_score': avg_score / 5.0,
            'panel_scores': panel_scores,
            'panel_details': panel_details
        }
    
    except Exception as e:
        return {
            'avg_score': None,
            'panel_scores': [],
            'error': f'Evaluation error: {str(e)}'
        }


def evaluate_file(xml_file, element_library, story_label, client):
    """Evaluate Continuity and Coherence for one file."""
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # 1. Continuity evaluation
        continuity_result = evaluate_continuity_llm(xml_content, client)
        
        # 2. Coherence evaluation
        coherence_result = evaluate_coherence_llm(xml_content, client)
        
        return {
            'file': str(xml_file),
            'continuity': continuity_result,
            'coherence': coherence_result
        }
    
    except Exception as e:
        return {
            'file': str(xml_file),
            'error': f'Evaluation error: {str(e)}'
        }


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
                        xml_files = list(story_dir.glob("*.xml"))
                        total_files += len(xml_files)
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation for {total_files} files")
    print(f"Estimated API calls: {total_files * 2} (2 per file: Continuity + Coherence)")
    print(f"Estimated time: {total_files * 2 * SLEEP_BETWEEN_CALLS / 60:.1f} minutes")
    print(f"{'='*60}\n")
    
    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        if model_name not in TARGET_MODELS:
            continue
        
        print(f"\nEvaluating model: {model_name}")
        
        for dimension_id, dimension_info in DIMENSIONS.items():
            dim_dir = model_dir / dimension_id
            if not dim_dir.exists():
                continue
            
            print(f"  Dimension: {dimension_id}")
            
            for story_dir in dim_dir.iterdir():
                if not story_dir.is_dir():
                    continue
                
                story_label = story_dir.name.replace('story_', '')
                
                # Load element library
                element_library = load_element_library(story_label)
                if not element_library:
                    print(f"    [WARN] Element library not found for: {story_label}")
                    continue
                
                # Evaluate all XML files in this story directory
                for xml_file in story_dir.glob("*.xml"):
                    evaluated_files += 1
                    print(f"    [{evaluated_files}/{total_files}] Evaluating: {xml_file.name}")
                    result = evaluate_file(xml_file, element_library, story_label, client)
                    result['model'] = model_name
                    result['dimension'] = dimension_id
                    result['story_label'] = story_label
                    all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results (JSON)
    detailed_file = EVAL_OUTPUT_DIR / f"detailed_results_{timestamp}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    generate_summary(all_results, timestamp)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {detailed_file}")
    print(f"Total files evaluated: {len(all_results)}")
    print(f"{'='*60}")
    
    return all_results


def generate_summary(all_results, timestamp):
    """Generate summary statistics."""
    summary = {}
    
    for result in all_results:
        if 'error' in result:
            continue
        
        model = result.get('model')
        dimension = result.get('dimension')
        key = f"{model}_{dimension}"
        
        if key not in summary:
            summary[key] = {
                'model': model,
                'dimension': dimension,
                'count': 0,
                'continuity': {
                    'scores': [],
                    'normalized_scores': []
                },
                'coherence': {
                    'avg_scores': [],
                    'normalized_scores': []
                }
            }
        
        summary[key]['count'] += 1
        
        # Continuity
        if 'continuity' in result and result['continuity'].get('score') is not None:
            cont = result['continuity']
            summary[key]['continuity']['scores'].append(cont.get('score', 0))
            summary[key]['continuity']['normalized_scores'].append(cont.get('normalized_score', 0))
        
        # Coherence
        if 'coherence' in result and result['coherence'].get('avg_score') is not None:
            coh = result['coherence']
            summary[key]['coherence']['avg_scores'].append(coh.get('avg_score', 0))
            summary[key]['coherence']['normalized_scores'].append(coh.get('normalized_score', 0))
    
    # Compute averages
    for key in summary:
        data = summary[key]
        
        # Continuity averages
        cont = data['continuity']
        cont['avg_score'] = sum(cont['scores']) / len(cont['scores']) if cont['scores'] else None
        cont['avg_normalized_score'] = sum(cont['normalized_scores']) / len(cont['normalized_scores']) if cont['normalized_scores'] else None
        
        # Coherence averages
        coh = data['coherence']
        coh['avg_score'] = sum(coh['avg_scores']) / len(coh['avg_scores']) if coh['avg_scores'] else None
        coh['avg_normalized_score'] = sum(coh['normalized_scores']) / len(coh['normalized_scores']) if coh['normalized_scores'] else None
    
    # Save summary
    summary_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save CSV summary
    csv_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Model,Dimension,Count,Continuity Score,Continuity Normalized,Coherence Score,Coherence Normalized\n")
        for key, data in sorted(summary.items()):
            model = data['model']
            dimension = data['dimension']
            count = data['count']
            
            cont_score = data['continuity'].get('avg_score', 'N/A')
            cont_norm = data['continuity'].get('avg_normalized_score', 'N/A')
            coh_score = data['coherence'].get('avg_score', 'N/A')
            coh_norm = data['coherence'].get('avg_normalized_score', 'N/A')
            
            f.write(f"{model},{dimension},{count},{cont_score},{cont_norm},{coh_score},{coh_norm}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    evaluate_all()

