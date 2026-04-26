#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diversity evaluation script.
Metrics:
1. Unique Usage Ratio (excluding facing)
2. Shannon Entropy over usage frequencies (excluding facing)
3. Source Distribution under multi-story settings, based on usage counts
4. Unique Source Share B_i/A under multi-story settings
"""

import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import math

# Base paths (stable regardless of current working directory)
SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_ROOT = SCRIPT_DIR.parent

# Import evaluation helpers
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(STAGE2_ROOT / "evaluate_llm"))

# Try importing load_element_library
try:
    from config_4dim import load_element_library, DIMENSIONS
except ImportError:
    # Fallback implementation if evaluation_config_4dim is unavailable
    SPEC_ELEMENTS_DIR = STAGE2_ROOT / "spec_story_elements"
    
    def load_element_library(story_label):
        """Load element library for a story or merged stories"""
        story_ids = [int(s) for s in story_label.split("_")]
        
        if len(story_ids) == 1:
            story_file = SPEC_ELEMENTS_DIR / f"story_{story_ids[0]}.json"
            if story_file.exists():
                with open(story_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        else:
            # Multiple stories - merge
            merged = {
                "characters": {},
                "poses": set(),
                "settings": set(),
                "facings": set(),
                "positions": set(),
                "balloon_status": set()
            }
            
            for story_id in story_ids:
                story_file = SPEC_ELEMENTS_DIR / f"story_{story_id}.json"
                if story_file.exists():
                    with open(story_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Merge characters
                        for char_id, char_data in data.get("characters", {}).items():
                            if char_id not in merged["characters"]:
                                merged["characters"][char_id] = char_data
                            else:
                                existing_variants = merged["characters"][char_id].get("variants", {})
                                new_variants = char_data.get("variants", {})
                                existing_variants.update(new_variants)
                                merged["characters"][char_id]["variants"] = existing_variants
                        
                        # Merge other elements
                        merged["poses"].update(data.get("poses", []))
                        merged["settings"].update(data.get("settings", []))
                        merged["facings"].update(data.get("facings", []))
                        merged["positions"].update(data.get("positions", []))
                        merged["balloon_status"].update(data.get("balloon_status", []))
            
            return {
                "characters": merged["characters"],
                "poses": sorted(list(merged["poses"])),
                "settings": sorted(list(merged["settings"])),
                "facings": sorted(list(merged["facings"])),
                "positions": sorted(list(merged["positions"])),
                "balloon_status": sorted(list(merged["balloon_status"]))
            }
        
        return None
    
    DIMENSIONS = {
        "dim1_single_char_single_story": {"char_count": 1, "story_count": "single", "panel_range": (6, 12)},
        "dim2_single_char_multi_story": {"char_count": 1, "story_count": "multi", "panel_range": (6, 12)},
        "dim3_multi_char_single_story": {"char_count": 2, "story_count": "single", "panel_range": (6, 12)},
        "dim4_multi_char_multi_story": {"char_count": 2, "story_count": "multi", "panel_range": (6, 12)},
    }

OUTPUT_DIR = STAGE2_ROOT / "outputs_constrained_4dim"
SPEC_ELEMENTS_DIR = STAGE2_ROOT / "spec_story_elements"
EVAL_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results_diversity"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_MODELS = ["gpt_4", "gpt_4o", "deepseek_v3", "deepseek_r1", "llama_3_70b"]

def extract_story_ids_from_label(story_label):
    """Extract story IDs from story_label."""
    if '_' in story_label:
        return [int(s) for s in story_label.split('_')]
    else:
        return [int(story_label)]


def calculate_shannon_entropy(counts):
    """Compute Shannon entropy: H = -sum(p_i log p_i)."""
    if not counts or sum(counts.values()) == 0:
        return 0.0
    
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def analyze_diversity(root, element_library, story_label):
    """
    Analyze element diversity (excluding facing).
    Returns unique_usage_ratio, shannon_entropy, and source_distribution.
    """
    story_ids = extract_story_ids_from_label(story_label)
    
    # Load source-story libraries (for Source Distribution)
    source_story_libraries = {}
    for story_id in story_ids:
        source_lib = load_element_library(str(story_id))
        if source_lib:
            source_story_libraries[story_id] = source_lib
    
    # Collect available elements (excluding facing)
    available_elements = {
        'characters': set(element_library.get('characters', {}).keys()) if element_library else set(),
        'poses': set(element_library.get('poses', [])) if element_library else set(),
        'settings': set(element_library.get('settings', [])) if element_library else set(),
        'positions': set(element_library.get('positions', [])) if element_library else set(),
    }
    
    # Count used elements by frequency (excluding facing)
    used_elements_count = {
        'characters': Counter(),
        'poses': Counter(),
        'settings': Counter(),
        'positions': Counter(),
    }
    
    # Track unique used elements (set, excluding facing)
    used_elements_unique = {
        'characters': set(),
        'poses': set(),
        'settings': set(),
        'positions': set(),
    }
    
    # Count usage attributable to each source story (for Source Distribution)
    # Format: {story_id: {element_type: {element_value: count}}}
    elements_by_story_count = {story_id: defaultdict(Counter) for story_id in story_ids}
    
    scenes = root.findall('.//scene')
    for scene in scenes:
        panels = scene.findall('panel')
        for panel in panels:
            # Count setting
            setting = panel.find('setting')
            if setting is not None and setting.text:
                setting_val = setting.text.strip()
                used_elements_count['settings'][setting_val] += 1
                used_elements_unique['settings'].add(setting_val)
                
                # Check which source story contains this value (count all matches)
                for story_id, source_lib in source_story_libraries.items():
                    if setting_val in source_lib.get('settings', []):
                        elements_by_story_count[story_id]['settings'][setting_val] += 1
            
            # Count figures (excluding facing)
            for pos in ['left', 'middle', 'right']:
                pos_elem = panel.find(pos)
                if pos_elem is not None:
                    used_elements_count['positions'][pos] += 1
                    used_elements_unique['positions'].add(pos)
                    
                    # Check which source story contains this position
                    for story_id, source_lib in source_story_libraries.items():
                        if pos in source_lib.get('positions', []):
                            elements_by_story_count[story_id]['positions'][pos] += 1
                    
                    figure = pos_elem.find('figure')
                    if figure is not None:
                        # Character ID
                        char_id = figure.find('id')
                        if char_id is not None and char_id.text:
                            char_val = char_id.text.strip()
                            used_elements_count['characters'][char_val] += 1
                            used_elements_unique['characters'].add(char_val)
                            
                            # Check which source story contains this character
                            for story_id, source_lib in source_story_libraries.items():
                                if char_val in source_lib.get('characters', {}):
                                    elements_by_story_count[story_id]['characters'][char_val] += 1
                        
                        # Pose
                        pose = figure.find('pose')
                        if pose is not None and pose.text:
                            pose_val = pose.text.strip()
                            used_elements_count['poses'][pose_val] += 1
                            used_elements_unique['poses'].add(pose_val)
                            
                            # Check which source story contains this pose
                            for story_id, source_lib in source_story_libraries.items():
                                if pose_val in source_lib.get('poses', []):
                                    elements_by_story_count[story_id]['poses'][pose_val] += 1
    
    # 1.1 Unique Usage Ratio
    total_available = sum(len(available_elements[elem_type]) for elem_type in available_elements)
    total_used_unique = sum(len(used_elements_unique[elem_type]) for elem_type in used_elements_unique)
    unique_usage_ratio = total_used_unique / total_available if total_available > 0 else 0.0
    
    # 1.2 Shannon Entropy (merged element types)
    all_counts = Counter()
    for elem_type in used_elements_count:
        all_counts.update(used_elements_count[elem_type])
    
    shannon_entropy = calculate_shannon_entropy(all_counts)
    
    # Compute max entropy for normalization
    total_unique = sum(len(used_elements_unique[elem_type]) for elem_type in used_elements_unique)
    max_entropy = math.log2(total_unique) if total_unique > 1 else (1.0 if total_unique == 1 else 0.0)
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    
    # 1.3 Source Distribution (multi-story usage share)
    source_distribution = {}
    if len(story_ids) > 1:  # Multi-story only
        # Count total usage for each source story
        story_total_counts = {}
        for story_id in story_ids:
            story_total = 0
            for elem_type in ['characters', 'poses', 'settings', 'positions']:
                story_total += sum(elements_by_story_count[story_id][elem_type].values())
            story_total_counts[story_id] = story_total
        
        # Count total usage
        total_usage = sum(story_total_counts.values())
        
        # Compute per-story ratio
        for story_id in story_ids:
            story_count = story_total_counts[story_id]
            source_distribution[story_id] = {
                'count': story_count,
                'ratio': story_count / total_usage if total_usage > 0 else 0.0
            }
        
        # Per-element-type breakdown
        source_distribution_by_type = {}
        for elem_type in ['characters', 'poses', 'settings', 'positions']:
            type_story_counts = {}
            for story_id in story_ids:
                type_count = sum(elements_by_story_count[story_id][elem_type].values())
                type_story_counts[story_id] = type_count
            
            type_total = sum(type_story_counts.values())
            source_distribution_by_type[elem_type] = {}
            for story_id in story_ids:
                type_count = type_story_counts[story_id]
                source_distribution_by_type[elem_type][story_id] = {
                    'count': type_count,
                    'ratio': type_count / type_total if type_total > 0 else 0.0
                }
        
        source_distribution['by_type'] = source_distribution_by_type
    else:
        # Single-story: Source Distribution is None
        source_distribution = None

    # 1.4 Unique-source share (multi-story):
    # A = total unique used items; B_i = items that exist in source i; ratio = B_i / A.
    # One item can contribute to multiple B_i if it appears in multiple source libraries.
    unique_source_share = None
    A_unique = sum(len(used_elements_unique[t]) for t in ['characters', 'poses', 'settings', 'positions'])
    if len(story_ids) > 1 and A_unique > 0:
        unique_source_share = {}
        for story_id in story_ids:
            lib = source_story_libraries.get(story_id)
            if not lib:
                unique_source_share[str(story_id)] = {
                    "unique_attributed_count": 0,
                    "denominator_A": A_unique,
                    "ratio": 0.0,
                }
                continue
            n_attr = 0
            for char_val in used_elements_unique['characters']:
                if char_val in lib.get('characters', {}):
                    n_attr += 1
            for pose_val in used_elements_unique['poses']:
                if pose_val in lib.get('poses', []):
                    n_attr += 1
            for sett in used_elements_unique['settings']:
                if sett in lib.get('settings', []):
                    n_attr += 1
            for pos in used_elements_unique['positions']:
                if pos in lib.get('positions', []):
                    n_attr += 1
            unique_source_share[str(story_id)] = {
                'unique_attributed_count': n_attr,
                'denominator_A': A_unique,
                'ratio': n_attr / A_unique,
            }
    elif len(story_ids) > 1:
        unique_source_share = {
            str(sid): {'unique_attributed_count': 0, 'denominator_A': 0, 'ratio': 0.0} for sid in story_ids
        }

    return {
        'unique_usage_ratio': unique_usage_ratio,
        'shannon_entropy': shannon_entropy,
        'normalized_entropy': normalized_entropy,
        'max_entropy': max_entropy,
        'source_distribution': source_distribution,
        'unique_source_share': unique_source_share,
        'details': {
            'total_available': total_available,
            'total_used_unique': total_used_unique,
            'used_elements_unique': {k: len(v) for k, v in used_elements_unique.items()},
            'available_elements': {k: len(v) for k, v in available_elements.items()},
            'used_elements_count': {k: dict(v) for k, v in used_elements_count.items()},
            'elements_by_story_count': {
                story_id: {
                    elem_type: dict(counter)
                    for elem_type, counter in story_data.items()
                }
                for story_id, story_data in elements_by_story_count.items()
            }
        }
    }


def evaluate_file(xml_file, element_library, story_label):
    """Evaluate Diversity for one file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        diversity_result = analyze_diversity(root, element_library, story_label)
        
        return {
            'file': str(xml_file),
            'diversity': diversity_result
        }
    
    except Exception as e:
        return {
            'file': str(xml_file),
            'error': f'Evaluation error: {str(e)}'
        }


def evaluate_all():
    """Evaluate all files."""
    all_results = []
    
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
                    print(f"    Evaluating: {xml_file.name}")
                    result = evaluate_file(xml_file, element_library, story_label)
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
                'diversity': {
                    'unique_usage_ratios': [],
                    'shannon_entropies': [],
                    'normalized_entropies': [],
                    'source_distributions': [],  # Multi-story only (usage-count share)
                    'unique_source_shares': [],  # Multi-story unique-element B_i/A
                }
            }
        
        summary[key]['count'] += 1
        
        # Diversity
        if 'diversity' in result:
            div = result['diversity']
            summary[key]['diversity']['unique_usage_ratios'].append(div.get('unique_usage_ratio', 0))
            summary[key]['diversity']['shannon_entropies'].append(div.get('shannon_entropy', 0))
            summary[key]['diversity']['normalized_entropies'].append(div.get('normalized_entropy', 0))
            
            # Source Distribution (multi-story only)
            if div.get('source_distribution') is not None:
                summary[key]['diversity']['source_distributions'].append(div.get('source_distribution'))
            if div.get('unique_source_share') is not None:
                summary[key]['diversity']['unique_source_shares'].append(div.get('unique_source_share'))
    
    # Compute averages
    for key in summary:
        data = summary[key]
        
        # Diversity averages
        div = data['diversity']
        div['avg_unique_usage_ratio'] = sum(div['unique_usage_ratios']) / len(div['unique_usage_ratios']) if div['unique_usage_ratios'] else 0
        div['avg_shannon_entropy'] = sum(div['shannon_entropies']) / len(div['shannon_entropies']) if div['shannon_entropies'] else 0
        div['avg_normalized_entropy'] = sum(div['normalized_entropies']) / len(div['normalized_entropies']) if div['normalized_entropies'] else 0
        
        # Average Source Distribution (multi-story only)
        if div['source_distributions']:
            # Compute average ratio per story
            story_ratios_agg = defaultdict(list)
            for sd in div['source_distributions']:
                for story_id, story_data in sd.items():
                    if story_id != 'by_type' and isinstance(story_data, dict):
                        story_ratios_agg[story_id].append(story_data.get('ratio', 0))
            
            avg_source_distribution = {}
            for story_id, ratios in story_ratios_agg.items():
                avg_source_distribution[story_id] = {
                    'avg_ratio': sum(ratios) / len(ratios) if ratios else 0
                }
            
            div['avg_source_distribution'] = avg_source_distribution
        else:
            div['avg_source_distribution'] = None

        # Unique-source share B_i/A:
        # for story j, values exist only when j is in the sample label;
        # column-j average is the conditional mean over those samples.
        if div.get('unique_source_shares'):
            story_ratio_lists = defaultdict(list)
            for us in div['unique_source_shares']:
                if not us:
                    continue
                for story_id, row in us.items():
                    if isinstance(row, dict) and 'ratio' in row:
                        story_ratio_lists[str(story_id)].append(row['ratio'])
            div['avg_unique_source_share'] = {
                sid: {
                    'avg_ratio': sum(vals) / len(vals) if vals else 0.0,
                    'n_files': len(vals),
                }
                for sid, vals in story_ratio_lists.items()
            }
        else:
            div['avg_unique_source_share'] = None

    # Save summary
    summary_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save CSV summary
    csv_file = EVAL_OUTPUT_DIR / f"summary_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(
            "Model,Dimension,Count,Unique Usage Ratio,Shannon Entropy,Normalized Entropy,"
            "Source Distribution (usage-weighted),Unique Source Share B/A per story\n"
        )
        for key, data in sorted(summary.items()):
            model = data['model']
            dimension = data['dimension']
            count = data['count']
            div = data['diversity']
            
            unique_ratio = div.get('avg_unique_usage_ratio', 0)
            shannon = div.get('avg_shannon_entropy', 0)
            normalized = div.get('avg_normalized_entropy', 0)
            
            # Format Source Distribution
            if div.get('avg_source_distribution'):
                sd_str = "; ".join(
                    [
                        f"Story{story_id}: {data['avg_ratio']:.3f}"
                        for story_id, data in sorted(
                            div['avg_source_distribution'].items(), key=lambda x: int(x[0])
                        )
                    ]
                )
            else:
                sd_str = "N/A (single story)"

            if div.get('avg_unique_source_share'):
                us_str = "; ".join(
                    [
                        f"Story{story_id}: {data['avg_ratio']:.3f}"
                        for story_id, data in sorted(
                            div['avg_unique_source_share'].items(), key=lambda x: int(x[0])
                        )
                    ]
                )
            else:
                us_str = "N/A (single story)"

            f.write(
                f"{model},{dimension},{count},{unique_ratio:.4f},{shannon:.4f},{normalized:.4f},"
                f'"{sd_str}","{us_str}"\n'
            )
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    evaluate_all()

