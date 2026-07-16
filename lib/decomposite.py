#!/usr/bin/env python3
"""
Scene Decomposition Tool
Analyzes a scene, extracts character descriptions, and generates clean character sheets and background.
"""

import sys, os
sys.path.append('./lib')
from config import load_environ
load_environ()

ANIME = os.environ.get("ANIME", "False") != "False"
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if ANIME:
    from anime_gen import CreateCharacterSheet, CreateBackground, ImageGen
else:
    from image_gen import CreateCharacterSheet, CreateBackground, ImageGen

import argparse
import json
from pathlib import Path
from image_analysis import AnalyzeImage


def decompose_scene(input_image, output_dir, width=832, height=480, seed=42):
    """
    Decompose a scene into individual character sheets and background plate.
    
    Args:
        input_image: Path to scene image
        output_dir: Directory to save extracted assets
        width: Output width for final assets
        height: Output height for final assets
        seed: Random seed for generation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Analyzing scene: {input_image}")
    
    # Step 1: Analyze scene to identify characters and environment
    analysis_prompt = """Analyze this image and identify all distinct characters/people present and the environment.

For each character, provide:
1. POSITION: Where they are in the frame (left, center, right)
2. APPEARANCE: Detailed physical description (age range, gender presentation, ethnicity, hair color and style, body type)
3. CLOTHING: Detailed clothing description (top, bottom, shoes, accessories, colors, styles)
4. POSE: Current pose and orientation

For the environment, provide:
- LOCATION: Type of location (indoor/outdoor, specific setting)
- ATMOSPHERE: Lighting, time of day, mood, weather if applicable
- KEY ELEMENTS: Notable objects, furniture, architectural features

Output format:
CHARACTER_1:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [detailed description]
POSE: [description]

CHARACTER_2:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [detailed description]
POSE: [description]

[Continue for all characters]

ENVIRONMENT:
LOCATION: [description]
ATMOSPHERE: [description]
KEY ELEMENTS: [description]

TOTAL_CHARACTERS: [number]"""
    
    result = AnalyzeImage(input_image, analysis_prompt)
    analysis = result['analysis']
    print(analysis)
    
    # Parse character count
    char_count = 1
    for line in analysis.split('\n'):
        if 'TOTAL_CHARACTERS:' in line:
            try:
                char_count = int(line.split(':')[1].strip())
            except:
                pass
    
    print(f"\n✓ Found {char_count} character(s)")
    
    # Step 2: Generate character sheets for each character
    characters = []
    with ImageGen() as igen:
        for i in range(1, char_count + 1):
            print(f"\n🎨 Generating character sheet {i}...")
            
            # Extract character description from analysis
            char_desc = extract_character_description(analysis, i)
            
            char_output = output_dir / f"character_{i}.png"
            
            # Use CreateCharacterSheet to generate clean reference
            status = CreateCharacterSheet(
                prompt=char_desc,
                output=str(char_output),
                seed=seed + i,
                imagegen=igen
            )
            
            characters.append({
                'id': i,
                'path': str(char_output),
                'description': char_desc,
                'prompt_used': status.get('prompt', '')
            })
            
            print(f"  ✓ Saved: {char_output}")
    
    # Step 3: Generate clean background plate
    print(f"\n🏞️ Generating clean background plate...")
    
    # Extract environment description
    env_desc = extract_environment_description(analysis)
    
    bg_output = output_dir / "background.png"
    
    # Use CreateBackground to generate clean plate
    status = CreateBackground(
        prompt=env_desc,
        output=str(bg_output),
        seed=seed
    )
    
    print(f"  ✓ Saved: {bg_output}")
    
    # Step 4: Save manifest
    manifest = {
        'source': input_image,
        'background': str(bg_output),
        'environment_description': env_desc,
        'characters': characters,
        'dimensions': {'width': width, 'height': height},
        'analysis': analysis
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Decomposition complete!")
    print(f"   Background: {bg_output}")
    print(f"   Characters: {len(characters)}")
    print(f"   Manifest: {manifest_path}")
    
    return manifest


def extract_character_description(analysis, char_num):
    """Extract character description from analysis text."""
    lines = analysis.split('\n')
    char_lines = []
    in_char_section = False
    current_field = None
    
    for line in lines:
        line = line.strip()
        
        # Start of target character section
        if line.startswith(f'CHARACTER_{char_num}:'):
            in_char_section = True
            continue
        
        # End of character section (next character or environment)
        if in_char_section and (line.startswith('CHARACTER_') or line.startswith('ENVIRONMENT:') or line.startswith('TOTAL_CHARACTERS:')):
            break
        
        if in_char_section:
            if line.startswith('APPEARANCE:'):
                current_field = 'appearance'
                char_lines.append(line.split(':', 1)[1].strip())
            elif line.startswith('CLOTHING:'):
                current_field = 'clothing'
                char_lines.append(line.split(':', 1)[1].strip())
            elif current_field and line and not line.startswith('POSITION:') and not line.startswith('POSE:'):
                # Continuation line
                char_lines.append(line)
    
    return ' '.join(char_lines)


def extract_environment_description(analysis):
    """Extract environment description from analysis text."""
    lines = analysis.split('\n')
    env_lines = []
    in_env_section = False
    current_field = None
    
    for line in lines:
        line = line.strip()
        
        # Start of environment section
        if line.startswith('ENVIRONMENT:'):
            in_env_section = True
            continue
        
        # End of environment section
        if in_env_section and line.startswith('TOTAL_CHARACTERS:'):
            break
        
        if in_env_section:
            if line.startswith('LOCATION:'):
                current_field = 'location'
                env_lines.append(line.split(':', 1)[1].strip())
            elif line.startswith('ATMOSPHERE:'):
                current_field = 'atmosphere'
                env_lines.append(line.split(':', 1)[1].strip())
            elif line.startswith('KEY ELEMENTS:'):
                current_field = 'elements'
                env_lines.append(line.split(':', 1)[1].strip())
            elif current_field and line:
                # Continuation line
                env_lines.append(line)
    
    return ' '.join(env_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose scene into characters and background")
    parser.add_argument('-I', '--input', type=str, required=True, help="Input scene image")
    parser.add_argument('-O', '--output', type=str, required=True, help="Output directory")
    parser.add_argument('-W', '--width', type=int, default=832, help="Output width")
    parser.add_argument('-H', '--height', type=int, default=480, help="Output height")
    parser.add_argument('-S', '--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    decompose_scene(
        input_image=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed
    )