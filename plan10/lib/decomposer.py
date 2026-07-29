#!/usr/bin/env python3
"""
Scene Decomposition Tool
Analyzes a scene, extracts character descriptions, and generates clean character sheets and background.
"""

import sys, os
from plan10.lib.config import load_environ
os.environ['BATCH'] = 'False'
load_environ()
from PIL import Image
from plan10.lib.image_edit import EditImage

ANIME = os.environ.get("ANIME", "False") != "False"
SEED = int(os.environ.get("SEED", "-1"))

if ANIME:
    from plan10.lib.anime_gen import CreateCharacterSheet, CreateBackground, ImageGen
else:
    from plan10.lib.image_gen import CreateCharacterSheet, CreateBackground, ImageGen

from plan10.lib.compositor import CompositeScene

import argparse
import json
from pathlib import Path
from plan10.lib.image_analysis import AnalyzeImage


def decompose_scene(input_image, output_dir, seed=42):
    """
    Decompose a scene into individual character sheets and background plate.
    
    Args:
        input_image: Path to scene image
        output_dir: Directory to save extracted assets
        seed: Random seed for generation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Analyzing scene: {input_image}")
    
    # Step 1: Analyze scene to identify characters and environment
    analysis_prompt = """Analyze this image and identify up to 3 PRIMARY characters/people present.

CRITICAL GUIDELINES:
- Focus on FOREGROUND and PROMINENT characters only
- Ignore background characters, crowds, or people who are not clearly visible
- If there are more than 3 people, select only the 3 most prominent/foreground characters
- Count carefully - look for different positions, features, clothing, accessories

For EACH character (up to 3), provide:
1. POSITION: Where they are in the frame (left, center, right, foreground, background)
2. APPEARANCE: Detailed physical description (age, gender, hair color/style, eye color, distinguishing features)
3. CLOTHING: COMPLETE outfit description including:
   - Top (color, style, fit, material)
   - Bottom (color, style, fit - INFER if not visible)
   - Shoes/footwear (INFER if not visible)
   - Accessories (jewelry, bags, hats, etc.)
4. POSE: Current pose and orientation

For the environment:
- LOCATION: Type of location
- ATMOSPHERE: Lighting, time of day, mood
- KEY ELEMENTS: Notable objects, furniture, architectural features

Output format:
CHARACTER_1:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [complete outfit - visible AND inferred]
POSE: [description]

CHARACTER_2:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [complete outfit - visible AND inferred]
POSE: [description]

CHARACTER_3:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [complete outfit - visible AND inferred]
POSE: [description]

[Only include CHARACTER_2 and CHARACTER_3 if they exist and are prominent]

ENVIRONMENT:
LOCATION: [description]
ATMOSPHERE: [description]
KEY ELEMENTS: [description]

TOTAL_CHARACTERS: [actual count, maximum 3]"""

    if ANIME:
        analysis_prompt = f"""[ANIME MODE]
        
    {analysis_prompt}

    ANIME-SPECIFIC DETECTION:
    - Characters may share similar art styles but are DISTINCT individuals
    - Look for differences in hair color, eye color, accessories, clothing patterns
    - Pay attention to spatial positioning - characters in different locations are separate people
    - Do NOT merge similar-looking characters into one description
    - Focus on main characters, ignore background extras"""
        
    result = AnalyzeImage(input_image, analysis_prompt)
    analysis = result['analysis']
    print(analysis)
    
    # Parse character count (cap at 2 for compositor compatibility)
    char_count = 1
    for line in analysis.split('\n'):
        if 'TOTAL_CHARACTERS:' in line:
            try:
                detected_count = int(line.split(':')[1].strip())
                char_count = min(detected_count, 2)  # Cap at 2
                if detected_count > 2:
                    print(f"⚠️ Detected {detected_count} characters, but compositor only supports 2. Using first 2.")
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
                'description': status['description'],
                'prompt_used': status.get('prompt', '')
            })
            
            print(f"  ✓ Saved: {char_output}")
        
    # Step 3: Generate clean background plate
    print(f"\n🏞️ Generating clean background plate...")
    
    # Extract environment description
    #env_desc = extract_environment_description(analysis)
    
    bg_tmp = output_dir / "tmp_background.png"
    
    # Use CreateBackground to generate clean plate
    status = CompositeScene(
        background_path=input_image,
        characters=[],
        output=str(bg_tmp),
        seed=seed
    )
    
    bg_output = output_dir / "background.png"

    status = EditImage(prompt='remove people from image', images=[str(bg_tmp)], output=str(bg_output))
    
    print(f"  ✓ Saved: {bg_output}")
    
    # Step 4: Save manifest
    manifest = {
        'source': input_image,
        'background': str(bg_output),
        'environment_description': status['description'],
        'characters': characters,
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

def main():
    parser = argparse.ArgumentParser(description="Decompose scene into characters and background")
    parser.add_argument('-I', '--input', type=str, required=True, help="Input scene image")
    parser.add_argument('-O', '--output', type=str, required=True, help="Output directory")
    parser.add_argument('-S', '--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    decompose_scene(
        input_image=args.input,
        output_dir=args.output,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
