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
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if ANIME:
    from plan10.lib.anime_gen import CreateCharacterSheet, CreateBackground, ImageGen, add_metadata_loc
else:
    from plan10.lib.image_gen import CreateCharacterSheet, CreateBackground, ImageGen, add_metadata_loc

from plan10.lib.compositor import CompositeScene

import json
from pathlib import Path
from plan10.lib.image_analysis import AnalyzeImage


def analyze_scene(input_image: str, original_prompt: str = '', anime_mode: bool = False) -> dict:
    analysis_prompt = build_analysis_prompt(anime_mode)
    
    if original_prompt:
        analysis_prompt = f"""ORIGINAL IMAGE PROMPT (for context):
{original_prompt}

---

{analysis_prompt}

Use the original prompt to help identify characters and environment details that might be ambiguous."""
    
    result = AnalyzeImage(input_image, analysis_prompt, backend="")
    analysis = result['analysis']
    
    char_count = parse_character_count(analysis)
    
    print(f"✓ Found {char_count} character(s)")
    print(analysis)
    
    return {
        'analysis': analysis,
        'character_count': char_count
    }


def build_analysis_prompt(anime_mode: bool = False) -> str:
    """Build the analysis prompt based on mode."""
    base_prompt = """Analyze this image and identify up to 3 PRIMARY characters/people present.

CRITICAL GUIDELINES:
- Focus on FOREGROUND and PROMINENT characters only
- Ignore background characters, crowds, or people who are not clearly visible
- If there are more than 3 people, select only the 3 most prominent/foreground characters
- Count carefully - look for different positions, features, clothing, accessories

For EACH character (up to 3), provide:
1. POSITION: Where they are in the frame (left, center, right, foreground, background)
2. APPEARANCE: Detailed physical description (age, gender, ethnicity, race, hair color/style, eye color, distinguishing features)
3. CLOTHING: COMPLETE outfit description including:
   - Top (color, style, fit, material, pattern)
   - Bottom (color, style, fit, material, pattern - INFER if not visible)
   - Shoes/footwear (INFER if not visible)
   - Accessories (jewelry, bags, hats, etc.)
   - Cleavage, Torn, Burnt, Form Fitting, etc
   - Parts of the body that are exposed
4. POSE: Current pose and orientation
5. ART STYLE: (photorealistic, anime, etc)
6. THEME: (gothic, cyber punk, sci-fi, western, modern, etc.)

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
ART STYLE: [Art Style]
THEME: [Theme]

CHARACTER_2:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [complete outfit - visible AND inferred]
POSE: [description]
ART STYLE: [Art Style]
THEME: [Theme]

CHARACTER_3:
POSITION: [location]
APPEARANCE: [detailed description]
CLOTHING: [complete outfit - visible AND inferred]
POSE: [description]
ART STYLE: [Art Style]
THEME: [Theme]

[Only include CHARACTER_2 and CHARACTER_3 if they exist and are prominent]

ENVIRONMENT:
LOCATION: [description]
ATMOSPHERE: [description]
KEY ELEMENTS: [description]
ART STYLE: [Art Style]
THEME: [Theme]

TOTAL_CHARACTERS: [actual count, maximum 3]"""
    
    if anime_mode:
        return f"""[ANIME MODE]
        
{base_prompt}

ANIME-SPECIFIC DETECTION:
- Characters may share similar art styles but are DISTINCT individuals
- Look for differences in hair color, eye color, accessories, clothing patterns
- Pay attention to spatial positioning - characters in different locations are separate people
- Do NOT merge similar-looking characters into one description
- Focus on main characters, ignore background extras"""
    
    return base_prompt


def parse_character_count(analysis: str) -> int:
    """Extract character count from analysis text, capped at 2."""
    for line in analysis.split('\n'):
        if 'TOTAL_CHARACTERS:' in line:
            try:
                detected_count = int(line.split(':')[1].strip())
                count = min(detected_count, 2)
                if detected_count > 2:
                    print(f"⚠️ Detected {detected_count} characters, but compositor only supports 2. Using first 2.")
                return count
            except:
                pass
    return 1


def generate_character_sheets(
    analysis: str, 
    char_count: int, 
    output_dir: Path, 
    seed: int
) -> list:
    """
    Generate character sheets for each detected character.
    
    Returns:
        list of dicts with character metadata
    """
    characters = []
    
    with ImageGen() as igen:
        for i in range(1, char_count + 1):
            print(f"\n🎨 Generating character sheet {i}...")
            
            char_desc = extract_character_description(analysis, i)
            char_output = output_dir / f"character_{i}.png"
            
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
    
    return characters

def generate_portraits(
    analysis: str, 
    char_count: int, 
    output_dir: Path, 
    seed: int
) -> list:
    """
    Generate character sheets for each detected character.
    
    Returns:
        list of dicts with character metadata
    """
    portraits = []
    
    with ImageGen() as igen:
        for i in range(1, char_count + 1):
            print(f"\n🎨 Generating character sheet {i}...")
            
            char_desc = extract_character_description(analysis, i)
            char_output = output_dir / f"portrait_{i}.png"
            
            status = status = igen.generate(
                prompt=f'A studio portrait of {char_desc}',
                output=str(char_output),
                width=1024,
                height=1024,
                seed=seed + i,
            )
            
            portraits.append({
                'id': i,
                'path': str(char_output),
                'description': status['description'],
                'prompt_used': status.get('prompt', '')
            })
            
            print(f"  ✓ Saved: {char_output}")
    
    return portraits





def generate_background(
    input_image: str, 
    analysis: str,
    output_dir: Path, 
    seed: int
) -> dict:
    """
    Generate a clean background plate by compositing and removing people.
    
    Returns:
        dict with background metadata
    """
    print(f"\n🏞️ Generating clean background plate...")
    
    bg_tmp = output_dir / "tmp_background.png"
    bg_output = output_dir / "background.png"

    env_desc = extract_environment_description(analysis)

    # Inject the environment description so the model knows what to draw in the gaps
    edit_prompt = f"remove people from image. preserve the background environment exactly: {env_desc}. clean background plate, highly detailed background, no people."

    tmp = Image.open(input_image)
    
    EditImage(
        prompt=edit_prompt,
        images=[input_image],
        output=str(bg_output),
        width=tmp.width,
        height=tmp.height
    )
    
    add_metadata_loc(str(bg_output))
    
    print(f"  ✓ Saved: {bg_output}")
    
    return {
        'path': str(bg_output),
        'description': 'Clean background plate with people removed'
    }


def save_manifest(
    source_image: str,
    background: dict,
    characters: list,
    portraits: list,
    analysis: str,
    output_dir: Path
) -> Path:
    """Save the decomposition manifest to JSON."""
    manifest = {
        'source': source_image,
        'background': background['path'],
        'environment_description': background['description'],
        'characters': characters,
        'portraits': portraits,
        'analysis': analysis
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def decompose_scene(input_image: str, prompt: str, output_dir: str, seed: int = 42) -> dict:
    """
    Decompose a scene into individual character sheets and background plate.
    
    This is the orchestrator function. It coordinates the specialized functions
    but doesn't do the actual work itself.
    
    Args:
        input_image: Path to scene image
        output_dir: Directory to save extracted assets
        seed: Random seed for generation
        
    Returns:
        dict with paths to generated assets and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Analyzing scene: {input_image}")
    
    # Step 1: Analyze the scene
    scene_data = analyze_scene(input_image, original_prompt=prompt, anime_mode=ANIME)
    analysis = scene_data['analysis']
    char_count = scene_data['character_count']
    
    # Step 2: Generate character sheets
    characters = generate_character_sheets(
        analysis=analysis,
        char_count=char_count,
        output_dir=output_dir,
        seed=seed
    )

    portraits = generate_portraits(
        analysis=analysis,
        char_count=char_count,
        output_dir=output_dir,
        seed=seed
    )
    
    # Step 3: Generate background
    background = generate_background(
        input_image=input_image,
        analysis=analysis,
        output_dir=output_dir,
        seed=seed
    )
    
    # Step 4: Save manifest
    manifest_path = save_manifest(
        source_image=input_image,
        background=background,
        characters=characters,
        portraits=portraits,
        analysis=analysis,
        output_dir=output_dir
    )
    
    print(f"\n✅ Decomposition complete!")
    print(f"   Background: {background['path']}")
    print(f"   Characters: {len(characters)}")
    print(f"   Manifest: {manifest_path}")
    
    return {
        'background': background['path'],
        'characters': characters,
        'manifest': str(manifest_path)
    }


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
    from plan10.lib.util import extract_frame
    from plan10.lib.image_gen import prompt_metadata
    import argparse
    parser = argparse.ArgumentParser(description="Decompose scene into characters and background")
    parser.add_argument('-I', '--input', type=str, required=True, help="Input scene image")
    parser.add_argument('-O', '--output', type=str, required=True, help="Output directory")
    parser.add_argument('-S', '--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    _, image = extract_frame(args.input, WIDTH, HEIGHT, 'first_frame.png', False)

    original_prompt = prompt_metadata(image)
    
    decompose_scene(
        input_image=image,
        prompt=original_prompt,
        output_dir=args.output,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
