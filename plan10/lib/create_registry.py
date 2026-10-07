#!/usr/bin/env python3
"""
Scene Registry & Context Generator
Takes character images + background image, reads their metadata descriptions,
analyzes the visuals, and generates structured JSON (registry.json, context.json, or both).
"""

import sys, os, argparse, json
from pathlib import Path
from PIL import Image
from plan10.lib.config import load_environ
os.environ['BATCH'] = 'False'
load_environ()
from plan10.lib.image_analysis import AnalyzeImage


def read_image_metadata(image_path):
    """Extract description/metadata from image file.
    
    Checks PNG text chunks, EXIF UserComment, and common metadata fields.
    """
    img = Image.open(image_path)
    metadata = {}
    
    # PNG text chunks (info dict)
    if hasattr(img, 'info') and img.info:
        for key in ['description', 'Description', 'comment', 'Comment', 
                    'prompt', 'Prompt', 'parameters']:
            if key in img.info and img.info[key]:
                metadata['description'] = str(img.info[key])
                # Also save as 'prompt' specifically for context.json compatibility
                metadata['prompt'] = str(img.info[key])
                break
    
    # EXIF data fallback
    if not metadata.get('description'):
        try:
            exif = img.getexif()
            if exif:
                for tag_id in [37510, 270]: # UserComment, ImageDescription
                    if tag_id in exif:
                        val = exif[tag_id]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        if val and str(val).strip():
                            metadata['description'] = str(val).strip()
                            metadata['prompt'] = str(val).strip()
                            break
        except Exception:
            pass
    
    img.close()
    return metadata


def parse_json_response(text):
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    
    return json.loads(text)


def analyze_character(image_path, metadata_desc):
    """Analyze a character image and return structured biography dict."""
    prompt = f"""Analyze this character image and produce a structured biography.

KNOWN DESCRIPTION (from image metadata - treat as ground truth):
{metadata_desc}

Use the known description as the primary source. Only infer missing details by analyzing the image visually.

Output STRICTLY as a JSON object with these exact fields:
{{
  "name": "string - character name (from metadata, or infer if missing)",
  "age": "string - estimated or stated age",
  "gender": "string",
  "race": "string (e.g., 'Human', 'Elf', 'Android')",
  "ethnicity_species": "string - specific ethnicity or species",
  "appearance": "Combined physical description (build, face shape, skin, facial features)",
  "clothing": "Silhouette, material, and color",
  "hair": "Silhouette, color, and style",
  "distinctive_visual_markers": ["Unique visual trait 1", "Unique visual trait 2"],
  "movement_style": "Broad, observable physical traits suggesting how they move",
  "personality_traits": "1-2 filmable physical traits"
}}

Output ONLY the JSON object. No markdown fences, no commentary."""
    
    result = AnalyzeImage(image_path, prompt)
    return parse_json_response(result['analysis'])


def analyze_background(image_path, metadata_desc):
    """Analyze background image and return setting + location with zones."""
    prompt = f"""Analyze this background/environment image and produce structured scene data.

KNOWN DESCRIPTION (from image metadata - treat as ground truth):
{metadata_desc}

CORE RULES:
- NO camera references (no "camera", "angle", "frame", "shot", "lens")
- NO character names, appearance, clothing, or actions
- Describe ONLY the physical environment
- Large furniture MUST NOT dominate the foreground or block the central area
- The center and foreground of every zone MUST have clear, open floor space

Output STRICTLY as a JSON object with this exact structure:
{{
  "setting": {{
    "room_form": "3-5 sentences describing overall form, major fixed structures, openings, ground material, lighting sources, architectural style, and spatial scale. MUST include exact time of day, sky condition, sun position, and external light color.",
    "time_of_day": "string",
    "sky_condition": "string",
    "external_lighting": "string"
  }},
  "location": {{
    "name": "string - descriptive name for this location",
    "architectural_shell": "3-5 sentences describing shape, fixed structures, openings, materials, lighting, and scale.",
    "zones": [
      {{
        "zone_name": "string (physical area name, e.g., 'Corner Table', 'Bar Counter')",
        "zone_definition": "3-5 sentences: what part of location, fixed features with orientation relative to walls/windows, left side elements, right side elements, clear center/foreground floor space.",
        "purpose": "1-2 sentences describing functional purpose.",
        "anchored_elements": [
          {{
            "name": "string",
            "material": "string",
            "position": "string (left side, right side, background)",
            "orientation": "string (describe direction relative to walls/windows)"
          }}
        ],
        "visible_background_elements": ["5-8 specific background elements"]
      }}
    ]
  }}
}}

Output ONLY the JSON object. No markdown fences, no commentary."""
    
    result = AnalyzeImage(image_path, prompt)
    return parse_json_response(result['analysis'])


def build_registry(bg_data, char_bios, output_path):
    """Build and save the registry.json structure."""
    registry = {
        "setting": bg_data["setting"],
        "biographies": char_bios,
        "locations": [bg_data["location"]]
    }
    
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Registry written to: {output_path}")
    print(f"   Setting: {registry['setting']['time_of_day']} / {registry['setting']['sky_condition']}")
    print(f"   Location: {registry['locations'][0]['name']} ({len(registry['locations'][0]['zones'])} zones)")
    print(f"   Characters: {len(registry['biographies'])}")
    
    return registry


def build_context(background_path, character_paths, bg_data, char_bios, output_path):
    """Build and save the context.json structure."""
    assets = {}
    
    # 1. Add Characters
    for char_path, bio in zip(character_paths, char_bios):
        name = bio.get('name', 'UNKNOWN').upper().replace(' ', '_').replace('-', '_')
        key = f"CHAR_{name}"
        
        meta = read_image_metadata(char_path)
        prompt = meta.get('prompt', meta.get('description', ''))
        
        desc = f"{bio.get('appearance', '')} {bio.get('clothing', '')} {bio.get('hair', '')}".strip()
        
        assets[key] = {
            "path": str(Path(char_path).resolve()),
            "type": "image",
            "description": desc,
            "metadata": {
                "tool": "create_character_sheet",
                "prompt": prompt
            }
        }
        
    # 2. Add Background
    loc_name = bg_data['location']['name'].upper().replace(' ', '_').replace('-', '_')
    bg_key = f"{loc_name}_BACKGROUND"
    
    bg_meta = read_image_metadata(background_path)
    bg_prompt = bg_meta.get('prompt', bg_meta.get('description', ''))
    
    assets[bg_key] = {
        "path": str(Path(background_path).resolve()),
        "type": "image",
        "description": bg_data['setting']['room_form'],
        "metadata": {
            "tool": "create_background",
            "prompt": bg_prompt
        }
    }
    
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"assets": assets}, f, indent=2)
        
    print(f"\n✅ Context written to: {output_path}")
    print(f"   Assets generated: {list(assets.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene registry and/or context JSON from character and background images"
    )
    parser.add_argument('-B', '--background', type=str, required=True,
                        help="Background image path")
    parser.add_argument('-C', '--characters', type=str, nargs='+', required=True,
                        help="Character image paths (one or more)")
    parser.add_argument('-O', '--output', type=str, default='.',
                        help="Output directory or specific file path (default: current directory)")
    parser.add_argument('--format', type=str, choices=['registry', 'context', 'both'], default='both',
                        help="Output format: 'registry', 'context', or 'both' (default: both)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.background).exists():
        print(f"❌ Background not found: {args.background}")
        sys.exit(1)
    for cp in args.characters:
        if not Path(cp).exists():
            print(f"❌ Character image not found: {cp}")
            sys.exit(1)
    
    # Determine output paths
    out_path = Path(args.output)
    if args.format == 'both':
        # If 'both', treat -O as a directory
        out_dir = out_path if out_path.suffix == '' else out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        reg_path = out_dir / 'registry.json'
        ctx_path = out_dir / 'context.json'
    else:
        # If single format, treat -O as the exact file path
        if out_path.suffix == '':
            out_path = out_path / f"{args.format}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path = out_path if args.format == 'registry' else None
        ctx_path = out_path if args.format == 'context' else None

    print(f"🔍 Analyzing background: {args.background}")
    bg_meta = read_image_metadata(args.background)
    bg_desc = bg_meta.get('description', 'No metadata available')
    print(f"   Metadata: {bg_desc[:80]}{'...' if len(bg_desc) > 80 else ''}")
    
    bg_data = analyze_background(args.background, bg_desc)
    
    char_bios = []
    for i, char_path in enumerate(args.characters, 1):
        print(f"\n👤 Analyzing character {i}/{len(args.characters)}: {char_path}")
        char_meta = read_image_metadata(char_path)
        char_desc = char_meta.get('description', 'No metadata available')
        print(f"   Metadata: {char_desc[:80]}{'...' if len(char_desc) > 80 else ''}")
        
        bio = analyze_character(char_path, char_desc)
        char_bios.append(bio)
        print(f"   ✓ {bio.get('name', 'Unknown')}")
    
    # Generate requested formats
    if args.format in ['registry', 'both'] and reg_path:
        build_registry(bg_data, char_bios, reg_path)
        
    if args.format in ['context', 'both'] and ctx_path:
        build_context(args.background, args.characters, bg_data, char_bios, ctx_path)


if __name__ == "__main__":
    main()