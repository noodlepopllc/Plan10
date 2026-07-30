#!/usr/bin/env python3
"""
Scene Registry Generator
Takes character images + background image, reads their metadata descriptions,
analyzes the visuals, and generates a structured JSON registry matching the
required schema.
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
                break
    
    # EXIF data
    if not metadata.get('description'):
        try:
            exif = img.getexif()
            if exif:
                # UserComment tag 37510, ImageDescription tag 270
                for tag_id in [37510, 270]:
                    if tag_id in exif:
                        val = exif[tag_id]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore')
                        if val and str(val).strip():
                            metadata['description'] = str(val).strip()
                            break
        except Exception:
            pass
    
    return metadata


def parse_json_response(text):
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first and last lines if they're fences
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    
    # Find JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    print(text)
    return json.loads(text)


def analyze_character(image_path, metadata_desc):
    """Analyze a character image and return structured biography dict."""
    prompt = f"""Analyze this character image and produce a structured biography.

KNOWN DESCRIPTION (from image metadata - treat as ground truth):
{metadata_desc}

Use the known description as the primary source. Only infer missing details by analyzing the image visually. If metadata conflicts with visual, prefer metadata.

Output STRICTLY as a JSON object with these exact fields:
{{
  "name": "string - character name (from metadata, or infer if missing)",
  "age": "string - estimated or stated age",
  "gender": "string",
  "race": "string (e.g., 'Human', 'Elf', 'Android', 'Half-elf')",
  "ethnicity_species": "string - specific ethnicity or species",
  "appearance": "Combined physical description (build, face shape, skin, facial features) - ethnicity-appropriate",
  "clothing": "Silhouette, material, and color",
  "hair": "Silhouette, color, and style",
  "distinctive_visual_markers": ["Unique visual trait 1", "Unique visual trait 2"],
  "movement_style": "Broad, observable physical traits suggesting how they move",
  "personality_traits": "1-2 filmable physical traits"
}}

Output ONLY the JSON object. No markdown fences, no commentary, no explanation."""
    
    result = AnalyzeImage(image_path, prompt)
    return parse_json_response(result['analysis'])


def analyze_background(image_path, metadata_desc):
    """Analyze background image and return setting + location with zones."""
    prompt = f"""Analyze this background/environment image and produce structured scene data.

KNOWN DESCRIPTION (from image metadata - treat as ground truth):
{metadata_desc}

Use the known description as the primary source. Only infer missing details by analyzing the image visually.

Output STRICTLY as a JSON object with this exact structure:
{{
  "setting": {{
    "room_form": "3-5 sentences describing overall form, major fixed structures, openings, ground material, lighting sources, architectural style, and spatial scale. MUST include exact time of day, sky condition, sun position, and external light color.",
    "time_of_day": "string (e.g., 'late afternoon', 'night', 'early morning')",
    "sky_condition": "string (e.g., 'clear blue', 'overcast grey', 'dark with stars')",
    "external_lighting": "string (e.g., 'warm golden sunlight from west', 'no natural light', 'cool blue twilight')"
  }},
  "location": {{
    "name": "string - descriptive name for this location",
    "architectural_shell": "3-5 sentences describing shape, fixed structures, openings, materials, lighting, and scale.",
    "zones": [
      {{
        "zone_name": "string (physical area name, e.g., 'Corner Table', 'Bar Counter')",
        "zone_definition": "3-5 sentences describing physical space: what part of location it occupies, fixed features, furniture, spatial relationship to other zones, what environmental elements are on left side, what environmental elements are on right side, and clear open floor space in the center/foreground. NO camera references, NO character names, NO character appearance, NO character actions, NO large foreground-blocking objects.",
        "purpose": "1-2 sentences describing functional purpose. NO story events.",
        "anchored_elements": [
          {{
            "name": "string",
            "material": "string",
            "position": "string (physical position in zone: left side, right side, background, far left, far right. MUST NOT be foreground/center if large)",
            "orientation": "string"
          }}
        ],
        "visible_background_elements": ["5-8 background elements visible in the full zone"]
      }}
    ]
  }}
}}

Generate 2-4 zones that logically divide the physical space. Each zone must be a distinct physical area.

Output ONLY the JSON object. No markdown fences, no commentary, no explanation."""
    
    result = AnalyzeImage(image_path, prompt)
    return parse_json_response(result['analysis'])


def build_registry(background_path, character_paths, output_path):
    """Build the complete registry JSON from images."""
    print(f"🔍 Analyzing background: {background_path}")
    bg_meta = read_image_metadata(background_path)
    bg_desc = bg_meta.get('description', 'No metadata available')
    print(f"   Metadata: {bg_desc[:80]}{'...' if len(bg_desc) > 80 else ''}")
    
    bg_data = analyze_background(background_path, bg_desc)
    
    registry = {
        "setting": bg_data["setting"],
        "biographies": [],
        "locations": [bg_data["location"]]
    }
    
    for i, char_path in enumerate(character_paths, 1):
        print(f"\n👤 Analyzing character {i}/{len(character_paths)}: {char_path}")
        char_meta = read_image_metadata(char_path)
        char_desc = char_meta.get('description', 'No metadata available')
        print(f"   Metadata: {char_desc[:80]}{'...' if len(char_desc) > 80 else ''}")
        
        bio = analyze_character(char_path, char_desc)
        registry["biographies"].append(bio)
        print(f"   ✓ {bio.get('name', 'Unknown')}")
    
    # Write output
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Registry written to: {output_path}")
    print(f"   Setting: {registry['setting']['time_of_day']} / {registry['setting']['sky_condition']}")
    print(f"   Location: {registry['locations'][0]['name']} ({len(registry['locations'][0]['zones'])} zones)")
    print(f"   Characters: {len(registry['biographies'])}")
    
    return registry


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene registry JSON from character and background images"
    )
    parser.add_argument('-B', '--background', type=str, required=True,
                        help="Background image path")
    parser.add_argument('-C', '--characters', type=str, nargs='+', required=True,
                        help="Character image paths (one or more)")
    parser.add_argument('-O', '--output', type=str, default='registry.json',
                        help="Output JSON file (default: registry.json)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.background).exists():
        print(f"❌ Background not found: {args.background}")
        sys.exit(1)
    for cp in args.characters:
        if not Path(cp).exists():
            print(f"❌ Character image not found: {cp}")
            sys.exit(1)
    
    build_registry(
        background_path=args.background,
        character_paths=args.characters,
        output_path=args.output
    )


if __name__ == "__main__":
    main()