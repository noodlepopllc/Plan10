#!/usr/bin/env python3
"""
Scene Registry & Context Generator
Takes character images + background image, reads their metadata descriptions,
analyzes the visuals, and generates structured JSON (registry.json, context.json, or both).
"""

import sys, os, argparse, json, re
from pathlib import Path
from PIL import Image
from plan10.lib.config import load_environ
os.environ['BATCH'] = 'False'
load_environ()
from plan10.lib.image_analysis import AnalyzeImage

import argparse

def parse_header(header_text):
    characters = {}
    zones = {}
    
    current_section = None
    current_name = None
    current_filepath = None  # <-- was current_filename, now consistent
    current_prompts = []
    current_location = None
    
    def save_current():
        nonlocal current_name, current_filepath, current_prompts, current_location  # <-- fixed
        if current_name and current_filepath:  # <-- fixed
            if current_section == "characters":
                characters[current_name] = {
                    "filepath": current_filepath,
                    "prompt": "\n".join(current_prompts) if current_prompts else ""
                }
            elif current_section == "zones":
                zones[current_name] = {
                    "filepath": current_filepath,
                    "location": current_location or "UNKNOWN",
                    "prompt": "\n".join(current_prompts) if current_prompts else ""
                }
        current_name = None
        current_filepath = None  # <-- fixed
        current_prompts = []
        current_location = None
    
    for line in header_text.split("\n"):
        s = line.strip()
        if not s:
            continue
        
        upper = s.upper().lstrip("#").strip()
        if upper in ("CHARACTERS", "CHARACTER"):
            save_current()
            current_section = "characters"
            continue
        elif upper in ("ZONES", "ZONE", "LOCATIONS"):
            save_current()
            current_section = "zones"
            continue
        
        if s.startswith("#"):
            continue
        
        if s.startswith(">>"):
            prompt_text = s[2:].strip()
            if prompt_text.upper().startswith("LOCATION:"):
                current_location = prompt_text[9:].strip()
            else:
                current_prompts.append(prompt_text)
            continue
        
        if ":" in s and current_section:
            save_current()
            name, filepath = s.split(":", 1)
            current_name = name.strip().upper()
            current_filepath = filepath.strip()
    
    save_current()
    
    return characters, zones


def build_name_normalizer(names):
    if not names:
        return lambda t: t
    
    sorted_names = sorted(names, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b",
        re.IGNORECASE,
    )

    def normalize(text):
        if not text:
            return text
        return pattern.sub(lambda m: m.group(1).upper(), text)

    return normalize


def parse_script_txt(script_path):
    """
    Parses a self-contained script file with embedded header.
    No registry or context required.
    
    Returns:
        beats: list of beat dicts
        characters: dict of {name: {filename, prompt}}
        zones: dict of {name: {filename, location, prompt}}
    """
    raw = Path(script_path).read_text(encoding="utf-8")
    
    # Split header from body at separator line
    separator_re = re.compile(r"^[-=_]{3,}\s*$", re.MULTILINE)
    match = separator_re.search(raw)
    
    if match:
        header_text = raw[:match.start()]
        body_text = raw[match.end():]
    else:
        header_text = ""
        body_text = raw
    
    # Parse header
    characters, zones = parse_header(header_text)
    
    # Build lookup structures
    char_names = list(characters.keys())
    zone_names = list(zones.keys())
    
    normalize = build_name_normalizer(char_names)
    upper_names = set(n.upper() for n in char_names)
    
    # Build regex patterns
    if upper_names:
        name_alt = "|".join(re.escape(n) for n in sorted(upper_names, key=len, reverse=True))
        char_re = re.compile(rf"^({name_alt})\s*(?:\(([^)]+)\))?\s*$")
    else:
        char_re = None
    
    zone_re = re.compile(r"\[ZONE:\s*(.+?)\]")
    dialog_re = re.compile(r'^"(.*)"$')
    
    # Parse body
    lines = [ln.rstrip() for ln in body_text.split("\n")]
    
    beats = []
    beat = None
    current_char = None
    
    for line in lines:
        s = line.strip()
        if not s:
            continue
        
        # Zone header
        m = zone_re.match(s)
        if m:
            if beat:
                beats.append(beat)
            zone_name = m.group(1).strip().upper()
            zone_info = zones.get(zone_name, {})
            beat = {
                "zone": zone_name,
                "location": zone_info.get("location"),
                "zone_key": zone_name.replace(" ", "_"),
                "summary": None,
                "active_characters": [],
                "passive_characters": [],
            }
            current_char = None
            continue
        
        # Summary (>> line in body)
        if s.startswith(">>"):
            if beat:
                beat["summary"] = s[2:].strip()
            continue
        
        # Character name with optional delivery
        if char_re:
            m = char_re.match(s)
            if m and beat:
                current_char = {
                    "name": m.group(1).strip(),
                    "delivery": m.group(2).strip() if m.group(2) else None,
                    "dialog": None,
                    "action": None,
                }
                beat["active_characters"].append(current_char)
                continue
        
        # Dialogue
        m = dialog_re.match(s)
        if m and current_char:
            current_char["dialog"] = m.group(1).strip()
            continue
        
        # Action (line after character name that isn't dialogue)
        if current_char and not current_char["action"]:
            current_char["action"] = normalize(s)
            continue
    
    if beat:
        beats.append(beat)
    
    # Normalize summaries
    for beat in beats:
        if beat["summary"]:
            beat["summary"] = normalize(beat["summary"])
    
    # Passive character detection
    for beat in beats:
        active_set = {c["name"] for c in beat["active_characters"]}
        passive_by_name = {}
        
        # 1. Scan summary
        if beat["summary"]:
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in beat["summary"]:
                    passive_by_name[scan_name] = {
                        "name": scan_name,
                        "source": "summary",
                    }
        
        # 2. Scan dialog lines
        for char in beat["active_characters"]:
            if not char["dialog"]:
                continue
            dialog_upper = char["dialog"].upper()
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in dialog_upper:
                    existing = passive_by_name.get(scan_name)
                    if existing is None or existing["source"] == "summary":
                        passive_by_name[scan_name] = {
                            "name": scan_name,
                            "source": "dialog",
                            "mentioned_by": char["name"],
                        }
        
        # 3. Scan action lines
        for char in beat["active_characters"]:
            if not char["action"]:
                continue
            action_upper = char["action"].upper()
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in action_upper:
                    existing = passive_by_name.get(scan_name)
                    if existing is None or existing["source"] == "summary":
                        passive_by_name[scan_name] = {
                            "name": scan_name,
                            "source": "action",
                            "mentioned_by": char["name"],
                        }
        
        beat["passive_characters"] = list(passive_by_name.values())
    
    return beats, characters, zones

def infer_location_prefix(zone_keys):
    """
    Given a list of zone keys (without _BACKGROUND suffix), 
    find the longest common prefix that represents the location.
    """
    if len(zone_keys) <= 1:
        return "UNKNOWN"
    
    split_keys = [k.split('_') for k in zone_keys]
    common_prefix = []
    
    for parts in zip(*split_keys):
        if len(set(parts)) == 1:
            common_prefix.append(parts[0])
        else:
            break
    
    if not common_prefix:
        return "UNKNOWN"
    
    return '_'.join(common_prefix)

def build_header_from_context_only(context_path: str):
    """
    Builds the header lines from context.json alone (no registry).
    Returns list of lines.
    """
    context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    assets = context.get('assets', {})
    
    # 1. Extract characters from CHAR_* keys
    characters = {}
    for key, asset in assets.items():
        if key.startswith('CHAR_'):
            name = key[5:].upper()
            filename = Path(asset['path']).name
            prompt = asset.get('metadata', {}).get('prompt', '')
            characters[name] = {
                'filename': filename,
                'prompt': prompt
            }
    
    # 2. Extract zones from *_BACKGROUND keys
    raw_zone_keys = []
    zones_raw = {}
    for key, asset in assets.items():
        if key.endswith('_BACKGROUND'):
            zone_key = key[:-len('_BACKGROUND')]
            raw_zone_keys.append(zone_key)
            zones_raw[zone_key] = {
                'filename': Path(asset['path']).name,
                'prompt': asset.get('metadata', {}).get('prompt', '')
            }
    
    # 3. Infer location from common prefix
    location_prefix = infer_location_prefix(raw_zone_keys)
    
    # 4. Split each zone key into (location, zone_name)
    zones = {}
    for zone_key, data in zones_raw.items():
        if location_prefix == "UNKNOWN":
            zone_name = zone_key
            zone_location = "UNKNOWN"
        else:
            if zone_key.startswith(location_prefix + '_'):
                zone_name = zone_key[len(location_prefix) + 1:]
                zone_location = location_prefix.replace('_', ' ')
            else:
                zone_name = zone_key
                zone_location = "UNKNOWN"
        
        zones[zone_name] = {
            'filename': data['filename'],
            'prompt': data['prompt'],
            'location': zone_location
        }
    
    # 5. Build the header lines
    lines = ["# CHARACTERS"]
    lines.append("# Format: NAME: filename.png")
    lines.append("# >> lines contain the generation prompt for on-demand asset creation")
    
    for name, data in characters.items():
        lines.append(f"{name}: {data['filename']}")
        if data['prompt']:
            lines.append(f">> {data['prompt']}")
    
    lines.append("\n# ZONES")
    lines.append("# Format: ZONE_NAME: filename.png")
    lines.append("# >> Location: <location name>")
    lines.append("# >> <generation prompt>")
    
    for zone_name, data in zones.items():
        display_name = zone_name.replace('_', ' ')
        lines.append(f"{display_name}: {data['filename']}")
        lines.append(f">> Location: {data['location']}")
        if data['prompt']:
            lines.append(f">> {data['prompt']}")
    
    return lines, characters, zones

def build_header_from_context_and_registry(context_path: str, registry_path: str):
    """
    Builds the header lines from context.json + registry.json.
    Returns list of lines.
    """
    context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    registry = json.loads(Path(registry_path).read_text(encoding='utf-8'))
    
    # 1. Extract Character Names and Descriptions from Registry
    char_registry = {}
    for bio in registry.get("biographies", []):
        name = bio.get("name", "UNKNOWN").strip().upper()
        desc = bio.get("appearance", bio.get("description", f"A character named {name}"))
        char_registry[name] = desc

    # 2. Extract Zone Names and Location Names from Registry
    zone_registry = {}
    for loc in registry.get("locations", []):
        loc_name = loc.get("name", "UNKNOWN")
        for zone in loc.get("zones", []):
            zname = zone.get("zone_name", "UNKNOWN_ZONE").strip()
            zone_registry[zname] = {
                "location": loc_name
            }

    # 3. Map Registry Names to Context Assets
    def canonical_key(location_name: str, zone_name: str) -> str:
        key = f"{location_name}_{zone_name}"
        key = key.replace(' ', '_').replace('/', '_').replace('"', '').upper()
        return f"{key}_BACKGROUND"
    
    char_assets = {}
    char_prompts = {}
    for key, asset in context['assets'].items():
        if key.startswith('CHAR_'):
            name = key[5:].upper()
            char_assets[name] = asset['path']
            char_prompts[name] = asset.get('metadata', {}).get('prompt', '')

    zone_assets = {}
    zone_prompts = {}
    for key, asset in context['assets'].items():
        if key.endswith('_BACKGROUND'):
            zone_assets[key] = asset['path']
            zone_prompts[key] = asset.get('metadata', {}).get('prompt', '')

    # 4. Build the header lines
    lines = ["# CHARACTERS"]
    lines.append("# Format: NAME: filename.png")
    lines.append("# >> lines contain the generation prompt for on-demand asset creation")
    
    for name, desc in char_registry.items():
        filename = char_assets.get(name)
        
        if not filename:
            slug = name.replace(' ', '_').replace('/', '_').replace('"', '').upper()
            filename = f"placeholder_{slug}.png"
            
        lines.append(f"{name}: {filename}")
        prompt = char_prompts.get(name, desc)
        lines.append(f">> {prompt}")

    lines.append("\n# ZONES")
    lines.append("# Format: ZONE_NAME: filename.png")
    lines.append("# >> Location: <location name>")
    lines.append("# >> <generation prompt>")
    
    for zname, zone_data in zone_registry.items():
        canonical = canonical_key(zone_data['location'], zname)
        filename = zone_assets.get(canonical)
        
        if not filename:
            slug = zname.replace(' ', '_').replace('/', '_').replace('"', '').upper()
            filename = f"placeholder_{slug}_BACKGROUND.png"
            
        lines.append(f"{zname}: {filename}")
        lines.append(f">> Location: {zone_data['location']}")
        
        prompt = zone_prompts.get(canonical, '')
        if prompt:
            lines.append(f">> {prompt}")

    return lines, char_registry, zone_registry

def build_template_body(characters, zones):
    """Builds the separator and template beat body."""
    lines = []
    lines.append("=" * 50)  # <-- removed leading/trailing \n
    
    first_zone = list(zones.keys())[0] if zones else "UNKNOWN_ZONE"
    lines.append(f"[ZONE: {first_zone.replace('_', ' ')}]")
    lines.append(">> [Write your beat summary here.]")
    
    first_char = list(characters.keys())[0] if characters else "CHAR1"
    lines.append(f"{first_char}")
    lines.append('"[Write dialogue here]"')
    lines.append("They perform a specific action.")
    
    return lines


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
    parser.add_argument('-B', '--background', type=str, required=False,
                        help="Background image path")
    parser.add_argument('-C', '--characters', type=str, nargs='+', required=False,
                        help="Character image paths (one or more)")
    parser.add_argument('-O', '--output', type=str, default='.',
                        help="Output directory or specific file path (default: current directory)")
    parser.add_argument('--format', type=str, choices=['registry', 'context', 'both', 'header', parse'], default='both',
                        help="Output format: 'registry', 'context', 'both', 'parse' or 'header' (default: both)")
    parser.add_argument('--context', type=str, required=False, 
                        help='Path to context.json')
    parser.add_argument('--registry', type=str, default=None,
                        help='Path to registry.json (enables registry mode)')
    parser.add_argument('--script', type=str, default=None,
                    help='Existing script file to append to (instead of creating new template)')
    
    args = parser.parse_args()

    if args.format == 'parse':    
        if not args.script:
            print('Need script to parse')
            sys.exit(1)
        beats, characters, zones = parse_script_txt(args.script)
    
        result = {
            "characters": characters,
            "zones": zones,
            "beats": beats
        }
        
        print(json.dumps(result, indent=2))
        sys.exit(0)

    if args.format == 'header':
        main2(args)
        sys.exit(0)
    
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

def main2(args):    
    # Determine output path
    output_path = Path(args.output)
    
    # Build header based on mode
    if args.registry:
        if not Path(args.registry).exists():
            print(f"[Error] Registry file not found: {args.registry}")
            sys.exit(1)
        header_lines, characters, zones = build_header_from_context_and_registry(args.context, args.registry)
    else:
        header_lines, characters, zones = build_header_from_context_only(args.context)
    
    # Append or create
    if args.script and Path(args.script).exists():
        # Append mode: add separator, then header, then existing content
        existing = Path(args.script).read_text(encoding='utf-8')
        separator = "=" * 50
        new_content = '\n'.join(header_lines) + '\n\n' + separator + '\n\n' + existing.rstrip()
        output_path.write_text(new_content, encoding='utf-8')
        print(f"\n✅ Appended header to existing script at: {output_path}")
    else:
        # Create mode: header + separator + template body
        template_lines = build_template_body(characters, zones)
        full_content = '\n'.join(header_lines) + '\n\n' + '\n'.join(template_lines) + '\n'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding='utf-8')
        print(f"\n✅ Created new script at: {output_path}")
    
    print(f"   Characters: {list(characters.keys())}")
    print(f"   Zones: {list(zones.keys())}")


if __name__ == "__main__":
    main()