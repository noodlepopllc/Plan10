import sys, json, os
from pathlib import Path
from plan10.lib.image_analysis import EnhancePrompt, AnalyzeImage, translate_to_audio_prompt
from plan10.lib.qwen_llm import llm_analyze_media
from plan10.lib.util import video_to_img, to_absolute
from plan10.lib.image_gen import add_metadata_loc
from PIL import Image
import math

def get_or_analyze(image_path: str, prompt: str, cache_key: str, max_words: int = 15) -> str:
    """Get cached analysis from image metadata, or analyze and cache it."""
    img = Image.open(image_path)
    cached = img.info.get(cache_key)
    if cached:
        img.close()
        return cached
    
    result = AnalyzeImage(image_path, prompt=prompt, backend='')['analysis']
    img.close()
    from plan10.lib.util import load_metadata
    
    # Cache it
    with Image.open(image_path) as img:
        metadata = load_metadata(img)
        for key, value in img.info.items():
            if isinstance(value, str):
                metadata.add_text(key, value)
        metadata.add_text(cache_key, result)
        img.save(image_path, pnginfo=metadata)
    
    return result

def voice_prompt(gender, age):
    import random

    # Define your valid pitch options
    all_pitches = ['very low pitch', 'low pitch', 'moderate pitch', 'high pitch', 'very high pitch']

    # Set safe boundaries for pitch based on age/gender to prevent MiniMax distortion
    if age == 'child':
        # Children sound unnatural with heavy bass
        valid_pitches_for_char = ['moderate pitch', 'high pitch', 'very high pitch']
    elif gender == 'male' or age == 'elderly':
        # Adult males and elderly characters can sound highly robotic if forced into a squeaky register
        valid_pitches_for_char = ['very low pitch', 'low pitch', 'moderate pitch']
    else:
        # Young adult or middle-aged females can handle the full normal range safely
        valid_pitches_for_char = ['low pitch', 'moderate pitch', 'high pitch', 'very high pitch']

    # Randomly select a valid pitch
    selected_pitch = random.choice(valid_pitches_for_char)

    # Combine your tags to feed into your OmniVoice/MiniMax voice profile setup
    voice_profile = [gender, age, selected_pitch]
    return voice_profile

CHAR_PROMPT = '''
Return ONE sentence in this exact format:

"The {race/ethnicity} {gender} with {hair style} {hair color} hair is wearing {clothing list} and {accessory list}."

Use ONLY these slots. Do not reorder them.
'''

FACE_PROMPT = '''
Return ONE sentence in this exact format:

"The {race/ethnicity} {gender} has {face description} and {hair style} {hair color} hair."

Use ONLY these slots. Do not reorder them.

DEFINITIONS:
- {face description} includes 1–2 traits such as jawline, eyes, nose, freckles, or expression.
- {hair style} includes shape or length (short, long, wavy, straight, bob, layered).
- {hair color} is the observed color.
'''

def h3_ref(bg, refs, prompt, duration=10.0, visual_ids=[], char_names=[], shots=''):
    script = ""
    
    # 1. Background - CACHED
    bg_desc = add_metadata_loc(bg, prompt='', seed=-1, brief=True, update=False)
    script += f"bg | bg | {bg} | {bg_desc}\n"
    
    # 3. Generate shots FIRST to know who speaks
    char_labels = [f"char{ndx}" for ndx in range(1, len(refs) + 1)]

    shots = replace_character_names(shots, char_names)

    # Parse shots to find which characters speak (format: charX [verb] [English] "...")
    speaking_chars = set()
    for line in shots.split('\n'):
        # Look for [English] marker
        if '[English]' in line:
            # Find position of [English]
            idx = line.find('[English]')
            # Search backwards for char token
            before_english = line[:idx]
            # Find the last char* token before [English]
            for char in sorted(char_labels, key=len, reverse=True):  # longest first to avoid partial matches
                if char in before_english:
                    speaking_chars.add(char)
                    break
            
    
    # 4. Characters - CACHED (only generate audio for speakers)
    portrait_entries = ''
    for ndx, ref in enumerate(refs, start=1):
        label = f"char{ndx}"
        char_desc = get_or_analyze(ref, CHAR_PROMPT, 'Description', max_words=100)
        script += f"char | {label} | {ref} | {char_desc}\n"

        port_path = os.path.splitext(ref)[0] + '_portrait.png'
        portrait_desc = get_or_analyze(ref, FACE_PROMPT, 'Description', max_words=100)
        portrait_entries += f"portrait | portrait_{ndx} | {port_path} | {label} | A portrait of {label}\n"
        
        # Only generate audio if this character speaks
        if label in speaking_chars:
            voice_data = get_or_analyze(ref,
                "Identify the character's gender (male, female) and age bracket (child, teenager, young adult, middle-aged, elderly). Return exactly: 'gender, age bracket'.",
                'voice_profile')
            
            gender, age = [item.strip().lower() for item in voice_data.split(',')]
            voice_profile = voice_prompt(gender, age)
            
            wav_path = os.path.splitext(ref)[0] + '.wav'
            script += f"audio | voice_{ndx} | {wav_path} | {label} | {','.join(voice_profile)}\n"
    
    script += portrait_entries
    script += f"summary | {replace_character_names(prompt, char_names)}\n"
    script += f"soundscape | {translate_to_audio_prompt(bg_desc)}\n"
    script += shots + "\n"

    #script = replace_character_names(script, char_names)
    
    return script

import re
import unicodedata

import re

import re

def replace_character_names(script: str, char_names: list) -> str:
    """
    Replaces character names with charX tokens, skipping quoted segments.
    """
    # Build name → token mapping
    name_map = {}
    for i, name in enumerate(char_names, 1):
        if name and name.lower() != "unknown":
            name_map[name] = f"char{i}"
    
    # Split by quotes - even indices are outside quotes, odd are inside
    parts = re.split(r'(["\'])', script)
    result = []
    
    for i, part in enumerate(parts):
        if part in ['"', "'"]:
            result.append(part)
        elif i % 2 == 0:  # Not in quotes
            for name, token in name_map.items():
                # Case-insensitive replacement
                part = re.sub(re.escape(name), token, part, flags=re.IGNORECASE)
            result.append(part)
        else:  # In quotes
            result.append(part)
    
    return ''.join(result)

    # ------------------------------------------------------------
    # 4. Call your LLM
    # ------------------------------------------------------------

    response = llm_analyze_media('', prompt=llm_prompt, max_tokens=8192, temperature=0.4)['analysis']

    return response


def get_visual_id(ref_path):
        prompt = """Analyze this image and extract a complete profile for character ONLY.


For each, provide an 8-12 word visual identifier including:
- Approximate age and ethnicity
- Hair color and style
- Key clothing (1-2 items with colors)

"""
        
        result = AnalyzeImage(ref_path, prompt)['analysis'].strip()
        return result

def filter_empty(entry):
    vals = set([])
    for x in ("actor", "speaker", "action", "dialog"):
        vals.add(entry[x])
    return not len(vals) <= 1

def canonical_key(location_name: str, zone_name: str) -> str:
    key = f"{location_name}_{zone_name}"
    key = key.replace(' ', '_')
    key = key.replace('/', '_')
    key = key.replace('"','')
    key = key.upper()
    return f"{key}_BACKGROUND"

def fix_locations(base, lines, registry, context):
    # 1. Build location → zones map from registry
    locations = {}
    for location in registry['locations']:
        name = location['name']
        zones = [z['zone_name'] for z in location['zones']]
        locations[name] = zones

    # 2. For each line, infer location from zone and assign background
    for line in lines:
        zone = line['zone']

        # Find which location this zone belongs to
        loc = None
        for location_name, zone_list in locations.items():
            if zone in zone_list:
                loc = location_name
                break

        if loc is None:
            raise KeyError(f"Zone '{zone}' not found in any registry location")

        # Persist inferred location on the line
        line['location'] = loc

        # 3. Reconstruct canonical background key
        full_key = canonical_key(loc, zone)

        if full_key not in context['assets']:
            raise KeyError(f"Background key '{full_key}' not found in assets")

        # 4. Assign resolved background path
        line['background'] = str((base / Path(context['assets'][full_key]['path'])).resolve())

    return lines

def get_characters(base, registry, context):
    characters = {}
    for key in context['assets']:
        if key.startswith('CHAR') and 'VOICE' not in key:
            asset_path = str((base / Path(context['assets'][key]['path'])).resolve())
            char_key = key.split('_')[1].upper()
            if not char_key in characters:
                characters[char_key] = {}
            characters[char_key]['reference_path'] = asset_path
            characters[char_key]['Visual_Id'] = get_visual_id(asset_path)
        if key.startswith('CHAR') and 'VOICE' in key:
            asset_path = str((base / Path(context['assets'][key]['path'])).resolve())
            char_key = key.split('_')[1].upper()
            if not char_key in characters:
                characters[char_key] = {}
            characters[char_key]['Voice'] = asset_path
    return characters

def to_h3_prompt(entry, characters):
    # 1. Location + zone header
    header = f"{entry['location']}, {entry['zone']}"

    # 2. Character roster (from your character dict)
    roster_parts = []
    for name, info in characters.items():
        # Visual_Id is already a clean description
        desc = info['Visual_Id']
        roster_parts.append(f"{name} ({desc})")
    roster_line = ", ".join(roster_parts)

    # 3. Action line (required by H3)
    action_line = f"Action: {entry['action']}"

    # 4. Dialog line (optional)
    dialog_line = f"Dialog: {entry['dialog']}" if entry['dialog'] else ""

    # 5. Camera default (your runner expects this)
    camera_line = "Camera: static shot, medium framing"

    # 6. Combine into final string
    return "\n\n".join(x for x in [
        header,
        roster_line,
        action_line,
        dialog_line,
        camera_line
    ] if x)

import math

def group_pop_front(shots_text: str, max_total=15):
    lines = [ln for ln in shots_text.split("\n") if ln.strip()]

    fixed = []
    for line in lines:
        parts = line.split('|')
        try:
            final = math.ceil(float(''.join([x for x in parts.pop() if x in '0123456789.'])))
        except:
            final = 3
        fixed.append('|'.join(parts + [str(final)]))

    shots_text = '\n'.join(fixed)
    lines = fixed

    durations = [int(ln.split("|")[-1]) for ln in lines]
    total = sum(durations)

    if total <= max_total:
        return [shots_text]

    remaining = total
    added = 0
    idx = 0

    while idx < len(lines) and remaining > max_total and added < 5:
        remaining -= durations[idx]
        added += durations[idx]
        idx += 1

    bucket1 = "\n".join(lines[:idx])
    bucket2 = "\n".join(lines[idx:])

    return [bucket1, bucket2]


def main():
    from parse_script import parse_script_txt
    from director import direct, build_beat_character_list
    scene_base = sys.argv[1]
    context = json.loads((Path(scene_base) / 'scene/context.json').read_text(encoding='utf-8'))
    base = Path(scene_base).parent
    registry = json.loads((Path(scene_base) / 'output/registry.json').read_text(encoding='utf-8'))
    if (Path(scene_base) / 'output/script.json').exists():
        lines = json.loads((Path(scene_base) / 'output/script.json').read_text(encoding='utf-8'))
    else:
        lines = parse_script_txt(Path(scene_base) / 'output/script.txt', (Path(scene_base) / 'output/registry.json'))
        (Path(scene_base) / 'output/script.json').write_text(json.dumps(lines, indent=4), encoding='utf-8')
    characters = get_characters(base, registry, context)
    lines = fix_locations(base, lines, registry, context)
    character_refs =  [characters[x]['reference_path'] for x in characters]
    visual_ids = [characters[x]['Visual_Id'] for x in characters]
    character_names=[x for x in characters]
    notes = ''
    for beat, line in enumerate(lines, start=1):


        shots, notes = direct(line, notes)
        characters_in_scene = build_beat_character_list(line)

        # Build name→index lookup once (outside the loop)
        name_to_idx = {name.strip().upper(): i for i, name in enumerate(character_names)}

        actor_refs = []
        actor_names = []
        actor_identities = []

        for actor in characters_in_scene:
            actor_upper = actor['name'].strip().upper()
            if actor_upper in name_to_idx:
                idx = name_to_idx[actor_upper]
                actor_refs.append(character_refs[idx])
                actor_names.append(character_names[idx])
                actor_identities.append(visual_ids[idx])

        for subbeat, dentry in enumerate(group_pop_front(shots), start=1):
            script = h3_ref(
                line['background'],
                actor_refs,
                line['summary'],
                duration=10.0,
                visual_ids=actor_identities,  # ← comma added
                char_names=actor_names,
                shots=dentry
            )

            outname = f"beat_{beat:03d}_{subbeat:03d}.txt"

            (Path(scene_base) / outname).write_text(script, encoding='utf-8')
            print(script)

if __name__ == '__main__':
    main()