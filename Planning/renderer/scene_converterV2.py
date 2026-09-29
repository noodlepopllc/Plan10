import sys, json
from pathlib import Path
from plan10.lib.image_analysis import EnhancePrompt, AnalyzeImage, translate_to_audio_prompt
from plan10.lib.qwen_llm import llm_analyze_media
from plan10.lib.util import video_to_img, to_absolute
from plan10.lib.image_gen import add_metadata_loc
from PIL import Image

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
    img = Image.open(image_path)
    metadata = load_metadata(img)
    for key, value in img.info.items():
        if isinstance(value, str):
            metadata.add_text(key, value)
    metadata.add_text(cache_key, result)
    img.save(image_path, pnginfo=metadata)
    img.close()
    
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
    script += f"summary | {prompt}\n"
    script += f"soundscape | {translate_to_audio_prompt(bg_desc)}\n"
    script += shots + "\n"

    script = replace_character_names(script, char_names)
    
    return script

import re
import unicodedata

def replace_character_names(script, char_names):
    # Normalize Unicode punctuation (curly quotes, fancy apostrophes)
    script = unicodedata.normalize("NFKC", script)

    # Split into quoted and non-quoted segments
    segments = re.split(r'(".*?"|\'.*?\')', script)

    # Process only non-quoted segments
    for cndx, name in enumerate(char_names, 1):
        token = f"char{cndx}"

        # Whole-word replacement
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)

        # Possessive replacement (Sora's, Lindsy's)
        pattern_possessive = re.compile(
            rf"\b{re.escape(name)}'s\b", re.IGNORECASE
        )

        for idx, segment in enumerate(segments):
            # Skip quoted segments entirely
            if segment and segment[0] in {'"', "'"}:
                continue

            # Apply replacements only outside quotes
            segment = pattern.sub(token, segment)
            segment = pattern_possessive.sub(f"{token}'s", segment)
            segments[idx] = segment

    return ''.join(segments)



import re

def normalize_shot_characters(shot_text: str, char_labels: list) -> str:
    """Replace character names with char tokens in a single shot, excluding quoted strings."""
    result = shot_text

    # Sort by length descending to avoid partial replacements
    sorted_labels = sorted(char_labels, key=len, reverse=True)

    # Split into quoted and non-quoted segments, keeping quotes
    # Matches "..." or '...'
    segments = re.split(r'(".*?"|\'.*?\')', result)

    # Process only non-quoted segments
    for i, label in enumerate(sorted_labels, 1):
        original_index = char_labels.index(label)
        token = f"char{original_index + 1}"

        for idx, segment in enumerate(segments):
            # Skip quoted segments (start with " or ')
            if not segment or segment[0] in {'"', "'"}:
                continue

            segments[idx] = re.sub(
                rf'\b{re.escape(label)}\b',
                token,
                segment,
                flags=re.IGNORECASE,
            )

    # Reassemble the text
    return ''.join(segments)


import os

def expand_to_shots(prompt: str,
                    bg_label: str,
                    char_labels: list,
                    duration: float) -> str:
    """
    Returns raw shot lines ready to append to your script,
    grounded in the actual first frame.
    """

    # ------------------------------------------------------------
    # 1. Character mapping string
    # ------------------------------------------------------------
    mapping = ""
    for idx, label in enumerate(char_labels, start=1):
        mapping += f"char{idx} = {label}\n"

    char_list = ", ".join(char_labels)
    duration_hint = f"Target duration: {int(duration)} seconds (approximate)."

    # ------------------------------------------------------------
    # 2. REWRITTEN SHOT-GENERATOR PROMPT
    # ------------------------------------------------------------
    llm_prompt = f"""
You are a deterministic video director generating Minimax‑friendly shots.

INPUT DATA:
- Characters: {char_list}
- Background: {bg_label}
- Scene description: {prompt}
- Target duration: {int(duration)} seconds (approximate)

CHARACTER MAPPING:
{mapping}

GOAL:
Produce a sequence of SHORT, STABLE, NON‑DRIFTING shots that Minimax H3 can render without filler motion.

SHOT COUNT RULE:
- Generate 3–7 shots depending on action complexity.
- More shots with shorter durations are preferred.
- Never produce fewer than 3 shots.

DURATION RULES (Minimax‑optimized):
- DEFAULT shot duration: 2 seconds.
- Only exceed 3 seconds when multiple distinct physical phases occur.
- NEVER exceed 4 seconds.
- Do NOT attempt to match the target duration exactly; prioritize stability.

ACTION DENSITY RULES:
- Split shots when the action contains distinct phases (e.g., “runs → jumps → lands”).
- Merge micro‑actions (glancing, shifting stance, breathing) into the nearest major shot.
- Avoid long continuous shots; Minimax destabilizes after ~3 seconds.

CAMERA RULES:
- Shot 1 may include slow camera movement (pan/tilt/dolly).
- All subsequent shots MUST use static medium framing.
- Medium shot = waist/chest upward, environment visible.
- No sudden angle changes between shots unless described.

CONTINUITY RULES:
- Maintain character posture, gaze direction, and spatial position across shots unless explicitly changed.
- No teleporting, no 180° rotations, no spontaneous stance changes.
- Lighting, shadows, and environment remain identical.

MOUTH/EXPRESSION RULES (when no dialogue):
- Every shot MUST specify mouth/jaw state:
  “lips pressed together”, “jaw clenched”, “mouth shut firmly”, “breathing through nose”
- No neutral faces without mouth description.

DIALOGUE RULES:
- Dialogue format: charX speaks [English] "text"
- Max 15 words per shot.
- Place dialogue near the end of the shot.
- No filler actions after speaking.

FOLEY RULES:
- EVERY shot MUST begin with a foley cue.
- Foley must match environment + physical action.
- No silence unless explicitly described.

FORMAT:
shot | foley + description | duration_seconds

NOW GENERATE THE SHOTS.
"""

    # ------------------------------------------------------------
    # 4. Call your LLM
    # ------------------------------------------------------------

    response = llm_analyze_media('', prompt=llm_prompt, max_tokens=8192, temperature=0.4)['analysis']

    return response


def get_visual_id(ref_path):
        prompt = """Analyze this image and extract a complete profile for character ONLY.


For EACH prominent character, provide:
1. 15-25 word description including ethnicity, exact age range, hair color and style (length, texture), skin tone, face shape, distinctive facial features, and main clothing items with specific colors

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

def group_pop_front(shots_text: str, max_total=15):
    # Temporarily split into lines; each line is one shot
    lines = [ln for ln in shots_text.split("\n") if ln.strip()]
    durations = [int(ln.split("|")[-1]) for ln in lines]

    total = sum(durations)
    # If everything fits, return the original string as a single bucket
    if total <= max_total:
        return [shots_text]

    # Otherwise, peel shots from the front until the remainder fits
    idx = 0
    while idx < len(lines) and sum(durations[idx:]) > max_total:
        idx += 1

    bucket1_lines = lines[:idx]
    bucket2_lines = lines[idx:]

    bucket1 = "\n".join(bucket1_lines)
    bucket2 = "\n".join(bucket2_lines)

    return [bucket1, bucket2]


def main():
    from parse_script import parse_script_txt
    from director import build_director_entries, direct
    scene_base = sys.argv[1]
    context = json.loads((Path(scene_base) / 'scene/context.json').read_text(encoding='utf-8'))
    base = Path(scene_base).parent
    registry = json.loads((Path(scene_base) / 'output/registry.json').read_text(encoding='utf-8'))
    lines = parse_script_txt(Path(scene_base) / 'output/script.txt')
    characters = get_characters(base, registry, context)
    lines = fix_locations(base, lines, registry, context)
    character_refs =  [characters[x]['reference_path'] for x in characters]
    visual_ids = [characters[x]['Visual_Id'] for x in characters]
    character_names=[x for x in characters]
    notes = ''
    for beat, line in enumerate(lines, start=1):
        #director_entries = build_director_entries(line)
        shots, notes = direct(line, notes)

        for subbeat, dentry in enumerate(group_pop_front(shots), start=1):
            #prompt = to_h3_prompt(dentry, characters)
            script = h3_ref(
                line['background'],
                character_refs,
                line['summary'],
                duration=10.0,
                visual_ids=visual_ids,
                char_names=character_names,
                shots=dentry
            )

            outname = f"beat_{beat:03d}_{subbeat:03d}.txt"
            (Path(scene_base) / outname).write_text(script, encoding='utf-8')
            print(script)

if __name__ == '__main__':
    main()