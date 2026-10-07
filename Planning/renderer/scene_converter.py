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
    print("BACKGROUND: ", bg)
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
    from plan10.lib.create_registry import parse_script_txt
    from director import direct, build_beat_character_list
    
    script_path = Path(sys.argv[1])
    scene_base = Path(sys.argv[2])
    
    # 1. Parse the self-contained script (no registry/context needed!)
    beats, header_chars, header_zones = parse_script_txt(script_path)
    
    # 2. Build character lookup with absolute paths and visual IDs
    characters = {}
    for name, data in header_chars.items():
        ref_path = data['filepath']
        
        characters[name] = {
            'reference_path': ref_path,
            'Visual_Id': get_visual_id(ref_path)
        }
    
    # 3. Process beats
    notes = ''
    for beat_idx, beat in enumerate(beats, start=1):
        # Resolve background path from header zones
        zone_name = beat.get('zone', '').upper()
        zone_data = header_zones.get(zone_name, {})
        bg_filepath = zone_data.get('filepath', '')
        
        possible_bg_paths = [
            Path(scene_base) / 'scene' / bg_filepath,
            Path(scene_base) / bg_filepath,
            Path(bg_filepath)
        ]
        bg_path = next((str(p.resolve()) for p in possible_bg_paths if p.exists() and bg_filepath), "")
        beat['background'] = bg_path
        
        # Generate shots and get characters in this specific beat
        shots, notes = direct(beat, notes)
        characters_in_scene = build_beat_character_list(beat)
        
        actor_refs = []
        actor_names = []
        actor_identities = []
        
        for actor in characters_in_scene:
            actor_upper = actor['name'].strip().upper()
            if actor_upper in characters:
                actor_refs.append(characters[actor_upper]['reference_path'])
                actor_names.append(actor_upper)
                actor_identities.append(characters[actor_upper]['Visual_Id'])
                
        # Handle shot grouping if total duration exceeds max_total
        for subbeat, dentry in enumerate(group_pop_front(shots), start=1):
            script = h3_ref(
                beat['background'],
                actor_refs,
                beat.get('summary', ''),
                duration=10.0,
                visual_ids=actor_identities,
                char_names=actor_names,
                shots=dentry
            )
            
            outname = f"beat_{beat_idx:03d}_{subbeat:03d}.txt"
            (Path(scene_base) / outname).write_text(script, encoding='utf-8')
            print(f"✅ Generated {outname}")

if __name__ == '__main__':
    main()