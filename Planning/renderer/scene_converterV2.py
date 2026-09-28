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

def h3_ref(bg, ff, refs, portraits, prompt, duration=10.0, visual_ids=[], char_names=[]):
    script = ""
    
    # 1. First Frame - NO CACHE
    if ff:
        ff_desc = AnalyzeImage(ff, prompt='Briefly describe the scene composition, character positions, and environment. Max 15 words.')['analysis']
        script += f"ff | ff | {ff} | {ff_desc}\n"
    
    # 2. Background - CACHED
    bg_desc = add_metadata_loc(bg, prompt='', seed=-1, brief=True, update=False)
    script += f"bg | bg | {bg} | {bg_desc}\n"
    
    # 3. Generate shots FIRST to know who speaks
    char_labels = [f"char{ndx}" for ndx in range(1, len(refs) + 1)]
    shots = expand_to_shots(prompt, bg, char_labels, duration, first_frame_path=ff)

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

        if portraits:
            portrait_desc = get_or_analyze(portraits[ndx-1], FACE_PROMPT, 'Description', max_words=100)
            portrait_entries += f"portrait | portrait_{ndx} | {portraits[ndx-1]} | {label} | {portrait_desc}\n"
        else:
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
    script += f"summary | {[x for x in prompt.split('\n') if 'Action:' in x][0].replace('Action:','').strip()}\n"
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
                    duration: float,
                    first_frame_path: str = None) -> str:
    """
    Returns raw shot lines ready to append to your script,
    grounded in the actual first frame.
    """

    # ------------------------------------------------------------
    # 1. Optional first-frame analysis (your existing logic)
    # ------------------------------------------------------------
    scene_context = ""
    if first_frame_path and os.path.exists(first_frame_path):
        analysis = AnalyzeImage(first_frame_path, prompt="""
            Describe this exact frame for video generation:
            Where are the characters positioned? What are their poses and expressions?
            What is the camera angle? Be specific about spatial relationships.
            DO NOT describe clothing colors or minor details, just the layout and action.
        """)['analysis']

        scene_context = (
            "\n\nVISUAL CONTEXT (This is the EXACT starting frame at 00:00.000):\n"
            f"{analysis}\n"
        )

    # ------------------------------------------------------------
    # 2. Character mapping string
    # ------------------------------------------------------------
    mapping = ""
    for idx, label in enumerate(char_labels, start=1):
        mapping += f"char{idx} = {label}\n"

    char_list = ", ".join(char_labels)
    duration_hint = f"Target duration: {int(duration)} seconds (approximate)."

    # ------------------------------------------------------------
    # 3. REWRITTEN SHOT-GENERATOR PROMPT
    # ------------------------------------------------------------
    llm_prompt = f"""
You are an expert cinematic video director breaking a scene into sequential shots.

INPUT DATA:
- Characters: {char_list}
- Background: {bg_label}
- {duration_hint}
- Scene description: {prompt}

CHARACTER MAPPING:
{mapping}{scene_context}

TASK:
Generate a sequence of cinematic shots that follow the scene description and maintain visual continuity.

STRICT RULE: Produce BETWEEN 2 AND 5 SHOTS.
Never produce fewer than 2 shots.
Never exceed 5 shots.

DURATION RULES:
- Avoid 1-second shots unless the shot contains short dialog (≤5 words).
- Prefer 2–4 second shots for stability.
- Combine sequential minor actions into a single continuous shot.
- Use longer takes instead of additional cuts whenever possible.
- Total duration should approximate the target duration without exceeding 5 shots.

SHOT CONDENSATION RULES:
- If the director’s action describes a single continuous motion, represent it with 1–2 shots, not 3–5.
- Merge small physical beats (turning, glancing, breathing, shifting stance) into the nearest major shot.
- Only split shots when the director’s action contains distinct physical phases (e.g., “runs → jumps → lands”).

REACTION SHOT RULES:
- Only generate reaction shots when the director’s action explicitly implies another character is observing.
- Do NOT add reaction shots automatically.
- If a reaction shot is needed, limit it to ONE per beat.

CONTINUITY RULES:
- Maintain character posture, gaze direction, and spatial position across shots unless the director’s action changes them.
- Lighting, shadows, and weather remain identical.
- Characters do NOT teleport, rotate 180°, or change stance between shots unless described.
- Clothing, props, and environmental elements remain consistent.

SILENCE RULES (when no dialogue is present):
- Every shot MUST describe the character's mouth/jaw state explicitly:
  "lips pressed together", "jaw clenched", "mouth shut firmly", "breathing through nose"
- Focus audio attention on ENVIRONMENT and PHYSICAL EXERTION:
  heavy breathing, exertion sounds, environmental foley
- Never describe characters facing each other in neutral medium shot without a physical mouth state.

CAMERA RULES:
1. Shot 1 may include camera movement (pan, tilt, dolly) at slow speed.
2. All subsequent shots MUST use static medium framing.
3. Medium shot = waist/chest upward, environment visible.

DIALOGUE RULES:
- Dialogue format: charX speaks [English] "text"
- Max 15 words per shot.
- Do NOT add filler actions after speaking.
- If dialogue is present, place it near the end of the shot.

FOLEY RULES:
- EVERY shot MUST begin with a foley cue.
- Foley must match the environment and physical action.

FORMAT:
shot | foley + description | duration_seconds

NOW GENERATE THE SHOTS FOR THE INPUT DATA ABOVE.
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
    for line in lines:
        loc = line['location']
        zone = line['zone']

        full_key = canonical_key(loc, zone)

        if full_key not in context['assets']:
            raise KeyError(f"Background key '{full_key}' not found in registry")

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

def main():
    from parse_script import parse_script_txt
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
    from director import build_director_entries

    for beat, line in enumerate(lines, start=1):
        director_entries = build_director_entries(line)

        for subbeat, dentry in enumerate(director_entries, start=1):
            prompt = to_h3_prompt(dentry, characters)
            script = h3_ref(
                dentry['background'],
                None,
                character_refs,
                None,
                prompt,
                duration=10.0,
                visual_ids=visual_ids,
                char_names=character_names
            )

            outname = f"beat_{beat:03d}_{subbeat:02d}.txt"
            (Path(scene_base) / outname).write_text(script, encoding='utf-8')
            print(script)

if __name__ == '__main__':
    main()