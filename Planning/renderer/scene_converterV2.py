import sys, json
from pathlib import Path
from plan10.lib.image_analysis import AnalyzeImage
#from plan10.emergent.video_runner import h3_ref

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

def expand_to_shots(prompt: str, bg_label: str, char_labels: list, duration: float, first_frame_path: str = None) -> str:
    """Returns raw shot lines ready to append to your script, grounded in the actual first frame."""

    scene_context = ""
    if first_frame_path and os.path.exists(first_frame_path):
        # Assuming AnalyzeImage is defined elsewhere in your code
        analysis = AnalyzeImage(first_frame_path, prompt="""
            Describe this exact frame for video generation:
            Where are the characters positioned? What are their poses and expressions?
            What is the camera angle? Be specific about spatial relationships.
            DO NOT describe clothing colors or minor details, just the layout and action.
        """)['analysis']
        scene_context = f"\n\nVISUAL CONTEXT (This is the EXACT starting frame at 00:00.000):\n{analysis}\n"

    char_tokens = [f"char{i+1}" for i in range(len(char_labels))]
    char_list = ", ".join(char_tokens)
    mapping = "\n".join([f"- {char_tokens[i]} = {char_labels[i]}" for i in range(len(char_labels))])

    formatted_prompt = f"""You are an expert cinematic video director breaking a scene into sequential shots.

INPUT DATA:
- Characters: {char_list}
- Background: {bg_label}
- Target duration: {int(duration)} seconds (approximate)
- Scene description: {prompt}

CHARACTER MAPPING:
{mapping}
{scene_context}

TASK:
Generate a sequence of cinematic shots that follow the scene description and maintain visual continuity. 
CRITICAL CONSTRAINT: You must summarize and condense the action. Generate a STRICT MAXIMUM of 5 shots. Do not exceed 5 shots under any circumstances. Combine minor actions into continuous takes and focus only on the most crucial narrative beats.

SHOT DURATION GUIDELINES:
- Adjust shot durations to approximate the total target duration, but NEVER exceed 5 shots total.
- Quick dialogue (1-5 words): 1-2 seconds
- Medium dialogue (6-15 words): 2-3 seconds
- Simple actions (turn, look, gesture): 2-3 seconds
- Complex actions (crawl, stand up, walk): 3-5 seconds (Use longer takes to fill time instead of adding cuts)
- Reaction shots: 1-2 seconds

GLOBAL RULES:

SILENCE RULES (when no dialogue is present):
- Every shot MUST describe the character's mouth/jaw state explicitly:
  "lips pressed together", "jaw clenched", "mouth shut firmly", "breathing through nose"
- Focus audio attention on ENVIRONMENT and PHYSICAL EXERTION:
  heavy breathing, exertion sounds, environmental foley
- Never describe characters facing each other in neutral medium shot without a physical mouth state

1. Use MEDIUM SHOTS as the default framing for dialogue and action.
   - Characters visible from waist/chest upward.
   - Environment must remain visible.

2. Camera movement is ONLY allowed in Shot 1 (establishing).
   - After Shot 1, camera remains static or uses minimal drift.

3. Dialogue:
   - Dialogue format: char speaks [English] "text"
   - DO NOT add padding like "closes mouth" or "is silent" after speaking
   - Keep dialogue shots tight and natural

4. Physicality:
   - Dialogue shots MAY include a brief physical action before speaking (turns, breath)
   - DO NOT force physical actions after speaking - this creates padding

5. Dialogue length:
   - Max 15 words per shot. Break long dialogue into multiple shots (but remember the 5-shot total limit!).

6. Foley:
   - EVERY shot MUST begin with a foley cue.

7. Continuity:
   - Lighting, shadows, and weather remain identical.
   - Actions flow continuously between shots.

8. Pacing & Summarization:
   - Prioritize the core emotional or narrative beat of the scene.
   - Combine sequential minor actions (e.g., walking over and picking up an object) into a single shot instead of cutting.

FORMAT:
shot | foley + description | duration_seconds

EXAMPLE:
shot | Low wind through rafters. Medium shot. char1 shifts her stance, glancing toward char2. | 2
shot | Soft creak of wood. Medium shot of char1 facing char2. char1 speaks [English] "Stay back." | 1
shot | Distant hoofbeats. Medium shot. char2 reacts with a quick blink. | 2

NOW, generate the shots for the INPUT DATA provided above (REMEMBER: STRICT MAX 5 SHOTS):
"""
    
    response = llm_analyze_media('', prompt=formatted_prompt, max_tokens=8192, temperature=0.4)['analysis']

    lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("shot |"):
            line = normalize_shot_characters(line, char_labels)
            lines.append(line)

    return "\n".join(lines)

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

def fix_locations(base, lines, registry, context):
    locations = {}
    for location in registry['locations']:
        name = location['name']
        locations[name] = []
        for zone in location['zones']:
            locations[name].append(zone['zone_name'])
    #print(locations)

    location_info = {}
    for asset in context['assets']:
        if 'BACKGROUND' in asset:
            print(asset)
            for location in locations:
                if asset.startswith(location.replace(' ','_').upper()):
                    if location not in location_info:
                        location_info[location] = {}
                    for zone in locations[location]:
                        if zone.replace(' ','_').upper() in asset:
                            location_info[location][zone] = str((base / Path(context['assets'][asset]['path'])).resolve())
    for line in lines:
        for k, v in locations.items():
            if line['zone'] in v:
                line['location'] = k
                line['background'] = location_info[k][line['zone']]
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
    scene_base = sys.argv[1]
    context = json.loads((Path(scene_base) / 'scene/context.json').read_text())
    base = Path(scene_base).parent
    registry = json.loads((Path(scene_base) / 'output/registry.json').read_text())
    lines = json.loads((Path(scene_base) / 'output/bleh.json').read_text())
    #lines = [line for line in lines if filter_empty(line)]
    characters = get_characters(base, registry, context)
    lines = fix_locations(base, lines, registry, context)
    character_refs =  [characters[x]['reference_path'] for x in characters]
    visual_ids = [characters[x]['Visual_Id'] for x in characters]
    character_names=[x for x in characters]
    for beat, line in enumerate(lines, start=1):
        print(line)
        script = h3_ref(line['background'], None, character_refs, None, to_h3_prompt(line, characters), duration=10.0, visual_ids=visual_ids, char_names=character_names)
        (Path(scene_base) / f'beat_{beat:03d}.txt').write_text(script)
        print(script)

if __name__ == '__main__':
    main()