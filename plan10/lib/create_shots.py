import sys, json, os, time, re, math
from pathlib import Path
from plan10.lib.config import load_config
load_config()

from plan10.lib.image_analysis import EnhancePrompt, AnalyzeImage, translate_to_audio_prompt
from plan10.lib.qwen_llm import llm_analyze_media, LLMContext
from plan10.lib.util import video_to_img, to_absolute
from plan10.lib.image_gen import add_metadata_loc
from PIL import Image

def llm(prompt, cooloff=30, processor=None, model=None):
    if os.environ.get("LLM_BACKEND", "transformers") == "ollama":
        print(f'Cool off period: {cooloff} seconds')
        time.sleep(cooloff)
    response = llm_analyze_media('', prompt=prompt, max_tokens=8192, temperature=0.4, processor=processor, model=model)['analysis']
    return response.strip()

def parse_director_splits_with_shots(director_shots_text: str, final_shotlist: str, original_summary: str):
    """
    Parse director's output for summaries, then split the shot planner's output to match.
    """
    # Split director's output by parts
    parts = re.split(r'---\s*PART\s+\d+\s*---', director_shots_text)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) <= 1:
        # No split, use original summary and full shotlist
        yield (original_summary, final_shotlist.strip())
        return

    # Split detected: extract summaries and count shots per part
    part_summaries = []
    part_shot_counts = []
    
    for part_text in parts:
        # Extract summary
        summary_match = re.search(r'summary:\s*(.+?)(?=\nshot|\n---|$)', part_text, re.IGNORECASE | re.DOTALL)
        sub_summary = summary_match.group(1).strip() if summary_match else original_summary
        part_summaries.append(sub_summary)
        
        # Count shots in this part (case-insensitive to catch "Shot 1" or "shot 1")
        shot_count = len(re.findall(r'(?i)^shot\s+\d+', part_text, re.MULTILINE))
        part_shot_counts.append(shot_count)
    
    # Split the final_shotlist accordingly
    shot_lines = [line for line in final_shotlist.split('\n') if line.strip().lower().startswith('shot')]
    
    current_idx = 0
    for i, (summary, shot_count) in enumerate(zip(part_summaries, part_shot_counts)):
        sub_shots = '\n'.join(shot_lines[current_idx:current_idx + shot_count])
        current_idx += shot_count
        yield (summary, sub_shots)

shot_planner_prompt = '''
You are the shot planner.

Your job is to convert the approved director shot plan into final renderer-ready shot lines.

INPUT:
- Director shot plan: {director_shot_plan}

------------------------------------------------------------
SHOT PLANNER ROLE
------------------------------------------------------------
Convert the approved director shot plan into renderer-ready syntax.
Preserve the director shot plan exactly as written.

------------------------------------------------------------
DIALOG FORMATTING (CRITICAL)
------------------------------------------------------------
Look for exact quoted dialog in EITHER the "dialog:" or "audio:" lines of the director shot plan.
Format ALL spoken dialog using this exact renderer syntax:

character speaks [English] "exact dialog text"
They close their mouth and are silent.

The [English] tag is renderer metadata required for all dialog lines.
Preserve the quoted dialog text exactly without paraphrasing, summarizing, or splitting it.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
Format each shot as a single line:

shot | audio. camera. visual. dialog (if any). | duration

Requirements:
- Begin each line with "shot |"
- Place duration as the final pipe-delimited integer
- Write one shot per line
- Provide only the formatted shot lines without additional content

------------------------------------------------------------
NOW PRODUCE THE SHOT LIST.
'''

camera_prompt = '''
You are a professional camera operator filming a scene in real time.

INPUTS:
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

Your job: produce a moment-by-moment camera log describing what the camera sees and hears.
This is the raw temporal plan for the director, not a final shot list.

------------------------------------------------------------
CREATIVE CINEMATOGRAPHY
------------------------------------------------------------

Enrich the visual presentation through framing, composition, camera movement, facial expression, body language, and subject emphasis.

Enhance observable details that are consistent with the scene:
- Facial expressions matching the character's delivery and emotion
- Body language reinforcing the described action
- Camera angles emphasizing the emotional tone

Keep all character actions and expressions strictly aligned with the scene description.
The scene_description is the authoritative source for all character behavior.

------------------------------------------------------------
CAMERA MOVEMENT
------------------------------------------------------------

Each shot contains ONE smooth, motivated camera movement (or is static).

Every camera movement serves a specific purpose:
- Character movement motivates the camera to follow
- Gaze shifts motivate the camera to pan
- Emotional beats motivate a push-in or pull-back

Maintain spatial continuity across shots through consistent screen direction and framing, NOT through chained camera movements.

Prioritize simple, strong compositions over complex maneuvers.

------------------------------------------------------------
SPATIAL BLOCKING AND EYELINE GEOMETRY
------------------------------------------------------------

Before framing any shot, establish the spatial relationship between characters.

Identify where each character is physically located in the scene using the scene_description.
Map character positions to screen directions (screen left, screen right, center).
Establish an imaginary axis of action between interacting characters.
Keep the camera on one consistent side of this axis throughout the conversation.
Maintain consistent screen direction for each character across all moments.

Always direct the speaking character's eyeline toward the listener's established screen position.
Angle the speaker's gaze just past the lens in the listener's direction when the listener is off-screen.
Direct the speaker's gaze to match the spatial relationship described in the scene.
Keep the visual focus tightly on the active character described in the current moment.

------------------------------------------------------------
CHARACTER PRESENCE
------------------------------------------------------------

The character_list is the authoritative source for who is present in this beat.
Limit all character references to those appearing in character_list.
Treat context_notes as continuity of tone, emotion, and physical state only.

------------------------------------------------------------
PHYSICAL STATE (CRITICAL)
------------------------------------------------------------
If the scene_description states a character's physical position 
(sitting, standing, kneeling, etc.), you MUST use that exact position.
Do NOT infer or assume physical positions from the environment.
A character in a tavern is NOT necessarily sitting.
A character at a table is NOT necessarily seated.
Only use the physical state explicitly provided in scene_description.

------------------------------------------------------------
ACTOR ISOLATION
------------------------------------------------------------

Feature one active character per moment.
Two characters may both be active only when performing one synchronized physical action together.
Apply the single-active-character default when the scene describes no synchronized action.

------------------------------------------------------------
TEMPORAL & SHOT RULES
------------------------------------------------------------

Plan discrete SHOTS of 2 to 5 seconds each.
Open with a wide or medium-wide establishing shot.
Reserve slow pan or tilt for the first shot only.

CRITICAL CAMERA RULE: Each shot must contain EXACTLY ONE simple camera movement (e.g., ONLY a slow push-in, OR ONLY a slow pan, OR completely static). 
NEVER chain multiple movements in a single shot (e.g., "tilts up then pushes in" is strictly forbidden).

------------------------------------------------------------
DIALOG
------------------------------------------------------------

Extract exact quoted dialog verbatim from DIALOG: "..." lines in scene_description.
Include exact quoted dialog in audio notes when describing speech.
Use either exact dialog text or silent physical behavior for each moment.
Speaking moments require exact quoted dialog to be valid.

------------------------------------------------------------
OBSERVABLE REALITY
------------------------------------------------------------

Describe only what the camera and microphone directly observe.

Visual: visible elements, character actions, environmental details.
Audio: natural diegetic sounds (footsteps, objects, environment) and exact quoted dialog.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
shot N | duration_seconds (2-5)
camera: [ONE simple movement or static] + [angle]
visual: what is visible + character actions
audio: notable sounds + exact quoted dialog (if speaking)
------------------------------------------------------------
NOW PRODUCE THE CAMERA LOG.
'''

director_prompt = '''
Your job: verify the camera log faithfully represents the beat while maintaining continuity and cinematic grammar.

The beat is the source of truth. The camera log is an interpretation. Your role is to verify and correct.

INPUTS:
- Camera operator log: {camera_log}
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

------------------------------------------------------------
VALIDATION RULES (CRITICAL)
------------------------------------------------------------
1. NO CHAINED MOVEMENTS: If any shot in the camera log contains more than one camera movement (e.g., "pans then pushes in"), you MUST split it into two separate shots.
2. HARD DURATION CAP: No single shot may exceed 5 seconds. If a shot is longer, split it.
3. DIALOG PROTECTION: Never split a single line of dialog across two shots. If a dialog turn is long, keep it in one shot (up to 5 seconds max). 
4. ACTOR ISOLATION: Feature one active character per shot. When a character speaks, they are the sole moving subject.

------------------------------------------------------------
DURATION AND SPLITTING RULES
------------------------------------------------------------
Calculate the total estimated duration of all validated shots in this beat.
If the total duration exceeds 15 seconds, you MUST split the output into TWO distinct parts ("--- PART 1 ---" and "--- PART 2 ---") with unique summaries for each.

When splitting, you MUST:
1. Keep narrative units intact. NEVER separate a physical action from the dialog that accompanies it.
2. Generate a NEW, unique `summary:` for EACH part.

------------------------------------------------------------
OUTPUT FORMAT (Validated Shot Plan)
------------------------------------------------------------
summary: [One sentence describing the core visual action and dialog of this beat/part]
shot 1
type: establishing / action / dialog / reaction
duration: 2-5
camera: [ONE simple movement or static] + [angle]
visual: summary of visible elements
dialog: [Exact quoted dialog, or "None"]
audio: [Ambient sounds]

(Repeat for shot 2, shot 3, etc.)

------------------------------------------------------------
NOW PRODUCE THE VALIDATED DIRECTOR SHOT PLAN.
'''

def summarize_continuity_from_director_shots(director_shots_text: str) -> str:
    """
    Produces a short continuity summary from the previous beat's director output.
    This summary is passed as context_notes to the camera operator and director.
    """

    lines = director_shots_text.splitlines()
    camera_desc = []
    visual_desc = []
    audio_desc = []

    for line in lines:
        l = line.lower()

        if l.startswith("camera:"):
            camera_desc.append(line.replace("camera:", "").strip())

        elif l.startswith("visual:"):
            visual_desc.append(line.replace("visual:", "").strip())

        elif l.startswith("audio:"):
            audio_desc.append(line.replace("audio:", "").strip())

    # Build continuity summary
    summary_parts = []

    if visual_desc:
        summary_parts.append(f"Previously visible: {visual_desc[-1]}.")

    if camera_desc:
        summary_parts.append(f"Camera was positioned as: {camera_desc[-1]}.")

    if audio_desc:
        summary_parts.append(f"Ambient audio included: {audio_desc[-1]}.")

    # Final continuity paragraph
    continuity_summary = " ".join(summary_parts)

    return continuity_summary.strip()

import re

def quoted_word_count(text):
    quotes = re.findall(r'"([^"]*)"', text)
    return sum(len(q.split()) for q in quotes)

def build_beat_character_list(beat_entry: dict) -> list:
    """
    Extract only the characters actually present in this beat.
    Returns a list of dicts with minimal info for prompt consumption.
    """
    characters = []
    seen_names = set()
    
    # Active characters (speaking/acting)
    for char in beat_entry.get('active_characters', []):
        name = char['name']
        if name not in seen_names:
            seen_names.add(name)
            characters.append({
                'name': name,
                'role': 'active',
                'delivery': char.get('delivery'),
                'has_dialog': bool(char.get('dialog')),
                'has_action': bool(char.get('action'))
            })
    
    # Passive characters (mentioned but not active)
    for char in beat_entry.get('passive_characters', []):
        name = char['name']
        if name not in seen_names:
            seen_names.add(name)
            characters.append({
                'name': name,
                'role': 'passive',
                'source': char.get('source'),
                'mentioned_by': char.get('mentioned_by')
            })
    
    return characters

def direct(beat_entry: dict, notes='', scene_base=''):
    if notes: 
        notes = summarize_continuity_from_director_shots(notes)

    beat_characters = build_beat_character_list(beat_entry)
    scene_parts = []
    if beat_entry.get('summary'):
        scene_parts.append(beat_entry['summary'])
    for char in beat_entry.get('active_characters', []):
        # NEW: Inject explicit physical state
        physical_state = char.get('physical_state', '')
        if physical_state:
            scene_parts.append(f"{char['name']} is {physical_state}.")
        if char.get('action'):
            scene_parts.append(f"{char['name']}: {char['action']}")
        if char.get('dialog'):
            scene_parts.append(f'{char["name"]} DIALOG: "{char["dialog"]}"')
    scene_description = " ".join(scene_parts)
    with open(Path(scene_base) / 'director.log', 'a') as dlog:
        with LLMContext() as (p_ctx, m_ctx):
            dlog.write(f'\n{"*"*100} \nScene Description\n {scene_description}\n')
            camera_log = llm(camera_prompt.format(
                scene_description=scene_description,
                character_list=beat_characters,
                background_label=beat_entry['background'],
                context_notes=notes
            ), processor=p_ctx, model=m_ctx)
            print(f"Camera Done: \n{camera_log}")
            dlog.write(f'\n{"-"*100} \nCamera Log\n {camera_log}\n')

            director_shots = llm(
                director_prompt.format(
                    camera_log=camera_log,
                    scene_description=scene_description,
                    character_list=beat_characters,
                    background_label=beat_entry["background"],
                    context_notes=notes
                ), processor=p_ctx, model=m_ctx
            )
            print(f"Director Done: \n{director_shots}")
            dlog.write(f'\n{"-"*100} \nDirector Log\n {director_shots}\n')

            final_shotlist = llm(
                shot_planner_prompt.format(
                    director_shot_plan=director_shots
                ), processor=p_ctx, model=m_ctx
            )
            print(f"Shotlist Done: \n{final_shotlist}")
            dlog.write(f'\n{"-"*100} \nShot list\n {final_shotlist}\n')

    return final_shotlist, director_shots


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
    from plan10.lib.create_metadata import parse_script_txt
    
    script_path = Path(sys.argv[1])
    scene_base = Path(sys.argv[2])
    
    # 1. Parse the self-contained script (no registry/context needed!)
    beats, header_chars, header_zones = parse_script_txt(script_path)
    with open(Path(scene_base) / 'beats.json') as oj:
        json.dump(beat, oj, indent=4)
    
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
        shots, director_shots_text = direct(beat, notes, scene_base)
        notes = director_shots_text  # Pass director output to next beat for continuity
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
        for subbeat, (sub_summary, sub_shots) in enumerate(
            parse_director_splits_with_shots(director_shots_text, shots, beat.get('summary', '')), 
            start=1
        ):
            script = h3_ref(
                beat['background'],
                actor_refs,
                sub_summary,
                duration=10.0,
                visual_ids=actor_identities,
                char_names=actor_names,
                shots=sub_shots  # Now this is renderer-ready
            )
            
            outname = f"beat_{beat_idx:03d}_{subbeat:03d}.txt"
            (Path(scene_base) / outname).write_text(script, encoding='utf-8')
            print(f"✅ Generated {outname}")

if __name__ == '__main__':
    main()