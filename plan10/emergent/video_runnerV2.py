import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path
import os, traceback, re, math, time
from PIL import Image

from plan10.lib.config import load_environ
load_environ()

from plan10.lib.image_analysis import EnhancePrompt, AnalyzeImage, translate_to_audio_prompt
from plan10.lib.qwen_llm import llm_analyze_media, LLMContext
from plan10.lib.util import video_to_img, to_absolute

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
MMH3 = os.environ.get('MMH3', 'False') != 'False'
WIDTH = int(os.environ.get('WIDTH', '768'))
HEIGHT = int(os.environ.get('HEIGHT', '448'))
ANIME = os.environ.get('ANIME', 'False') != 'False'

# ==============================================================================
# DIRECTOR WORKFLOW INTEGRATION
# ==============================================================================

def llm_director(prompt, cooloff=30, processor=None, model=None):
    if os.environ.get("LLM_BACKEND", "transformers") == "ollama":
        print(f'Cool off period: {cooloff} seconds')
        time.sleep(cooloff)
    response = llm_analyze_media('', prompt=prompt, max_tokens=8192, temperature=0.4, processor=processor, model=model)['analysis']
    return response.strip()

shot_planner_prompt = '''
You are the shot planner.

Your job is to convert the approved director shot plan into final renderer-ready shot lines.

INPUT:
- Director shot plan: {director_shot_plan}

------------------------------------------------------------
SHOT PLANNER ROLE
------------------------------------------------------------

The director shot plan is authoritative.
Convert the approved director shot plan into renderer-ready syntax.
Preserve the director shot plan exactly as written.

------------------------------------------------------------
PRESERVATION REQUIREMENTS
------------------------------------------------------------

Keep all actions, camera descriptions, dialog, audio, and durations exactly as they appear in the director shot plan.
Maintain all camera angles, movements, shot sizes, and compositions without modification.
Use only ambient audio explicitly present in the director shot plan.
Apply the director duration exactly as specified.

------------------------------------------------------------
DIALOG FORMATTING
------------------------------------------------------------

Extract exact quoted dialog from DIALOG: "..." lines in the director shot plan.
Format all spoken dialog using this renderer syntax:

character speaks [English] "dialog text"
They close their mouth and are silent.

The [English] tag is renderer metadata required for all dialog lines.
Preserve the quoted dialog text exactly without paraphrasing or summarizing.

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
CONTINUOUS CAMERA MOVEMENT
------------------------------------------------------------

The camera moves through the scene as a continuous flowing presence.

Connect every moment with smooth, motivated camera movement.
Frame transitions describe the camera moving to its new position.
Each moment ends where the next moment begins, maintaining spatial continuity.
Maintain continuous spatial tracking when a character performs a multi-part action.

Every camera movement serves a specific narrative, emotional, or spatial purpose:
- Character movement motivates the camera to follow the action
- Gaze shifts motivate the camera to pan toward the new focus
- Spatial relationships motivate the camera to reveal the environment
- Emotional beats motivate the camera to push in for intimacy or pull back for isolation

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
ACTOR ISOLATION
------------------------------------------------------------

Feature one active character per moment.
Two characters may both be active only when performing one synchronized physical action together.
Apply the single-active-character default when the scene describes no synchronized action.

------------------------------------------------------------
TEMPORAL RULES
------------------------------------------------------------

Plan MOMENTS of 2 seconds each, with a maximum of 3 seconds.
These moments will be merged into shots by the director.
Open with a wide or medium-wide establishing shot.
Reserve slow pan or tilt for the first moment only.

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
moment N | duration_seconds
camera: angle + movement
visual: what is visible + character actions
audio: notable sounds

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

Your output is the semantic shot plan for the shot planner.

------------------------------------------------------------
BEAT FIDELITY
------------------------------------------------------------

Keep all actions, reactions, and object interactions strictly aligned with scene_description.
Characters remain still unless explicitly described in the scene.
When uncertain, prefer scene_description over camera_log.
Enhance the cinematic framing while preserving the source material.

------------------------------------------------------------
DIALOG EXTRACTION AND VERIFICATION
------------------------------------------------------------

Extract exact quoted dialog verbatim from DIALOG: "..." lines in scene_description.
Include exact quoted dialog in every shot describing speech.
Use exact quoted dialog text for all speaking moments.
Speaking moments require exact quoted dialog to be valid.

------------------------------------------------------------
ACTOR ISOLATION
------------------------------------------------------------

Feature one active character per shot.
Two characters may both be active only when performing one synchronized physical action together.
When a character speaks, they are the sole moving subject.
Other visible characters remain frozen and static during speech.
Passive characters appear with static language only.

------------------------------------------------------------
SPATIAL AND EYELINE VALIDATION
------------------------------------------------------------

Direct the speaker's eyeline toward the listener's established screen position.
Maintain consistent screen direction for each character across all shots.
Keep the camera on one consistent side of the axis of action.
Angle the speaker's gaze just past the lens when the listener is off-screen.
Match the speaker's gaze direction to the spatial relationship in scene_description.

------------------------------------------------------------
SHOT BOUNDARY RULES
------------------------------------------------------------

Start a new shot when:
- Action intent changes
- Gaze target changes
- Speech begins or ends
- Object interaction begins or ends
- Character enters or exits

Merge moments into shots of 2-10 seconds each.
Sum the durations of merged moments.

Merge moments only when:
- Camera angle remains identical
- Motion continues as part of the same phase
- Dialog belongs to the same turn
- No character enters or exits
- Only one active character is present

------------------------------------------------------------
SHOT TYPES
------------------------------------------------------------

- establishing: wide or medium-wide framing
- dialog: speaker isolated in frame
- action: one active performer
- reaction: one active performer

Duration: sum of merged moments, clamped to 2-10 seconds.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
shot N
type: establishing / action / dialog / reaction
moments: [list of moment numbers]
duration: estimated duration
purpose: what this shot accomplishes
camera: summary of angles and movement
visual: summary of visible elements
audio: summary of notable sounds
verification: why this shot boundary exists

------------------------------------------------------------
NOW PRODUCE THE DIRECTOR SHOT PLAN.
'''

def summarize_continuity_from_director_shots(director_shots_text: str) -> str:
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

    summary_parts = []
    if visual_desc:
        summary_parts.append(f"Previously visible: {visual_desc[-1]}.")
    if camera_desc:
        summary_parts.append(f"Camera was positioned as: {camera_desc[-1]}.")
    if audio_desc:
        summary_parts.append(f"Ambient audio included: {audio_desc[-1]}.")

    return " ".join(summary_parts).strip()

def quoted_word_count(text):
    quotes = re.findall(r'"([^"]*)"', text)
    return sum(len(q.split()) for q in quotes)

def build_beat_character_list(beat_entry: dict) -> list:
    characters = []
    seen_names = set()
    
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

def direct(beat_entry: dict, notes=''):
    if notes: 
        notes = summarize_continuity_from_director_shots(notes)

    beat_characters = build_beat_character_list(beat_entry)
    scene_parts = []
    
    if beat_entry.get('summary'):
        scene_parts.append(beat_entry['summary'])
    
    for char in beat_entry.get('active_characters', []):
        if char.get('action'):
            scene_parts.append(f"{char['name']}: {char['action']}")
        if char.get('dialog'):
            scene_parts.append(f'{char["name"]} DIALOG: "{char["dialog"]}"')
    
    scene_description = " ".join(scene_parts)

    with LLMContext() as (p_ctx, m_ctx):
        camera_log = llm_director(camera_prompt.format(
            scene_description=scene_description,
            character_list=beat_characters,
            background_label=beat_entry['background'],
            context_notes=notes
        ), processor=p_ctx, model=m_ctx)
        print(f"Camera Done: \n{camera_log}")

        director_shots = llm_director(
            director_prompt.format(
                camera_log=camera_log,
                scene_description=scene_description,
                character_list=beat_characters,
                background_label=beat_entry["background"],
                context_notes=notes
            ), processor=p_ctx, model=m_ctx
        )
        print(f"Director Done: \n{director_shots}")

        final_shotlist = llm_director(
            shot_planner_prompt.format(
                director_shot_plan=director_shots
            ), processor=p_ctx, model=m_ctx
        )
        print(f"Shotlist Done: \n{final_shotlist}")

    return final_shotlist, director_shots

# ==============================================================================
# CORE PIPELINE FUNCTIONS
# ==============================================================================

def get_or_analyze(image_path: str, prompt: str, cache_key: str, max_words: int = 15) -> str:
    """Get cached analysis from image metadata, or analyze and cache it."""
    cleaned_path = os.path.normpath(image_path)
    
    # 1. READ STEP: Safely extract existing cache
    with Image.open(cleaned_path) as img:
        cached = img.info.get(cache_key)
        if cached:
            return cached

    # 2. ANALYSIS STEP: Analyze outside of the read context handle
    result = AnalyzeImage(cleaned_path, prompt=prompt, backend='')['analysis']

    from plan10.lib.util import load_metadata
    
    # 3. WRITE STEP: Re-open a clean file stream, update, and write safely
    with Image.open(cleaned_path) as img:
        metadata = load_metadata(img)
        # Note: If load_metadata already copies img.info items, 
        # you can safely omit the secondary manual key-copy loop here.
        metadata.add_text(cache_key, result)
        
        # Load pixels into memory so Pillow drops file stream locks
        img.load() 

    # Save cleanly outside of the active file read handle
    with Image.open(cleaned_path) as out_img:
        out_img.save(cleaned_path, pnginfo=metadata)
    
    return result


if ANIME:
    from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground, add_metadata_loc
else:
    from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground, add_metadata_loc
from plan10.lib.dialog import DesignVoice

if WGP:
    from plan10.lib.wgp import GenerateVideo
elif LTX:
    from plan10.lib.ltx import GenerateVideo
elif MMH3:
    from plan10.lib.mmh3 import GenerateVideo
else:
    from plan10.lib.image_to_video import GenerateVideo

from plan10.emergent.state_manager import StateManager

def voice_prompt(gender, age):
    import random
    all_pitches = ['very low pitch', 'low pitch', 'moderate pitch', 'high pitch', 'very high pitch']

    if age == 'child':
        valid_pitches_for_char = ['moderate pitch', 'high pitch', 'very high pitch']
    elif gender == 'male' or age == 'elderly':
        valid_pitches_for_char = ['very low pitch', 'low pitch', 'moderate pitch']
    else:
        valid_pitches_for_char = ['low pitch', 'moderate pitch', 'high pitch', 'very high pitch']

    selected_pitch = random.choice(valid_pitches_for_char)
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

def replace_character_names(script, char_names):
    import unicodedata
    script = unicodedata.normalize("NFKC", script)
    segments = re.split(r'(".*?"|\'.*?\')', script)

    for cndx, name in enumerate(char_names, 1):
        token = f"char{cndx}"
        if name is None:
            continue
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        pattern_possessive = re.compile(rf"\b{re.escape(name)}'s\b", re.IGNORECASE)

        for idx, segment in enumerate(segments):
            if segment and segment[0] in {'"', "'"}:
                continue
            segment = pattern.sub(token, segment)
            segment = pattern_possessive.sub(f"{token}'s", segment)
            segments[idx] = segment

    return ''.join(segments)

def h3_ref(bg, ff, refs, portraits, prompt, duration=10.0, visual_ids=[], char_names=[], low_vram=False):
    script = ""
    char_labels = [f"char{ndx}" for ndx in range(1, len(refs) + 1)]
    
    # 1. First Frame - NO CACHE
    ff_notes = ""
    if ff:
        ff_desc = AnalyzeImage(ff, prompt='Briefly describe the scene composition, character positions, and environment. Max 15 words.')['analysis']
        script += f"ff | ff | {ff} | {ff_desc}\n"
        ff_notes = f"First frame visual context: {ff_desc}"
    
    if bg:
        # 2. Background - CACHED
        bg_desc = add_metadata_loc(bg, prompt='', seed=-1, brief=True, update=False)
        script += f"bg | bg | {bg} | {bg_desc}\n"

    bg_desc = bg_desc if bg else ff_desc
    
    # 3. Generate shots using the new director workflow
    beat_entry = {
        'summary': prompt,
        'background': bg_desc,
        'active_characters': [{'name': name} for name in char_names],
        'passive_characters': []
    }
    
    final_shotlist, director_shots = direct(beat_entry, notes=ff_notes)
    shots = final_shotlist

    shots = replace_character_names(shots, char_names)

    # Parse shots to find which characters speak (format: charX [verb] [English] "...")
    speaking_chars = set()
    for line in shots.split('\n'):
        if '[English]' in line:
            idx = line.find('[English]')
            before_english = line[:idx]
            for char in sorted(char_labels, key=len, reverse=True):
                if char in before_english:
                    speaking_chars.add(char)
                    break
            
    # 4. Characters - CACHED (only generate audio for speakers)
    portrait_entries = ''
    for ndx, ref in enumerate(refs, start=1):
        label = f"char{ndx}"
        char_desc = get_or_analyze(ref, CHAR_PROMPT, 'Description', max_words=100)
        script += f"char | {label} | {'-' if low_vram and ff else ref} | {char_desc}\n"

        if portraits:
            portrait_desc = get_or_analyze(portraits[ndx-1], FACE_PROMPT, 'Description', max_words=100)
            portrait_entries += f"portrait | portrait_{ndx} | {portraits[ndx-1]} | {label} | {portrait_desc}\n"
        elif not low_vram:
            port_path = os.path.splitext(ref)[0] + '_portrait.png'
            portrait_desc = get_or_analyze(ref, FACE_PROMPT, 'Description', max_words=100)
            portrait_entries += f"portrait | portrait_{ndx} | {port_path} | {label} | A portrait of {label}\n"
        
        if label in speaking_chars:
            voice_data = get_or_analyze(ref,
                "Identify the character's gender (male, female) and age bracket (child, teenager, young adult, middle-aged, elderly). Return exactly: 'gender, age bracket'.",
                'voice_profile')
            
            gender, age = [item.strip().lower() for item in voice_data.split(',')]
            voice_profile = voice_prompt(gender, age)
            
            wav_path = os.path.splitext(ref)[0] + '.wav'
            script += f"audio | voice_{ndx} | {wav_path} | {label} | {','.join(voice_profile)}\n"
    
    script += portrait_entries
    
    # Extract action from prompt if possible, or use a default
    action_summary = prompt
    for line in prompt.split('\n'):
        if 'Action:' in line:
            action_summary = line.replace('Action:','').strip()
            break
            
    script += f"summary | {action_summary}\n"
    script += f"soundscape | {translate_to_audio_prompt(bg_desc)}\n"
    script += shots + "\n"

    script = replace_character_names(script, char_names)
    
    return script

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    parser.add_argument('-M', '--scene-mode', action='store_true')
    parser.add_argument('--low-vram', action='store_true')
    args, _ = parser.parse_known_args()
    
    state_mgr = StateManager(args.output)
    
    if not state_mgr.exists():
        print("No state file found. Nothing to render.")
        sys.exit(0)

    state = state_mgr.load()
    video_queue = state.get('video_queue', [])
    duration = int(state.get('duration', '5'))
    refs = state.get('character_refs', [])
    portraits = state.get('portraits', [])
    initial = state.get('initial_media', '')
    visual_ids = state.get('visual_ids', [])
    char_names = state.get('char_names', [])
    bg = state.get('current_bg')
    output_dir = state.get('output_dir') or args.output
    
    # Find the first pending job
    pending_job = None
    for job in video_queue:
        if job['status'] == 'pending':
            pending_job = job
            break
    
    if not pending_job:
        print("No pending video jobs.")
        sys.exit(0)
    
    print(f"\n🎬 Rendering beat {pending_job['beat']}...")
    print(f"Input: {pending_job['input_media']}")
    print(f"Output: {pending_job['output_path']}")
    print(f"Prompt: {pending_job['prompt'][:100]}...")
    
    # Mark as processing
    pending_job['status'] = 'processing'
    state_mgr.save(state)
    
    try:
        prompt = pending_job['prompt']
        media = initial if args.scene_mode else pending_job['input_media']
        
        if isinstance(media, (list, tuple)) and len(media):
            start_image = media[0]
        elif media:
            start_image = to_absolute(media)
        else:
            start_image = None
            
        current_source = video_to_img(start_image, WIDTH, HEIGHT, True, True) if start_image else None
        current_source_path = None
        
        if current_source:
            current_source_path = Path(args.output) / 'tmp.png'
            current_source.save(str(current_source_path.resolve()))
            
        if args.scene_mode:
            script = h3_ref(None, str(current_source_path.resolve()), refs, None, prompt, duration, visual_ids=visual_ids, char_names=char_names, low_vram=args.low_vram)
        else:
            script = h3_ref(bg, None, refs, None, prompt, duration, visual_ids=visual_ids, char_names=char_names, low_vram=args.low_vram)
        Path(pending_job['output_path'].replace('.mp4', '_script.txt')).write_text(script, encoding='utf-8')

        if LTX:
            from plan10.emergent.ltx25_previewer import LTXPipeline
            converter = LTXPipeline()
            beat_out = pending_job['output_path'].replace('.mp4', '_script.txt')
            converted = converter.run(beat_out, style='', use_descriptions=False)
            print(converted)
            Path(beat_out.replace('.txt', '_ltx.txt')).write_text(
                f'RUNLENGTH (s):{converter.run_length}\n{converted}',
                encoding='utf-8'
            )

        # Mark as complete and update current_media
        pending_job['status'] = 'complete'
        state['current_media'] = pending_job['output_path']
        state_mgr.save(state)
        
        print(f"✅ Beat {pending_job['beat']} rendered successfully.")
        sys.exit(pending_job['beat'])  # Positive = success
        
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        traceback.print_exc()
        pending_job['status'] = 'failed'
        state_mgr.save(state)
        sys.exit(255)

if __name__ == "__main__":
    main()