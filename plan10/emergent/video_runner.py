import sys
import argparse
from pathlib import Path
import os, traceback, re
from PIL import Image  # Added for image resizing

from plan10.lib.config import load_environ
load_environ()

from plan10.lib.image_analysis import EnhancePrompt, AnalyzeImage, translate_to_audio_prompt
from plan10.lib.qwen_llm import llm_analyze_media
from plan10.lib.util import video_to_img, to_absolute

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
MMH3 = os.environ.get('MMH3', 'False') != 'False'
WIDTH = int(os.environ.get('WIDTH', '768'))
HEIGHT = int(os.environ.get('HEIGHT', '448'))
ANIME = os.environ.get('ANIME', 'False') != 'False'

def get_or_analyze(image_path: str, prompt: str, cache_key: str, max_words: int = 15) -> str:
    """Get cached analysis from image metadata, or analyze and cache it."""
    img = Image.open(image_path)
    cached = img.info.get(cache_key)
    if cached:
        img.close()
        return cached
    
    result = AnalyzeImage(image_path, prompt=prompt, backend='smol')['analysis']
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

if ANIME:
    from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground, add_metadata_loc
else:
    from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground, add_metadata_loc
from plan10.lib.dialog import DesignVoice

ENHANCE_Prompt = '''You enhance rough video prompts into structured audiovisual rewrite prompts for I2VA (first-frame image → video).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Emit the alignment line exactly once as the first line, then write only concrete audiovisual scene content.

Output rules:
1) First line must be exactly:
   For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
   Then one blank line.
2) Then output exactly these two fields in order — always both; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) Shot structure: [Shot N] At MM:SS.mmm, ... where Shot 1 has no timestamp (it's the opening frame), and each subsequent shot begins with its exact start time. Each shot must specify:
   - Framing: wide shot, medium shot, close-up, extreme close-up, over-the-shoulder, etc.
   - Camera motion: natural English with motion type (pan, tilt, dolly, zoom, handheld, tracking) and when meaningful, amplitude (small/large) and speed (slow/fast)
   - Subject action: what the characters/subjects are doing in this shot
5) Dialogue must NEVER be placed on a standalone line or appended to the end of the text. It must be woven directly inside the action sentence describing the speaker's lip movements using the format: saying in an on-screen voice "exact words". Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
6) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
7) Minimize descriptive tokens for static visual attributes (clothing colors, hair styles) already present in <Picture 1>. Open [Shot 1] by describing the initial composition and spatial relationships, then immediately transition with a clean motion trigger: "Breaking their initial layout, the characters activate smoothly into motion."
8) For each shot, budget the motion chronologically using explicit sequential time intervals formatted strictly in standard MM:SS.mmm timecode (e.g., "From 00:00.000 to 00:03.000...", "From 00:03.000 to 00:06.000..."). For each time block within a shot, assign exactly ONE primary subject action and ONE camera movement. The dialogue tag string must sit directly inside the specific timestamp block where it is being actively spoken. Define exactly when spoken dialogue ends and when lips close to halt motion.
9) Total video duration: Structure the entire sequence to match the requested duration. Use as many shots as needed to tell the complete story, ensuring smooth transitions between shots.
Duration: {duration} seconds.
'''

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
    #print(f"Generated Profile Tags: {voice_profile}")
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

BG_PROMPT = '''
Return ONE sentence in this exact format:

"The scene shows {environment description} with {key visual element}."

Use ONLY these slots. Do not reorder them.

DEFINITIONS:
- {environment description} is a short phrase describing the empty environment (no characters).
- {key visual element} is one notable object, structure, or terrain feature.
- Keep the entire sentence brief (under ~12 words naturally).
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
    cndx = 1
    for char_name in char_names:
        shots = shots.replace(char_name.lower(), f'char{cndx}')
        shots = shots.replace(char_name.capitalize(), f'char{cndx}' )
        cndx += 1


    
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

    cndx = 1
    for char_name in char_names:
        script = script.replace(char_name.lower(), f'char{cndx}')
        script = script.replace(char_name.capitalize(), f'char{cndx}' )
        cndx += 1
    
    return script

def normalize_shot_characters(shot_text: str, char_labels: list) -> str:
    """Replace character names with char tokens in a single shot."""
    result = shot_text
    
    # Sort by length descending to avoid partial replacements
    # (e.g., "Sarah" before "Sara" if both exist)
    sorted_labels = sorted(char_labels, key=len, reverse=True)
    
    for i, label in enumerate(sorted_labels, 1):
        # Find the original index for this label
        original_index = char_labels.index(label)
        token = f"char{original_index + 1}"
        
        # Case-insensitive word boundary replacement
        result = re.sub(rf'\b{re.escape(label)}\b', token, result, flags=re.IGNORECASE)
    
    return result

def expand_to_shots(prompt: str, bg_label: str, char_labels: list, duration: float, first_frame_path: str = None) -> str:
    """Returns raw shot lines ready to append to your script, grounded in the actual first frame."""

    scene_context = ""
    if first_frame_path and os.path.exists(first_frame_path):
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
Generate a sequence of cinematic shots that follow the scene description and maintain visual continuity. Use as many shots as needed.

SHOT DURATION GUIDELINES (use whole seconds only):
- Quick dialogue (1-5 words): 1 second
- Medium dialogue (6-15 words): 2 seconds
- Simple actions (turn, look, gesture): 2 seconds
- Complex actions (crawl, stand up, walk): 3-4 seconds
- Reaction shots: 2 seconds

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
   - Max 15 words per shot. Break long dialogue into multiple shots.

6. Foley:
   - EVERY shot MUST begin with a foley cue.

7. Continuity:
   - Lighting, shadows, and weather remain identical.
   - Actions flow continuously between shots.

FORMAT:
shot | foley + description | duration_seconds

EXAMPLE:
shot | Low wind through rafters. Medium shot. char1 shifts her stance, glancing toward char2. | 2
shot | Soft creak of wood. Medium shot of char1 facing char2. char1 speaks [English] "Stay back." | 1
shot | Distant hoofbeats. Medium shot. char2 reacts with a quick blink. | 2

NOW, generate the shots for the INPUT DATA provided above:
"""

    response = llm_analyze_media('', prompt=formatted_prompt, max_tokens=8192, temperature=0.4)['analysis']

    lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("shot |"):
            line = normalize_shot_characters(line, char_labels)
            lines.append(line)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    parser.add_argument('-F', '--fast', action='store_true')
    parser.add_argument('--debug', action='store_true')
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
    visual_ids = state.get('visual_ids',[])
    char_names = state.get('char_names', [])
    bg = state.get('current_bg')
    output_dir = state.get('output_dir') or args.output
    
    '''
    # --- MEMORY OPTIMIZATION FOR MMH3 ---
    if MMH3:
        os.makedirs(output_dir, exist_ok=True)
        # Fallback for older Pillow versions that don't have Image.Resampling
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS 
        # Resize background to target video resolution (e.g., 768x448)
        if bg and os.path.exists(bg):
            bg_img = Image.open(bg).convert("RGB")
            if bg_img.width > bg_img.height:
                bg_img = bg_img.resize((768, 448), resample_filter)
            else:
                bg_img = bg_img.resize((448, 768), resample_filter)
            bg_resized = os.path.join(output_dir, "resized_bg.png")
            bg_img.save(bg_resized)
            bg_img.close()
            bg = bg_resized
            
        # Resize character references to 512x512
        resized_refs = []
        for i, ref in enumerate(refs):
            if ref and os.path.exists(ref):
                ref_img = Image.open(ref).convert("RGB")
                ref_img = ref_img.resize((512, 512), resample_filter)
                ref_resized = os.path.join(output_dir, f"resized_ref_{i}.png")
                ref_img.save(ref_resized)
                ref_img.close()
                resized_refs.append(ref_resized)
            else:
                resized_refs.append(ref)
        refs = resized_refs
    '''
    # ------------------------------------

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

        from plan10.lib.director_mmh3 import get_builder
        media = pending_job['input_media']
        if isinstance(media, (list, tuple)) and len(media):
            start_image = media[0]
        elif media:
            start_image = to_absolute(media)
        current_source = video_to_img(start_image, WIDTH, HEIGHT, True, True)
        current_source.save('tmp.png')
        current_source_path = f'{os.getcwd()}/tmp.png'
        script = h3_ref(bg, None, refs, portraits, prompt,  duration, visual_ids=visual_ids)
        Path(pending_job['output_path'].replace('.mp4', '_script.txt')).write_text(script)
        if args.debug:
            # Mark as complete and update current_media
            pending_job['status'] = 'complete'
            state['current_media'] = pending_job['output_path']
            state_mgr.save(state)
            
            print(f"✅ Beat {pending_job['beat']} rendered successfully.")
            sys.exit(pending_job['beat'])  # Positive = success

        if MMH3:
            builder = get_builder(script, '')
            final_prompt = builder.generate()
            print("FINAL", final_prompt)
            
            # Extract paths dynamically from the builder instead of hardcoding
            img_refs = [data["path"] for data in builder.entities.values()]
            aud_refs = [data["path"] for data in builder.used_audio_refs.values()]
            '''

            if WGP:
                import asyncio
                from plan10.lib.director_mmh3 import send

                asyncio.run(send(
                    final_prompt, 
                    img_refs, 
                    aud_refs, 
                    output=Path(pending_job['output_path']).name, 
                    width=WIDTH, 
                    height=HEIGHT, 
                    duration=builder.duration,
                    steps=4 if args.fast else 8, 
                    upscale=False
                ))
            else:
                from plan10.lib.mmh3 import compose_video
                print(compose_video(final_prompt, img_refs, aud_refs, pending_job['output_path'], WIDTH, HEIGHT, builder.duration))
            '''

        else:
            from plan10.emergent.ltx25_previewer import LTXPipeline
            converter = LTXPipeline()
            beat_out = pending_job['output_path'].replace('.mp4', '_script.txt')
            converted = converter.run(beat_out, style='', use_descriptions=False)
            print(converted)
            Path(beat_out.replace('.txt', '_ltx.txt')).write_text(f'RUNLENGTH (s):{converter.run_length}\n{converted}')

            '''
            # Generate the video
            GenerateVideo(
                prompt=converted,
                media=pending_job['input_media'],
                output=pending_job['output_path'],
                duration_sec=float(converter.run_length),
                seed=pending_job['seed'],
                enhance=False
            )
            '''
        
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