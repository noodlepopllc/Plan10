import sys
import argparse
from pathlib import Path
import os, traceback

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

if ANIME:
    from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground
else:
    from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground
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
    print(f"Generated Profile Tags: {voice_profile}")
    return voice_profile


def h3_ref(bg, ff, refs, prompt, duration=10.0):
    script = ""
    print("First Frame", ff)
    
    # --- ASSETS ---
    # 1. First Frame (The actual starting composition with characters)
    if ff:
        ff_desc = AnalyzeImage(ff, prompt='Briefly describe the scene composition, character positions, and environment. Max 15 words.')['analysis']
        ff_label = "ff"
        script += f"ff | {ff_label} | {ff} | {ff_desc}\n"
    
    # 2. Background (Empty environment plate)
    bg_desc = AnalyzeImage(bg, prompt='Brief description of the empty environment/scene, no characters. Max 10 words.')['analysis']
    bg_label = "bg"
    script += f"bg | {bg_label} | {bg} | {bg_desc}\n"
    
    # 3. Characters
    char_labels = []
    for ndx, ref in enumerate(refs, start=1):
        label = f"char{ndx}"
        char_labels.append(label)
        ref_desc = AnalyzeImage(ref, prompt='Brief description of character appearance/clothing. Max 10 words.')['analysis']
        script += f"char | {label} | {ref} | {ref_desc}\n"

    # 4. Audio refs
    for ndx, ref in enumerate(refs, start=1):
        voice_label = f"voice_{ndx}"
        char_label = f"char{ndx}"
        # Expanded vision call to grab age along with gender
        analysis_result = AnalyzeImage(
            ref, 
            prompt="Identify the character's gender (male, female) and age bracket (child, teenager, young adult, middle-aged, elderly). Return exactly in this format: 'gender, age bracket'. Example: 'female, young adult'."
        )['analysis']

        # Parse the vision model response
        gender, age = [item.strip().lower() for item in analysis_result.split(',')]
        voice_profile = voice_prompt(gender, age)

        wav_path = os.path.splitext(ref)[0] + '.wav'
        script += f"audio | {voice_label} | {wav_path} | {char_label} | {','.join(voice_profile)}\n"
    
    # --- CONTEXT ---
    script += f"prompt | {prompt.replace('\n', ' ')}\n"
    script += f"soundscape | {translate_to_audio_prompt(bg_desc)}\n"
    
    # --- SHOTS (Pass the actual First Frame image for Shot 1 grounding) ---
    shots = expand_to_shots(prompt, bg_label, char_labels, duration, first_frame_path=ff)
    script += shots + "\n"
    
    return script

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
    
    char_list = ", ".join(char_labels)
    duration_hint = f"Total duration: approximately {duration} seconds."
    
    formatted_prompt = f"""You are an expert video director breaking down a scene into sequential shots.

INPUT DATA:
- Available characters: {char_list}
- Background: {bg_label}
- {duration_hint}
- Scene description: {prompt}
{scene_context}

TASK:
Generate a sequence of short video shots (2-4 seconds each) that bring the 'Scene description' to life, starting exactly from the 'VISUAL CONTEXT'.

HARD CONSTRAINTS:
1. DO NOT copy the example below. The example is ONLY to demonstrate the required format. Your content must be 100% unique and based strictly on the INPUT DATA above.
2. The video model ALREADY SEES the reference images. DO NOT describe static visual attributes. ONLY describe what CHANGES: camera movement, character actions, and facial expression shifts.
3. Output ONLY shot lines. No JSON, no markdown, no introductory text.
4. Format: shot | description | duration_seconds
5. Reference characters by their exact label: {char_list}
6. Shot 1: Based on the VISUAL CONTEXT, describe ONLY the first subtle motion that initiates the scene. DO NOT re-describe the static scene.
7. Dialogue format: character speaks [English] "exact words"
8. Structure: [ambient sounds] -> [action] -> [dialogue if any] -> [cut]. Ambient sounds FIRST, dialogue LAST.
9. If a shot has dialogue, it must END immediately after the character finishes speaking.
10. Never split speaking action from dialogue across multiple shots.
11. The LAST shot must be a medium shot showing the character(s) and environment. DO NOT end on a closeup.
12. TEMPORAL CONTINUITY: Lighting, shadows, and weather must remain identical. Actions must flow continuously.

EXAMPLE (DO NOT COPY THIS CONTENT, ONLY COPY THE FORMAT):
shot | Wide shot. Absolute silence, then a low mechanical hum. char1 floats upside down and ejects a single glowing object in zero gravity. | 3.0
shot | Closeup. Distant cosmic radiation crackling. char1's features widen in panic and it speaks [English] "Why is the butter floating?" | 2.0
shot | Medium shot. Muffled rhythmic thumping. char1 catches the object with a mechanical appendage and stares blankly at a passing space-capybara. Camera holds static. | 2.5

NOW, generate the shots for the INPUT DATA provided above:
"""

    response = llm_analyze_media('', prompt=formatted_prompt)['analysis']
    
    lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("shot |"):
            lines.append(line)
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    args, _ = parser.parse_known_args()
    
    state_mgr = StateManager(args.output)
    
    if not state_mgr.exists():
        print("No state file found. Nothing to render.")
        sys.exit(0)

    
    state = state_mgr.load()
    video_queue = state.get('video_queue', [])
    duration = int(state.get('duration', '5'))
    refs = state.get('character_refs', [])
    initial = state.get('initial_media', '')
    bg = state.get('current_bg')
    output_dir = state.get('output_dir')
    
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

        if MMH3:
            from plan10.lib.director_mmh3 import get_builder
            media = pending_job['input_media']
            if isinstance(media, list) and len(media):
                start_image = media.pop(0)
            elif media:
                start_image = to_absolute(media)
            current_source = video_to_img(start_image, 768, 448, True, True)
            current_source.save('tmp.png')
            current_source_path = f'{os.getcwd()}/tmp.png'
            script = h3_ref(bg, current_source_path, refs, prompt,  duration)
            print("SCRIPT: ",script)
            builder = get_builder(script, '')
            final_prompt = builder.generate()
            print("FINAL", final_prompt)
            
            # Extract paths dynamically from the builder instead of hardcoding
            img_refs = [data["path"] for data in builder.entities.values()]
            aud_refs = [data["path"] for data in builder.audio_refs.values()]

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
                    steps=8
                ))
            else:
                from plan10.lib.mmh3 import compose_video
                print(compose_video(final_prompt, img_refs, aud_refs, pending_job['output_path'], WIDTH, HEIGHT, builder.duration))

        else:
            if WGP and duration > 5:
                prompt = EnhancePrompt(image=pending_job['input_media'], prompt=prompt, enhancer=ENHANCE_Prompt.format(duration=duration), output=None, backend=None, ispath=False)
                enhance = False
            else:
                enhance = True

            # Generate the video
            GenerateVideo(
                prompt=prompt,
                media=pending_job['input_media'],
                output=pending_job['output_path'],
                duration_sec=float(duration),
                seed=pending_job['seed'],
                enhance=enhance
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