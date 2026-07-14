import torch
import os, sys, gc, random, json
sys.path.append('./lib')
from config import load_environ
load_environ()

from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from image_edit import EditImage
from util import video_to_img
from image_analysis import AnalyzeImage
from compositor import CompositeScene

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))
ANIME = os.environ.get("ANIME","False") != "False"
WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"

if WGP:
    from wgp import GenerateVideo
elif LTX:
    from ltx import GenerateVideo
else:
    from image_to_video import GenerateVideo

if ANIME:
    from anime_gen import CreateBackground, CreateCharacterSheet
else:
    from image_gen import CreateBackground, CreateCharacterSheet

def extract_visual_id_from_sheet(character_sheet_path):
    """Extract a visual ID by analyzing the actual character sheet image."""
    prompt = """Describe this character sheet in 5-10 words focusing ONLY on:
    - Hair color and style
    - Main clothing color/item (TOP HALF only - shirt, sweater, jacket)
    - Face/body type if distinctive

    DO NOT include shoes, pants, or accessories that might not be visible.

    Examples:
    - "blonde woman in blue dress"
    - "red-haired man in black suit"
    - "brunette woman in purple sweater"

    Output ONLY the description (5-10 words), nothing else."""
    
    result = AnalyzeImage(character_sheet_path, prompt)
    return result['analysis'].strip()


def is_character_visible(media_path, visual_id):
    """Check if the character matching the visual ID is visible in the last frame."""
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        last_frame_path = media_path.with_name(media_path.stem + '_lastframe.png')
        video_to_img(str(media_path), 1280, 720, True, True).save(str(last_frame_path))
        check_path = str(last_frame_path)
    else:
        check_path = str(media_path)
    
    prompt = f"Looking for: {visual_id}\n\nIs this person visible in the image? Answer YES or NO."
    result = AnalyzeImage(check_path, prompt)
    response = result['analysis'].strip().upper()
    
    print(f"  [DEBUG] Looking for: '{visual_id}'")
    print(f"  [DEBUG] AnalyzeImage response: '{response}'")
    
    return "YES" in response


def get_visual_description(media_path):
    """Analyze image or video with SmolVLM2, return plain English description."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        media_type = "video"
        prompt_text = "Describe this video in detail. Focus on character actions and expressions, and EXPLICITLY LIST any visible props or objects in the room."
    else:
        media_type = "image"
        prompt_text = "Describe this image in detail. Focus on character actions and expressions, and EXPLICITLY LIST any visible props or objects in the room."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": str(media_path)},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    del model, processor, inputs, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_texts[0].split(':')[-1].strip()


def generate_next_action_two_characters(
    current_description, 
    story_context, 
    recent_actions, 
    current_location,
    char_a_visible,
    char_b_visible,
    char_a_id,
    char_b_id
):
    """Generate next action for two-character scene, deciding which characters appear."""
    
    history_text = "\n".join([f"- {action}" for action in recent_actions[-3:]]) if recent_actions else "This is the first beat."
    
    visibility_status = f"""
CHARACTER VISIBILITY:
- Character A ({char_a_id}): {'VISIBLE' if char_a_visible else 'NOT VISIBLE'}
- Character B ({char_b_id}): {'VISIBLE' if char_b_visible else 'NOT VISIBLE'}
"""
    
    prompt = f"""You are directing a scene using improv principles.

CURRENT SCENE:
{current_description}

CURRENT LOCATION:
{current_location}

STORY CONTEXT:
{story_context}

RECENT ACTIONS:
{history_text}

{visibility_status}

CRITICAL DIRECTOR RULES:
1. Characters MUST remain clearly visible in the frame when present.
2. Characters MUST NOT leave the current location ({current_location}).
3. OBJECT PERMANENCE: Props and objects stay exactly where they are in the room.
4. CHARACTER SELECTION: Based on the visibility status above, decide which characters should appear in the next shot:
   - If both are visible: You can show both characters interacting, or focus on one while the other remains in frame
   - If only one is visible: You can keep showing that character, or reintroduce the missing character naturally
   - If neither is visible: Reintroduce at least one character naturally
5. NATURAL ACTIONS: Characters can interact with visible props if it makes sense, but don't have to.

TASK:
Using "yes, and..." improv logic, describe the next 10-second action. 
Start your response with either "CHARACTERS: A" (only character A), "CHARACTERS: B" (only character B), or "CHARACTERS: AB" (both characters), then describe the action.

Output format:
CHARACTERS: [A/B/AB]
[Action description in 2-3 sentences]
"""
    
    result = llm_analyze_media(
        media="",
        prompt=prompt,
        system="You are a scene director managing two characters. Maintain spatial continuity and object permanence.",
        max_tokens=200,
        temperature=0.7
    )['analysis']
    
    return result.strip()


def parse_character_selection(action_text):
    """Parse the character selection from the LLM's response."""
    lines = action_text.strip().split('\n')
    
    if lines and 'CHARACTERS:' in lines[0].upper():
        first_line = lines[0].upper()
        if 'AB' in first_line or ('A' in first_line and 'B' in first_line):
            return 'AB', '\n'.join(lines[1:]).strip()
        elif 'A' in first_line:
            return 'A', '\n'.join(lines[1:]).strip()
        elif 'B' in first_line:
            return 'B', '\n'.join(lines[1:]).strip()
    
    # Default to both if parsing fails
    return 'AB', action_text


def emergent_feedback_loop_two_characters(
    story_context,
    character_ref_a,
    character_ref_b,
    initial_location="living room",
    initial_media=None,
    output_dir="two_char_output",
    max_beats=8,
    width=1280,
    height=720
):
    """
    Two-character emergent feedback loop:
    - Manages two characters with separate visibility tracking
    - LLM decides which characters appear in each shot
    - Can show character A only, B only, or both
    - Reintroduces missing characters as needed
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Extract visual IDs for both characters
    print("\n🔍 Extracting visual IDs from character sheets...")
    visual_id_a = extract_visual_id_from_sheet(character_ref_a)
    visual_id_b = extract_visual_id_from_sheet(character_ref_b)
    print(f"Character A: {visual_id_a}")
    print(f"Character B: {visual_id_b}")
    
    # 2. Generate initial background if not provided
    if not initial_media:
        print(f"\n🌍 Generating {initial_location} background...")
        initial_media = output_dir / f"bg_{initial_location.lower().replace(' ', '_')}.png"
        CreateBackground(
            prompt=f"{initial_location}, empty, atmospheric, detailed, wide establishing shot",
            output=str(initial_media),
            seed=SEED
        )
        initial_media = str(initial_media)
        print(f"✓ Initial background generated: {initial_media}")
    else:
        print(f"\n✓ Using provided initial media: {initial_media}")
    
    original_background = initial_media
    
    # 3. Composite both characters into initial scene
    print(f"\n🎨 Compositing both characters into initial scene...")
    initial_composite = output_dir / "initial_composite.png"
    
    CompositeScene(
        background_path=initial_media,
        characters=[character_ref_a, character_ref_b],
        shot_type="two_shot",
        action="standing naturally in the room, facing towards the viewer",
        output=str(initial_composite),
        width=width,
        height=height,
        seed=SEED
    )
    print(f"✓ Initial composite saved: {initial_composite}")
    
    # 4. Run the feedback loop
    current_media = str(initial_composite)
    current_location = initial_location
    history = []
    
    for beat in range(max_beats):
        print(f"\n{'='*70}")
        print(f"BEAT {beat + 1}/{max_beats}")
        print(f"{'='*70}")
        
        # 1. ANALYZE: Check visibility of both characters
        print("\n👁️  Checking character visibility...")
        char_a_visible = is_character_visible(current_media, visual_id_a)
        char_b_visible = is_character_visible(current_media, visual_id_b)
        
        print(f"Character A ({visual_id_a}): {'VISIBLE' if char_a_visible else 'NOT VISIBLE'}")
        print(f"Character B ({visual_id_b}): {'VISIBLE' if char_b_visible else 'NOT VISIBLE'}")
        
        # 2. Generate next action with character selection
        print("\n🎭 Generating next action...")
        description = get_visual_description(current_media)
        full_action = generate_next_action_two_characters(
            description, 
            story_context, 
            history, 
            current_location,
            char_a_visible,
            char_b_visible,
            visual_id_a,
            visual_id_b
        )
        
        # Parse which characters should appear
        char_selection, next_action = parse_character_selection(full_action)
        print(f"LLM Director says: {full_action}")
        print(f"Selected characters: {char_selection}")
        
        # 3. Determine which characters to composite
        if char_selection == 'A':
            characters_to_composite = [character_ref_a]
        elif char_selection == 'B':
            characters_to_composite = [character_ref_b]
        else:  # 'AB'
            characters_to_composite = [character_ref_a, character_ref_b]
        
        # 4. Determine if we need to composite or can use last frame directly
        print(f"\n🎨 Checking if compositing is needed...")
        
        # Check if current visibility matches what LLM wants
        current_visible_chars = []
        if char_a_visible:
            current_visible_chars.append('A')
        if char_b_visible:
            current_visible_chars.append('B')
        
        # Determine if we need to composite
        needs_compositing = False
        
        if char_selection == 'AB':
            # Need both characters - composite if either is missing
            if not char_a_visible or not char_b_visible:
                needs_compositing = True
                print("⚠️ Need to composite: Missing character(s) in current frame")
        elif char_selection == 'A':
            # Need only A - composite if A is missing OR B is present (need to remove B)
            if not char_a_visible or char_b_visible:
                needs_compositing = True
                print("⚠️ Need to composite: Character state doesn't match selection")
        elif char_selection == 'B':
            # Need only B - composite if B is missing OR A is present (need to remove A)
            if not char_b_visible or char_a_visible:
                needs_compositing = True
                print("⚠️ Need to composite: Character state doesn't match selection")
        
        if needs_compositing:
            # Extract last frame as base for compositing
            base_image_for_edit = output_dir / f"frame_beat_{beat+1:03d}.png"
            video_to_img(current_media, width, height, True, True).save(str(base_image_for_edit))
            base_image_for_edit = str(base_image_for_edit)
            
            edit_output = output_dir / f"edit_beat_{beat+1:03d}.png"
            
            # Determine shot type based on character selection
            shot_type = "two_shot" if char_selection == 'AB' else "medium"
            
            CompositeScene(
                background_path=base_image_for_edit,
                characters=characters_to_composite,
                shot_type=shot_type,
                action="Facing towards viewer",
                output=str(edit_output),
                width=width,
                height=height,
                seed=SEED + beat
            )
            video_input = str(edit_output)
        else:
            # Characters are already in the correct state - use last frame directly
            print("✓ Characters already in correct state - using last frame directly")
            base_image_for_edit = output_dir / f"frame_beat_{beat+1:03d}.png"
            video_to_img(current_media, width, height, True, True).save(str(base_image_for_edit))
            video_input = str(base_image_for_edit)
        
        # 5. Generate video
        print("\n🎬 Animating 10-second sequence...")
        video_output = output_dir / f"video_beat_{beat+1:03d}.mp4"
        
        GenerateVideo(
            prompt=next_action,
            media=video_input,
            output=str(video_output),
            duration_sec=10 if WGP or LTX else 5,
            width=width,
            height=height,
            seed=SEED + beat
        )
        print(f"✓ Video saved: {video_output}")
        
        # 6. ADVANCE
        history.append(f"[{char_selection}] {next_action}")
        current_media = str(video_output)
        
        print(f"\n✅ Beat {beat + 1} complete")
        print(f"Current location: {current_location}")
        print(f"Output: {video_output}")
    
    print(f"\n{'='*70}")
    print(f"TWO-CHARACTER FEEDBACK LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {max_beats} beats in {output_dir}")
    
    return history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run two-character emergent feedback loop")
    parser.add_argument('-S', '--story', type=str, required=True, help="Story context/goals")
    parser.add_argument('-A', '--character-a', type=str, required=True, help="Character A reference sheet")
    parser.add_argument('-B', '--character-b', type=str, required=True, help="Character B reference sheet")
    parser.add_argument('-L', '--location', type=str, default="living room", help="Starting location")
    parser.add_argument('-I', '--initial', type=str, default=None, help="Initial image or video (optional)")
    parser.add_argument('-O', '--output', type=str, default="two_char_output", help="Output directory")
    parser.add_argument('-N', '--beats', type=int, default=8, help="Number of beats to generate")
    parser.add_argument('-W', '--width', type=int, default=WIDTH, help="Video width")
    parser.add_argument('-H', '--height', type=int, default=HEIGHT, help="Video height")
    
    args = parser.parse_args()
    
    history = emergent_feedback_loop_two_characters(
        story_context=args.story,
        character_ref_a=args.character_a,
        character_ref_b=args.character_b,
        initial_location=args.location,
        initial_media=args.initial,
        output_dir=args.output,
        max_beats=args.beats,
        width=args.width,
        height=args.height
    )
    
    print("\nAction history:")
    for i, action in enumerate(history, 1):
        print(f"{i}. {action}")