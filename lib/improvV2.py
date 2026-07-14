import torch
import os
import gc
import random
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from image_edit import EditImage
from util import video_to_img
from image_analysis import AnalyzeImage
from config import load_environ

load_environ()
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
# ============================================================================
# VISUAL ANALYSIS (SmolVLM2)
# ============================================================================

# ============================================================================
# VISUAL ANALYSIS
# ============================================================================

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
    """Check if the character matching the visual ID is visible in the last frame.
    Uses AnalyzeImage (static image analysis) - NOT SmolVLM2."""
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    # Extract last frame if it's a video
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        last_frame_path = media_path.with_name(media_path.stem + '_lastframe.png')
        video_to_img(str(media_path), 1280, 720, True, True).save(str(last_frame_path))
        check_path = str(last_frame_path)
    else:
        check_path = str(media_path)
    
    # Use AnalyzeImage for single-frame character detection
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
    
    # Detect media type
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        media_type = "video"
        prompt_text = "Describe this video in detail. Focus on character actions and expressions, and EXPLICITLY LIST any visible props or objects in the room (e.g., ironing board, clothes on floor, mug, books, lamp)."
    else:
        media_type = "image"
        prompt_text = "Describe this image in detail. Focus on character actions and expressions, and EXPLICITLY LIST any visible props or objects in the room (e.g., ironing board, clothes on floor, mug, books, lamp)."
    
    
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
    
    # Clean up
    del model, processor, inputs, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_texts[0].split(':')[-1].strip()


# ============================================================================
# NARRATIVE DIRECTION (LLM "Yes, And...")
# ============================================================================

def generate_next_action(current_description, story_context, recent_actions, current_location):
    history_text = "\n".join([f"- {action}" for action in recent_actions[-3:]]) if recent_actions else "This is the first beat."
    
    prompt = f"""You are directing a scene using improv principles.

CURRENT SCENE:
{current_description}

CURRENT LOCATION:
{current_location}

STORY CONTEXT:
{story_context}

RECENT ACTIONS:
{history_text}

CRITICAL DIRECTOR RULES:
1. The character MUST remain clearly visible in the frame at all times.
2. The character MUST NOT leave the current location ({current_location}).
3. ACTION REQUIREMENT: Look at the CURRENT SCENE description. If there are props or objects mentioned (like an ironing board, clothes, a mug, etc.), the character MUST interact with them naturally in this next action. Do not just sit or lay there; give them a purposeful, ambient action involving the environment.

TASK:
Using "yes, and..." improv logic, describe the next 10-second action. 
Output ONLY the next action description (2-3 sentences), nothing else.
"""
    
    result = llm_analyze_media(
        media="",
        prompt=prompt,
        system="You are a scene director. Enforce strict spatial constraints and FORCE interaction with visible props.",
        max_tokens=150,
        temperature=0.7
    )['analysis']
    
    return result.strip()


def generate_character_description(story_context):
    """Use LLM to generate a detailed character description based on story context."""
    prompt = f"""Based on this story context, create a detailed character description for the protagonist.

STORY CONTEXT:
{story_context}

TASK:
Generate a detailed physical description of the main character (2-3 sentences) including:
- Age range
- Hair color and style
- Clothing style and colors
- General appearance

Output ONLY the character description (2-3 sentences), nothing else.
"""
    
    result = llm_analyze_media(
        media="",
        prompt=prompt,
        system="You are a character designer creating detailed physical descriptions.",
        max_tokens=100,
        temperature=0.7
    )['analysis']
    
    return result.strip()


# ============================================================================
# EMERGENT FEEDBACK LOOP (STABLE SINGLE-LOCATION MODE)
# ============================================================================

def emergent_feedback_loop(
    story_context,
    initial_location="bedroom",
    character_ref=None,
    initial_media=None,
    output_dir="emergent_output",
    max_beats=8,
    width=1280,
    height=720
):
    """
    Run the full emergent feedback loop with single-location stability:
    - Generate character sheet if not provided
    - Generate initial background if not provided
    - Loop: check visibility → analyze → direct → composite → animate
    - Reset to pristine background if character becomes invisible
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Generate character sheet if not provided
    if not character_ref:
        print("\n🎨 No character reference provided. Generating character sheet...")
        # Generate a basic character description for the sheet
        char_prompt = f"Create a character based on this context: {story_context}"
        
        character_ref = output_dir / "character_sheet.png"
        CreateCharacterSheet(
            prompt=char_prompt,
            output=str(character_ref),
            seed=42
        )
        character_ref = str(character_ref)
        print(f"✓ Character sheet generated: {character_ref}")
    else:
        print(f"\n✓ Using provided character reference: {character_ref}")

    # 2. Extract visual ID FROM the actual character sheet
    print("\n🔍 Extracting visual ID from character sheet...")
    visual_id = extract_visual_id_from_sheet(character_ref)
    print(f"Visual ID for matching: {visual_id}")
    
    # 2. Generate initial background if not provided
    if not initial_media:
        print(f"\n🌍 No initial media provided. Generating {initial_location} background...")
        initial_media = output_dir / f"bg_{initial_location.lower().replace(' ', '_')}.png"
        CreateBackground(
            prompt=f"{initial_location}, empty, atmospheric, detailed, wide establishing shot",
            output=str(initial_media),
            seed=42
        )
        initial_media = str(initial_media)
        print(f"✓ Initial background generated: {initial_media}")
    else:
        print(f"\n✓ Using provided initial media: {initial_media}")
    
    # Save the PRISTINE original background for resets
    original_background = initial_media
    
    # 3. Composite character into initial scene
    print(f"\n🎨 Compositing character into initial scene...")
    initial_composite = output_dir / "initial_composite.png"
    
    initial_action_prompt = f"{visual_id} at {initial_location}. Match the lighting and perspective of the environment perfectly. Maintain exact character appearance from reference."
    
    
    EditImage(
        prompt=initial_action_prompt,
        images=[initial_media, character_ref],
        output=str(initial_composite),
        width=width,
        height=height,
        seed=42
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
        
        # 1. ANALYZE: Is the character still visible?
        print(f"Visual ID for matching: {visual_id}")
        visible = is_character_visible(current_media, visual_id)
        
        # 2. Generate next action
        print("\n🎭 Generating next action...")
        description = get_visual_description(current_media)
        next_action = generate_next_action(description, story_context, history, current_location)
        print(f"LLM Director says: {next_action}")
        
        if not visible:
            # Need to edit - composite character back into scene
            print("⚠️ CHARACTER LOST! Compositing character back into scene...")
            base_image_for_edit = original_background
            edit_output = output_dir / f"edit_beat_{beat+1:03d}.png"
            
            edit_prompt = f"{visual_id} {next_action}. Match the lighting and perspective of the environment perfectly. Maintain exact character appearance from reference."
            
            EditImage(
                prompt=edit_prompt,
                images=[base_image_for_edit, character_ref],
                output=str(edit_output),
                width=width,
                height=height,
                seed=42 + beat
            )
            video_input = str(edit_output)
        else:
            # Character is visible - use last frame directly
            print("✓ Character is visible. Using last frame directly.")
            base_image_for_edit = output_dir / f"frame_beat_{beat+1:03d}.png"
            video_to_img(current_media, width, height, True, True).save(str(base_image_for_edit))
            video_input = str(base_image_for_edit)

        # Generate video from either edited or direct frame
        print("\n🎬 Animating 10-second sequence...")
        video_output = output_dir / f"video_beat_{beat+1:03d}.mp4"
        
        GenerateVideo(
            prompt=next_action,
            media=video_input,
            output=str(video_output),
            duration_sec=10,
            width=width,
            height=height,
            seed=42 + beat
        )
        print(f"✓ Video saved: {video_output}")
        
        # 5. ADVANCE: The new video becomes the current state for the next loop
        history.append(next_action)
        current_media = str(video_output)
        
        print(f"\n✅ Beat {beat + 1} complete")
        print(f"Current location: {current_location}")
        print(f"Output: {video_output}")
    
    
    print(f"\n{'='*70}")
    print(f"FEEDBACK LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {max_beats} beats in {output_dir}")
    print(f"Final location: {current_location}")
    
    return history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run emergent feedback loop for visual storytelling")
    parser.add_argument('-S', '--story', type=str, required=True, help="Story context/goals")
    parser.add_argument('-L', '--location', type=str, default="bedroom", help="Starting location")
    parser.add_argument('-C', '--character', type=str, default=None, help="Character reference sheet (optional - will generate if not provided)")
    parser.add_argument('-I', '--initial', type=str, default=None, help="Initial image or video (optional - will generate if not provided)")
    parser.add_argument('-O', '--output', type=str, default="emergent_output", help="Output directory")
    parser.add_argument('-N', '--beats', type=int, default=8, help="Number of beats to generate")
    parser.add_argument('-W', '--width', type=int, default=1280, help="Video width")
    parser.add_argument('-H', '--height', type=int, default=720, help="Video height")
    
    args = parser.parse_args()
    
    history = emergent_feedback_loop(
        story_context=args.story,
        initial_location=args.location,
        character_ref=args.character,
        initial_media=args.initial,
        output_dir=args.output,
        max_beats=args.beats,
        width=args.width,
        height=args.height
    )
    
    print("\nAction history:")
    for i, action in enumerate(history, 1):
        print(f"{i}. {action}")