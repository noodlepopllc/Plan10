import torch, os, sys, cv2
sys.path.append('./lib')
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from util import video_to_img
from config import load_environ
from image_edit import FrameDetailer, EditImage
from image_analysis import AnalyzeImage
from image_gen import GenerateImage
from uniface.detection import RetinaFace

load_environ()

WGP = os.environ.get("WGP","False") != "False"
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if WGP:
    from wgp import GenerateVideo
else:
    from image_to_video import GenerateVideo


# SmolVLM2 setup
def get_visual_description(media_path):
    """Analyze image or video with SmolVLM2, return plain English description."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Detect media type from file extension
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        media_type = "video"
        prompt_text = """Describe this video in detail. You MUST include:
- Character actions and expressions
- EXACT clothing: top/shirt color and style, pants/bottom color and style, shoes, accessories
- Hair color and style
- Environment and props
Be extremely specific about clothing colors and styles."""
    else:
        media_type = "image"
        prompt_text = """Describe this image in detail. You MUST include:
- Character actions and expressions
- EXACT clothing: top/shirt color and style, pants/bottom color and style, shoes, accessories
- Hair color and style
- Environment and props
Be extremely specific about clothing colors and styles."""
    
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
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_texts[0].split(':')[-1].strip()


def generate_next_action(current_description, story_context, recent_actions):
    """Use LLM to generate next action using 'yes, and...' improv logic."""
    
    history_text = "\n".join([f"- {action}" for action in recent_actions[-3:]]) if recent_actions else "This is the first beat."
    
    prompt = f"""You are directing a scene using improv principles.

CURRENT SCENE:
{current_description}

STORY CONTEXT:
{story_context}

RECENT ACTIONS:
{history_text}

TASK:
Using "yes, and..." improv logic, describe what happens next. Accept what you see as canonical and build naturally on it. Focus on the next physical action or transition.

Output ONLY the next action description (1-2 sentences), nothing else.
"""
    
    result = llm_analyze_media(
        media="",
        prompt=prompt,
        system="You are a scene director continuing a visual story. Be concise and action-focused.",
        max_tokens=100,
        temperature=0.7
    )['analysis']
    
    return result.strip()


def upscale(media):
    detailer = FrameDetailer()
    status = detailer.enhance(media)
    del detailer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return status['output_path']


def is_face_visible(media_path):
    """Check if character's face is visible (not back to camera).
    Simplified version - just checks if ANY face is detected."""
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        last_frame_path = media_path.with_name(media_path.stem + '_lastframe.png')
        video_to_img(str(media_path), WIDTH, HEIGHT, True, True).save(str(last_frame_path))
        check_path = str(last_frame_path)
    else:
        check_path = str(media_path)
    
    # Check if face is visible using RetinaFace
    img = cv2.imread(check_path)
    detector = RetinaFace()
    faces = detector.detect(img)
    
    if not faces:
        return False, "no_face_detected"
    
    return True, "face_visible"


def recreate_frame_facing_camera(current_media, description, output_path):
    """Use img2img to recreate frame with character facing camera."""
    
    # Extract current frame
    media_path = Path(current_media)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        frame = video_to_img(current_media, WIDTH, HEIGHT, True, True)
    else:
        from PIL import Image
        frame = Image.open(current_media)
    
    # Use img2img with prompt to turn character around
    prompt = f"{description}. Character is facing the camera, front view, face clearly visible."
    
    result = EditImage(
        prompt=prompt,
        ref_image=frame,
        output=str(output_path),
        width=WIDTH,
        height=HEIGHT,
        strength=0.6
    )
    
    return result['output_path']


def feedback_loop(initial_media, story_context, output_dir="feedback_output", max_beats=8):
    """Run the feedback loop: analyze → generate next action → render → repeat."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔧 Initializing frame detailer...")
    
    current_media = upscale(initial_media)
    history = []
    
    for beat in range(max_beats):
        print(f"\n{'='*60}")
        print(f"BEAT {beat + 1}/{max_beats}")
        print(f"{'='*60}")
        
        # 1. Check face visibility
        print("👁️ Checking face visibility...")
        face_visible, status = is_face_visible(current_media)
        print(f"Face status: {status}")
        
        if not face_visible:
            print("⚠️ Face not visible - recreating frame...")
            description = get_visual_description(current_media)
            recreated_path = output_dir / f"beat_{beat+1:03d}_recreated.png"
            current_media = recreate_frame_facing_camera(current_media, description, recreated_path)
            current_media = upscale(current_media)
        
        # 2. Analyze current visual
        print("📊 Analyzing current visual...")
        description = get_visual_description(current_media)
        print(f"Description: {description[:100]}...")
        
        # 3. Generate next action
        print("\n🎭 Generating next action...")
        next_action = generate_next_action(description, story_context, history)
        print(f"Next action: {next_action}")
        
        # 4. Build I2V prompt
        i2v_prompt = f"{description}. {next_action}"
        print(f"\n🎬 I2V prompt: {i2v_prompt[:100]}...")
        
        # 5. Generate video
        output_path = output_dir / f"beat_{beat+1:03d}.mp4"
        print(f"\n🎥 Generating video: {output_path}")
        
        result = GenerateVideo(
            prompt=i2v_prompt,
            media=str(current_media),
            output=str(output_path),
            duration_sec=10 if WGP else 5,
            seed=42 + beat
        )
        
        # 6. Update state
        history.append(next_action)
        current_media = upscale(str(output_path))
        
        print(f"\n✅ Beat {beat + 1} complete")
    
    
    print(f"\n{'='*60}")
    print(f"FEEDBACK LOOP COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {max_beats} beats in {output_dir}")
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feedback loop for visual storytelling")
    parser.add_argument('-I', '--initial', type=str, default='', required=False, help="Initial image or video")
    parser.add_argument('-P', '--prompt', type=str, default='', required=False, help="Prompt for initial image generation")
    parser.add_argument('-C', '--context', type=str, required=True, help="Story context/goals")
    parser.add_argument('-O', '--output', type=str, default="feedback_output", help="Output directory")
    parser.add_argument('-N', '--beats', type=int, default=8, help="Number of beats to generate")
    
    args = parser.parse_args()
    initial = args.initial
    if not initial:
        if not args.prompt:
            print("Error: Must provide either --initial or --prompt")
            sys.exit(1)
        GenerateImage(prompt=args.prompt, output='improv.png', width=WIDTH, height=HEIGHT, seed=SEED)
        initial = 'improv.png'
    
    feedback_loop(
        initial_media=initial,
        story_context=args.context,
        output_dir=args.output,
        max_beats=args.beats
    )