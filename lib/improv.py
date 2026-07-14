import torch, os, sys, cv2, gc
import numpy as np
sys.path.append('./lib')
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from util import video_to_img
from config import load_environ
from image_edit import FrameDetailer, EditImage
from image_gen import GenerateImage
from uniface.detection import RetinaFace

load_environ()

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if WGP:
    from wgp import GenerateVideo
elif LTX:
    from ltx import GenerateVideo
else:
    from image_to_video import GenerateVideo


def cleanup():
    """Clear VRAM after model usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_visual_description(media_path):
    """Analyze image/video with SmolVLM2."""
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    media_type = "video" if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'] else "image"
    
    prompt = """Describe this in detail. MUST include:
- Character actions and expressions
- EXACT clothing: top color/style, pants/bottom color/style, shoes, accessories
- Hair color and style
- Environment and props
Be extremely specific about clothing."""
    
    messages = [{"role": "user", "content": [
        {"type": media_type, "path": str(media_path)},
        {"type": "text", "text": prompt}
    ]}]
    
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    del model, processor, inputs, generated_ids
    cleanup()
    
    return result.split(':')[-1].strip()


def generate_next_action(description, story_context, history):
    """Generate next action using LLM."""
    history_text = "\n".join([f"- {a}" for a in history[-3:]]) if history else "First beat."
    
    prompt = f"""CURRENT SCENE: {description}
STORY CONTEXT: {story_context}
RECENT ACTIONS: {history_text}

Describe what happens next (1-2 sentences). Build naturally on what you see."""
    
    result = llm_analyze_media(
        media="", prompt=prompt,
        system="Scene director. Be concise.",
        max_tokens=100, temperature=0.7
    )['analysis']
    
    return result.strip()


def is_face_visible(media_path):
    """Check if face is visible."""
    detector = RetinaFace()
    
    media_path = Path(media_path)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        frame = video_to_img(str(media_path), WIDTH, HEIGHT, True, True)
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(media_path))
    
    faces = detector.detect(img)
    
    del detector
    cleanup()
    
    return len(faces) > 0


def recreate_frame(current_media, description, output_path):
    """Recreate frame with character facing camera."""
    detailer = FrameDetailer()
    
    media_path = Path(current_media)
    ext = media_path.suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        frame = video_to_img(current_media, WIDTH, HEIGHT, True, True)
    else:
        frame = Image.open(current_media)
    
    prompt = f"{description}. Character facing camera, front view, face clearly visible."
    
    result = EditImage(
        prompt=prompt, ref_image=frame, output=str(output_path),
        width=WIDTH, height=HEIGHT, strength=0.6
    )
    
    enhanced = detailer.enhance(result['output_path'])['output_path']
    
    del detailer
    cleanup()
    
    return enhanced


def feedback_loop(initial_media, story_context, output_dir="feedback_output", max_beats=8):
    """Run feedback loop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    current_media = initial_media
    
    history = []
    
    for beat in range(max_beats):
        print(f"\n{'='*60}\nBEAT {beat + 1}/{max_beats}\n{'='*60}")
        
        # Check face visibility
        if not is_face_visible(current_media):
            print("⚠️ Face not visible - recreating...")
            description = get_visual_description(current_media)
            recreated_path = output_dir / f"beat_{beat+1:03d}_recreated.png"
            current_media = recreate_frame(current_media, description, recreated_path)
        
        # Analyze and direct
        description = get_visual_description(current_media)
        next_action = generate_next_action(description, story_context, history)
        
        # Generate video
        output_path = output_dir / f"beat_{beat+1:03d}.mp4"
        i2v_prompt = f"{description}. {next_action}"
        
        GenerateVideo(
            prompt=i2v_prompt, media=current_media, output=str(output_path),
            duration_sec=10 if WGP or LTX else 5, seed=SEED + beat
        )
        
        # Upscale for next iteration
        current_media = str(output_path)
        
        history.append(next_action)
        print(f"✅ Beat {beat + 1} complete")
    
    print(f"\n{'='*60}\nCOMPLETE: {max_beats} beats\n{'='*60}")
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--initial', type=str, default='', help="Initial image/video")
    parser.add_argument('-P', '--prompt', type=str, default='', help="Prompt for initial image")
    parser.add_argument('-C', '--context', type=str, required=True, help="Story context")
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    parser.add_argument('-N', '--beats', type=int, default=8)
    
    args = parser.parse_args()
    
    if not args.initial:
        if not args.prompt:
            print("Error: --initial or --prompt required")
            sys.exit(1)
        GenerateImage(prompt=args.prompt, output='improv.png', width=WIDTH, height=HEIGHT, seed=SEED)
        args.initial = 'improv.png'
    
    feedback_loop(args.initial, args.context, args.output, args.beats)