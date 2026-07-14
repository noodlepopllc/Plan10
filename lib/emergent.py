import torch, os, sys, cv2, gc, time
import numpy as np
sys.path.append('./lib')
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from util import video_to_img
from config import load_environ
from image_edit import EditImage
from image_gen import GenerateImage
from uniface.detection import RetinaFace
from compositor import CompositeScene

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


class FeedbackLoop:
    """Single-character emergent feedback loop for visual storytelling."""
    
    def __init__(self, character_ref, output_dir="feedback_output", width=WIDTH, height=HEIGHT, seed=SEED):
        self.character_ref = character_ref
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.width = width
        self.height = height
        self.seed = seed
        self.history = []
        self.current_media = None
    
    def get_visual_description(self, media_path):
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
    
    def generate_next_action(self, description, story_context):
        """Generate next action using LLM."""
        history_text = "\n".join([f"- {a}" for a in self.history[-3:]]) if self.history else "First beat."
        
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
    
    def is_face_visible(self, media_path):
        """Check if face is visible using RetinaFace."""
        detector = RetinaFace()
        
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frame = video_to_img(str(media_path), self.width, self.height, True, True)
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(media_path))
        
        faces = detector.detect(img)
        
        del detector
        cleanup()
        
        return len(faces) > 0
    
    def recreate_frame(self, media_path, description):
        """Recreate frame using two-step compositor flow: strip characters → composite back."""
        
        beat_num = len(self.history)
        
        # Step 1: Extract last frame and strip characters to get clean background
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            last_frame = video_to_img(str(media_path), self.width, self.height, True, True)
        else:
            last_frame = Image.open(media_path)
        
        last_frame_path = self.output_dir / f"last_frame_{beat_num:03d}.png"
        last_frame.save(str(last_frame_path))
        
        # Strip characters to get clean background
        clean_bg_path = self.output_dir / f"clean_bg_{beat_num:03d}.png"
        print("  → Stripping characters to establish clean background...")
        
        CompositeScene(
            background_path=str(last_frame_path),
            characters=[],  # Empty = establishing shot, no humans
            shot_type="establishing",
            action="maintain exact environment, lighting, and props, but ensure no people are present",
            output=str(clean_bg_path),
            width=self.width,
            height=self.height,
            seed=self.seed + beat_num
        )
        
        # Step 2: Composite character back onto clean background
        composite_path = self.output_dir / f"recreated_{beat_num:03d}.png"
        print(f"  → Compositing character onto clean background...")
        
        composite_action = f"Character facing camera, front view, face clearly visible. {description}"
        
        CompositeScene(
            background_path=str(clean_bg_path),
            characters=[self.character_ref],
            shot_type="medium",
            action=composite_action,
            output=str(composite_path),
            width=self.width,
            height=self.height,
            seed=self.seed + beat_num
        )
        
        return str(composite_path)
    
    def run(self, initial_media, story_context, max_beats=8):
        """Run the feedback loop."""
        self.current_media = initial_media
        self.history = []
        
        for beat in range(max_beats):
            print(f"\n{'='*60}\nBEAT {beat + 1}/{max_beats}\n{'='*60}")
            
            # Check face visibility
            if not self.is_face_visible(self.current_media):
                print("⚠️ Face not visible - recreating frame...")
                description = self.get_visual_description(self.current_media)
                self.current_media = self.recreate_frame(self.current_media, description)
            
            # Analyze and direct
            description = self.get_visual_description(self.current_media)
            next_action = self.generate_next_action(description, story_context)
            
            # Generate video
            output_path = self.output_dir / f"beat_{beat+1:03d}.mp4"
            i2v_prompt = f"{description}. {next_action}"
            
            GenerateVideo(
                prompt=i2v_prompt, media=self.current_media, output=str(output_path),
                duration_sec=10 if WGP or LTX else 5, seed=self.seed + beat
            )
            
            # Update state
            self.current_media = str(output_path)
            self.history.append(next_action)
            
            print(f"✅ Beat {beat + 1} complete")
        
        print(f"\n{'='*60}\nCOMPLETE: {max_beats} beats\n{'='*60}")
        return self.history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--ref', type=str, required=True, help="Character reference image")
    parser.add_argument('-I', '--initial', type=str, default='', help="Initial image/video")
    parser.add_argument('-P', '--prompt', type=str, default='', help="Prompt for initial image")
    parser.add_argument('-C', '--context', type=str, required=True, help="Story context")
    parser.add_argument('-O', '--output', type=str, default="feedback_output")
    parser.add_argument('-N', '--beats', type=int, default=8)
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-S', '--seed', type=int, default=SEED)
    
    args = parser.parse_args()
    
    initial = args.initial
    if not initial:
        if not args.prompt:
            print("Error: --initial or --prompt required")
            sys.exit(1)
        GenerateImage(prompt=args.prompt, output='improv.png', width=args.width, height=args.height, seed=args.seed)
        initial = 'improv.png'
    
    loop = FeedbackLoop(
        character_ref=args.ref,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    history = loop.run(initial, args.context, args.beats)
    
    print("\nAction history:")
    for i, action in enumerate(history, 1):
        print(f"{i}. {action}")