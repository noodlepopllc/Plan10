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
from image_analysis import AnalyzeImage
from uniface.detection import RetinaFace
from compositor import CompositeScene

load_environ()

WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
ANIME = os.environ.get("ANIME", "False") != "False"
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))

if ANIME:
    from anime_gen import GenerateImage
else:
    from image_gen import GenerateImage

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
        
        # Extract visual ID once from character reference
        print("\n🔍 Extracting visual ID from character reference...")
        self.visual_id = self.extract_visual_id(character_ref)
        print(f"Visual ID: {self.visual_id}")
    
    def extract_visual_id(self, character_ref_path):
        """Extract a short visual identifier from the character reference."""
        prompt = """Describe this character in 5-10 words focusing ONLY on:
- Hair color and style
- Main clothing color/item (TOP HALF only - shirt, sweater, jacket)
- Face/body type if distinctive

DO NOT include shoes, pants, or accessories.

Examples:
- "blonde woman in blue dress"
- "red-haired man in black suit"
- "brunette woman in purple sweater"

Output ONLY the description (5-10 words), nothing else."""
        
        result = AnalyzeImage(character_ref_path, prompt)
        return result['analysis'].strip()
    
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

CRITICAL DIRECTOR RULES:
1. Character MUST remain clearly visible in the frame.
2. OBJECT PERMANENCE: Props and objects stay where they are. If the character interacts with an object, they must move to it naturally. Objects do not teleport.
3. Build naturally on what you see using "yes, and..." improv logic.

Describe what happens next (1-2 sentences)."""
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="Scene director. Maintain spatial continuity and object permanence.",
            max_tokens=150, temperature=0.7
        )['analysis']
        
        return result.strip()
    
    def _extract_frame_for_check(self, media_path):
        """Extract a frame from media for analysis, returns (cv2_image, path_for_analyzeimage)."""
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frame = video_to_img(str(media_path), self.width, self.height, True, True)
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            check_path = self.output_dir / f"check_{len(self.history):03d}.png"
            frame.save(str(check_path))
            check_path = str(check_path)
        else:
            img = cv2.imread(str(media_path))
            check_path = str(media_path)
        
        return img, check_path
    
    def is_character_adequately_visible(self, media_path):
        """Two-tier visibility check:
        - Tier 1: RetinaFace (fast) - is ANY face visible and facing camera?
        - Tier 2: AnalyzeImage (slower) - is it the RIGHT character?
        
        Returns: (visible: bool, reason: str)
        """
        img, check_path = self._extract_frame_for_check(media_path)
        
        # TIER 1: RetinaFace - fast pose check
        detector = RetinaFace()
        faces = detector.detect(img)
        del detector
        cleanup()
        
        if not faces:
            print(f"  [DEBUG] RetinaFace: NO faces detected")
            return False, "no_face"
        
        print(f"  [DEBUG] RetinaFace: {len(faces)} face(s) detected")
        
        # TIER 2: AnalyzeImage - identity check (only if face exists)
        prompt = f"""Looking for: {self.visual_id}

Is this specific person clearly visible in the image?
Focus on: hair color/style, clothing color/style, face/body type.

Answer YES if this person is clearly visible, NO if not."""
        
        result = AnalyzeImage(check_path, prompt)
        response = result['analysis'].strip().upper()
        
        print(f"  [DEBUG] AnalyzeImage for '{self.visual_id}': {response}")
        
        if "YES" in response:
            return True, "visible"
        else:
            return False, "wrong_character"
    
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
            characters=[],
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
            
            # Two-tier visibility check
            print(f"\n👁️ Checking character visibility (looking for: {self.visual_id})...")
            visible, reason = self.is_character_adequately_visible(self.current_media)
            
            if not visible:
                if reason == "no_face":
                    print("⚠️ No face visible (back to camera?) - recreating frame...")
                else:
                    print("⚠️ Wrong character detected - recreating frame...")
                
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