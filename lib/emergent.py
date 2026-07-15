
import sys
sys.path.append('./lib')
from config import load_environ
load_environ()
import torch, os, cv2, gc, time
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_llm import llm_analyze_media
from util import video_to_img
from image_edit import EditImage
from image_analysis import AnalyzeImage
from uniface.detection import RetinaFace
from compositor import CompositeScene


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

class CharacterProfile:
    """Cached character profile - ground truth for appearance."""
    
    def __init__(self, character_ref_path):
        self.ref_path = character_ref_path
        self.visual_id = self._extract_visual_id()
        self.clothing = self._extract_clothing()
        self.appearance = self._extract_appearance()
    
    def _extract_visual_id(self):
        """Short identifier for visibility checks."""
        prompt = """Describe this character in 5-10 words focusing ONLY on:
- Hair color and style
- Main clothing color/item (TOP HALF only)
- Face/body type if distinctive

Output ONLY the description (5-10 words), nothing else."""
        return AnalyzeImage(self.ref_path, prompt)['analysis'].strip()
    
    def _extract_clothing(self):
        """Detailed clothing description - stable ground truth."""
        prompt = """Describe this character's clothing in detail:
- Top/shirt: color, style, fit, pattern
- Bottom/pants: color, style, fit
- Shoes: type, color
- Accessories: jewelry, bags, etc.
- Hair: color, style, length

Be extremely specific about colors and styles. This description will be used to maintain consistency."""
        return AnalyzeImage(self.ref_path, prompt)['analysis'].strip()
    
    def _extract_appearance(self):
        """Physical appearance details."""
        prompt = """Describe this character's physical appearance:
- Age range
- Gender
- Hair color and style
- Face features
- Body type/build
- Any distinctive features

Be specific. This will be used to maintain character consistency."""
        return AnalyzeImage(self.ref_path, prompt)['analysis'].strip()
    
    def get_full_description(self):
        """Complete character description for prompts."""
        return f"""CHARACTER PROFILE:
Visual ID: {self.visual_id}
Appearance: {self.appearance}
Clothing: {self.clothing}"""


class FeedbackLoop:
    """Feedback loop with reality checking."""
    
    def __init__(self, character_ref, output_dir="feedback_output", width=WIDTH, height=HEIGHT, seed=SEED):
        self.character_ref = character_ref
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.width = width
        self.height = height
        self.seed = seed
        
        # Cached character profile - ground truth
        print("\n🔍 Building character profile from reference...")
        self.character_profile = CharacterProfile(character_ref)
        print(f"Visual ID: {self.character_profile.visual_id}")
        
        self.history = []
        self.current_media = None
    
    def analyze_reality(self, media_path, intended_action):
        """SmolVLM2 analyzes what we ACTUALLY created vs what we intended."""
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        media_type = "video" if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'] else "image"
        
        prompt = f"""We intended to create this: "{intended_action}"

Analyze what ACTUALLY happened in this {media_type}:
1. What is the character doing? (actions, expressions)
2. What props/objects are visible?
3. Any issues or unexpected elements?

Be factual about what you see, not what was intended."""
        
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
    
    def compare_and_decide(self, intended_action, actual_reality):
        """Compare prompt vs reality, decide next action."""
        prompt = f"""INTENDED ACTION: {intended_action}

ACTUAL RESULT: {actual_reality}

CHARACTER PROFILE:
{self.character_profile.get_full_description()}

RECENT HISTORY:
{chr(10).join([f"- {a}" for a in self.history[-3:]]) if self.history else "First beat."}

TASK:
1. Did the actual result match the intention? (YES/PARTIAL/NO)
2. Are there any issues to fix? (character drift, wrong action, missing elements)
3. What should happen next?

Output format:
MATCH: [YES/PARTIAL/NO]
ISSUES: [list any problems or "none"]
NEXT: [what should happen next, 1-2 sentences]"""
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="Scene director analyzing feedback and planning next steps.",
            max_tokens=200, temperature=0.7
        )['analysis']
        
        return result.strip()
    
    def parse_decision(self, decision_text):
        """Parse the decision output."""
        lines = decision_text.split('\n')
        match = "UNKNOWN"
        issues = "none"
        next_action = ""
        
        for line in lines:
            if line.startswith("MATCH:"):
                match = line.split(":", 1)[1].strip()
            elif line.startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()
            elif line.startswith("NEXT:"):
                next_action = line.split(":", 1)[1].strip()
        
        return match, issues, next_action
    
    def is_character_adequately_visible(self, media_path):
        """Two-tier visibility check."""
        img, check_path = self._extract_frame_for_check(media_path)
        
        # Tier 1: RetinaFace
        detector = RetinaFace()
        faces = detector.detect(img)
        del detector
        cleanup()
        
        if not faces:
            return False, "no_face"
        
        # Tier 2: AnalyzeImage identity check
        prompt = f"""Looking for: {self.character_profile.visual_id}

Is this specific person clearly visible in the image?
Answer YES or NO."""
        
        result = AnalyzeImage(check_path, prompt)
        response = result['analysis'].strip().upper()
        
        if "YES" in response:
            return True, "visible"
        else:
            return False, "wrong_character"
    
    def _extract_frame_for_check(self, media_path):
        """Extract frame for analysis."""
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frame = video_to_img(str(media_path), self.width, self.height, True, True)
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            check_path = self.output_dir / f"check_{len(self.history):03d}.png"
            frame.save(str(check_path))
        else:
            img = cv2.imread(str(media_path))
            check_path = str(media_path)
        
        return img, str(check_path)
    
    def recreate_frame(self, media_path, next_action):
        """Recreate frame using two-step compositor."""
        beat_num = len(self.history)
        
        # Step 1: Strip characters
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            last_frame = video_to_img(str(media_path), self.width, self.height, True, True)
        else:
            last_frame = Image.open(media_path)
        
        last_frame_path = self.output_dir / f"last_frame_{beat_num:03d}.png"
        last_frame.save(str(last_frame_path))
        
        clean_bg_path = self.output_dir / f"clean_bg_{beat_num:03d}.png"
        print("  → Stripping characters...")
        
        CompositeScene(
            background_path=str(last_frame_path),
            characters=[],
            shot_type="establishing",
            action="maintain environment, no people",
            output=str(clean_bg_path),
            width=self.width,
            height=self.height,
            seed=self.seed + beat_num
        )
        
        # Step 2: Composite character back
        composite_path = self.output_dir / f"recreated_{beat_num:03d}.png"
        print("  → Compositing character...")
        
        composite_action = f"{self.character_profile.get_full_description()}\n\nAction: {next_action}"
        
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
        """Run feedback loop with reality checking."""
        self.current_media = initial_media
        self.history = []
        
        for beat in range(max_beats):
            print(f"\n{'='*60}\nBEAT {beat + 1}/{max_beats}\n{'='*60}")
            
            # Step 1: Check visibility
            print(f"\n👁️ Checking visibility...")
            visible, reason = self.is_character_adequately_visible(self.current_media)
            
            if not visible:
                print(f"⚠️ Character not visible ({reason}) - recreating...")
                next_action = f"{self.character_profile.visual_id} appears in the scene."
                self.current_media = self.recreate_frame(self.current_media, next_action)
                self.history.append(next_action)
                continue
            
            # Step 2: Get previous action (what we intended)
            if self.history:
                intended_action = self.history[-1]
            else:
                intended_action = "Initial scene setup"
            
            # Step 3: Analyze reality (what we actually got)
            print(f"\n🔍 Analyzing reality...")
            actual_reality = self.analyze_reality(self.current_media, intended_action)
            print(f"Reality: {actual_reality[:100]}...")
            
            # Step 4: Compare and decide
            print(f"\n🤔 Comparing intention vs reality...")
            decision = self.compare_and_decide(intended_action, actual_reality)
            match, issues, next_action = self.parse_decision(decision)
            
            print(f"Match: {match}")
            print(f"Issues: {issues}")
            print(f"Next: {next_action}")
            
            # Step 5: If issues detected, recreate frame
            if "NO" in match or "drift" in issues.lower() or "wrong" in issues.lower():
                print(f"\n⚠️ Issues detected - recreating frame...")
                self.current_media = self.recreate_frame(self.current_media, next_action)
            
            # Step 6: Generate video
            output_path = self.output_dir / f"beat_{beat+1:03d}.mp4"
            i2v_prompt = f"{self.character_profile.get_full_description()}\n\nScene: {actual_reality}\n\nNext action: {next_action}"
            
            print(f"\n🎬 Generating video...")
            GenerateVideo(
                prompt=i2v_prompt, 
                media=self.current_media, 
                output=str(output_path),
                duration_sec=10 if WGP or LTX else 5, 
                seed=self.seed + beat
            )
            
            # Update state
            self.current_media = str(output_path)
            self.history.append(next_action)
            
            print(f"\n✅ Beat {beat + 1} complete")
        
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