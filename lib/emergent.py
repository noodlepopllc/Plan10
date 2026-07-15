
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
    - Ethnicity
    - Age range
    - Hair color and style
    - Main clothing color/item (TOP HALF only)

    Output ONLY the description (5-10 words), nothing else."""
        return AnalyzeImage(self.ref_path, prompt)['analysis'].strip()

    def _extract_appearance(self):
        """Physical appearance details."""
        prompt = """Describe this character's physical appearance:
    - Ethnicity (be specific: East Asian, Southeast Asian, Caucasian, African, Hispanic, South Asian, Middle Eastern, etc.)
    - Age range
    - Gender
    - Face shape (oval, round, square, heart, oblong, etc.)
    - Eye shape (almond, round, hooded, monolid, downturned, upturned, etc.)
    - Hair color and style
    - Body type/build
    - Any distinctive features

    Be specific. This will be used to maintain character consistency."""
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
    def __init__(self, character_ref, output_dir="feedback_output", width=WIDTH, height=HEIGHT, seed=SEED):
        self.character_ref = character_ref
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.width = width
        self.height = height
        self.seed = seed
        
        # Extract character profile ONCE
        print("\n🔍 Building character profile from reference...")
        self.character_profile = CharacterProfile(character_ref)
        
        # Store as simple string - reuse everywhere
        self.character_desc = self.character_profile.get_full_description()
        self.visual_id = self.character_profile.visual_id
        
        print(f"✓ Character ready: {self.visual_id}")
        
        self.history = []
        self.current_media = None

    def recreate_frame(self, media_path, current_state):
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
        
        # ENFORCE SAFE ANGLE: frontal or 3/4 view
        composite_action = f"Character in clear frontal or 3/4 view, face fully recognizable. {self.character_desc}\n\nCurrent state: {current_state}"
        
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

    def analyze_reality(self, media_path, intended_action):
        """Analyze what we ACTUALLY created vs what we intended."""
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        # Extract frame if video
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            check_path = self.output_dir / f"reality_{len(self.history):03d}.png"
            video_to_img(str(media_path), self.width, self.height, True, True).save(str(check_path))
            check_path = str(check_path)
        else:
            check_path = str(media_path)
        
        prompt = f"""We intended to create this: "{intended_action}"

    Analyze what ACTUALLY happened in this image:
    1. What is the character doing? (actions, expressions)
    2. What props/objects are visible?
    3. Any issues or unexpected elements?

    Be factual about what you see, not what was intended."""
        
        result = AnalyzeImage(check_path, prompt)
        return result['analysis'].strip()

    def compare_and_decide(self, intended_action, actual_reality, story_context):
        """Compare prompt vs reality, and generate the 'Yes, And...' next step."""
        history_text = "\n".join([f"- {a}" for a in self.history[-3:]]) if self.history else "First beat."
        
        prompt = f"""STORY CONTEXT: {story_context}

    PREVIOUS INTENTION: {intended_action}
    ACTUAL SCENE STATE: {actual_reality}
    CHARACTER PROFILE: {self.character_desc}
    RECENT HISTORY: {history_text}

    TASK: Apply "Yes, And..." improv logic.
    1. YES: Accept the ACTUAL SCENE STATE as the new, absolute canonical truth of the scene.
    2. AND: Naturally build upon this new reality with the next logical physical action or transition, keeping the STORY CONTEXT in mind.

    Output format:
    MATCH: [YES/PARTIAL/NO - did it fundamentally break the scene?]
    ISSUES: [List any critical problems like "character vanished" or "none"]
    NEXT: [The "And" - what happens next, 1-2 sentences, building directly on the ACTUAL SCENE STATE while advancing the STORY CONTEXT]"""
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are an improv scene director. Accept reality, build on it, and advance the story context.",
            max_tokens=200, temperature=0.7
        )['analysis']
        
        return result.strip()

    def parse_decision(self, decision_text):
        """Parse the decision output into components."""
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

    def is_character_adequately_visible(self, media_path):
        """Two-tier visibility check:
        - Tier 1: RetinaFace (fast) - is ANY face visible and facing camera?
        - Tier 2: AnalyzeImage (slower) - is it the RIGHT character?
        
        Returns: (visible: bool, reason: str)
        """
        img, check_path = self._extract_frame_for_check(media_path)
        
        # Tier 1: RetinaFace
        detector = RetinaFace()
        faces = detector.detect(img)
        del detector
        cleanup()
        
        if not faces:
            return False, "no_face"
        
        # Tier 2: AnalyzeImage identity check
        prompt = f"""Looking for: {self.visual_id}

    Is this specific person clearly visible in the image?
    Answer YES or NO."""
        
        result = AnalyzeImage(check_path, prompt)
        response = result['analysis'].strip().upper()
        
        if "YES" in response:
            return True, "visible"
        else:
            return False, "wrong_character"
    
    def run(self, initial_media, story_context, max_beats=8):
        self.current_media = initial_media
        self.history = []
        beat_count = 0
        
        while beat_count < max_beats:
            print(f"\n{'='*60}\nBEAT {beat_count + 1}/{max_beats}\n{'='*60}")
            
            # Check visibility
            visible, reason = self.is_character_adequately_visible(self.current_media)
            
            if not visible:
                print(f"⚠️ Character not visible ({reason}) - recreating...")
                # Pass current state, not next action
                current_state = f"{self.visual_id} is now visible in the scene, facing the camera in a frontal or 3/4 view."
                self.current_media = self.recreate_frame(self.current_media, current_state)
                # Don't increment beat_count - we'll generate a video this iteration
                # Don't continue - fall through to generate video
            
            # Get previous intention
            if self.history:
                intended_action = self.history[-1]
            else:
                intended_action = "Initial scene setup"
            
            # Analyze what ACTUALLY happened
            print(f"\n🔍 Analyzing reality...")
            actual_reality = self.analyze_reality(self.current_media, intended_action)
            
            # Compare and decide
            print(f"\n🤔 Comparing intention vs reality...")
            # PASS STORY CONTEXT HERE
            decision = self.compare_and_decide(intended_action, actual_reality, story_context)
            match, issues, next_action = self.parse_decision(decision)
            
            print(f"Match: {match}, Issues: {issues}")
            
            # If major issues, rebuild frame showing CURRENT state
            if "NO" in match or "drift" in issues.lower():
                print(f"\n⚠️ Major issues detected - rebuilding frame to current state...")
                self.current_media = self.recreate_frame(self.current_media, actual_reality)
            
            # Generate video
            output_path = self.output_dir / f"beat_{beat_count+1:03d}.mp4"
            i2v_prompt = f"{self.character_desc}\n\nCurrent scene: {actual_reality}\n\nNext action: {next_action}"
            
            print(f"\n🎬 Generating video...")
            GenerateVideo(
                prompt=i2v_prompt, 
                media=self.current_media, 
                output=str(output_path),
                duration_sec=10 if WGP or LTX else 5, 
                seed=self.seed + beat_count
            )
            
            self.current_media = str(output_path)
            self.history.append(next_action)
            beat_count += 1
            
            print(f"\n✅ Beat {beat_count} complete")
        
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