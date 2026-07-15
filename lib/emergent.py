
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
    """Cached character profile - extracts ONLY prominent foreground characters."""
    
    def __init__(self, character_ref_path):
        self.ref_path = character_ref_path
        self.characters = self._extract_all_characters()
    
    def _extract_all_characters(self):
        """Extract complete profiles for ONLY prominent foreground characters."""
        prompt = """Analyze this image and extract a complete profile for the 1 to 3 MOST PROMINENT FOREGROUND characters ONLY.

CRITICAL RULES:
1. IGNORE background people, crowds, blurry figures, or distant subjects.
2. Focus ONLY on characters who are large, in focus, and clearly the main subjects of the image.
3. Maximum of 3 characters. If there is only 1 prominent person, output only CHARACTER_1.

For EACH prominent character, provide:
1. VISUAL_ID: 5-10 word description (ethnicity, age range, hair color/style, main top clothing)
2. APPEARANCE: Physical details (ethnicity, age, gender, face shape, eye shape, hair, body type, distinctive features)
3. CLOTHING: Detailed clothing (top color/style/fit, bottom color/style/fit, shoes, accessories, hair details)

Output format (use this EXACT structure):
CHARACTER_1:
VISUAL_ID: [description]
APPEARANCE: [description]
CLOTHING: [description]

CHARACTER_2: (ONLY if a second prominent foreground character exists)
VISUAL_ID: [description]
APPEARANCE: [description]
CLOTHING: [description]

CHARACTER_3: (ONLY if a third prominent foreground character exists)
VISUAL_ID: [description]
APPEARANCE: [description]
CLOTHING: [description]

Be extremely specific about colors, styles, and physical features. This will be used to maintain consistency."""
        
        result = AnalyzeImage(self.ref_path, prompt)['analysis'].strip()
        return self._parse_character_data(result)
    
    def _parse_character_data(self, text):
        """Parse the structured output into a list of character dicts."""
        characters = []
        current_char = {}
        current_field = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('CHARACTER_'):
                if current_char:
                    characters.append(current_char)
                current_char = {}
                current_field = None
            elif line.startswith('VISUAL_ID:'):
                current_field = 'visual_id'
                current_char['visual_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('APPEARANCE:'):
                current_field = 'appearance'
                current_char['appearance'] = line.split(':', 1)[1].strip()
            elif line.startswith('CLOTHING:'):
                current_field = 'clothing'
                current_char['clothing'] = line.split(':', 1)[1].strip()
            elif current_field and current_char:
                # Continuation line for multi-sentence descriptions
                current_char[current_field] += ' ' + line
        
        # Don't forget the last character
        if current_char:
            characters.append(current_char)
        
        return characters
    
    def get_character_count(self):
        """Return number of prominent characters detected."""
        return len(self.characters)
    
    def get_character(self, index):
        """Get a specific character by index (0-based)."""
        if 0 <= index < len(self.characters):
            return self.characters[index]
        return None
    
    def get_all_visual_ids(self):
        """Get list of all visual IDs."""
        return [char['visual_id'] for char in self.characters]
    
    def get_full_description(self, character_index=None):
        """Get formatted description for one or all prominent characters."""
        if character_index is not None:
            char = self.get_character(character_index)
            if not char:
                return ""
            return f"""CHARACTER PROFILE:
Visual ID: {char['visual_id']}
Appearance: {char['appearance']}
Clothing: {char['clothing']}"""
        
        # Return all prominent characters
        descriptions = []
        for i, char in enumerate(self.characters):
            descriptions.append(f"""CHARACTER {i+1}:
Visual ID: {char['visual_id']}
Appearance: {char['appearance']}
Clothing: {char['clothing']}""")
        
        return '\n\n'.join(descriptions)


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