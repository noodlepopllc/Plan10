
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
    from anime_gen import GenerateImage, CreateBackground
else:
    from image_gen import GenerateImage, CreateBackground

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
                    # Only add if we have at least visual_id
                    if current_char.get('visual_id'):
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
                current_char[current_field] += ' ' + line
        
        # Don't forget the last character
        if current_char and current_char.get('visual_id'):
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
    Visual ID: {char.get('visual_id', 'unknown')}
    Appearance: {char.get('appearance', 'not specified')}
    Clothing: {char.get('clothing', 'not specified')}"""
        
        # Return all prominent characters
        descriptions = []
        for i, char in enumerate(self.characters):
            descriptions.append(f"""CHARACTER {i+1}:
    Visual ID: {char.get('visual_id', 'unknown')}
    Appearance: {char.get('appearance', 'not specified')}
    Clothing: {char.get('clothing', 'not specified')}""")
        
        return '\n\n'.join(descriptions)


class FeedbackLoop:
    def __init__(self, character_refs, output_dir="feedback_output", width=WIDTH, height=HEIGHT, seed=SEED):
        # Accept list of character references
        if isinstance(character_refs, str):
            character_refs = [character_refs]
        
        self.character_refs = character_refs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.width = width
        self.height = height
        self.seed = seed
        
        # Build profiles for ALL character references
        print("\n🔍 Building character profiles from references...")
        self.character_profiles = []
        for i, ref in enumerate(character_refs):
            print(f"  → Analyzing character {i+1}: {ref}")
            profile = CharacterProfile(ref)
            self.character_profiles.append(profile)
        
        # Build combined description for all characters
        self.character_desc = self._build_combined_description()
        
        # Get first character's visual_id for primary visibility tracking
        if self.character_profiles and self.character_profiles[0].get_character_count() > 0:
            self.visual_id = self.character_profiles[0].get_character(0)['visual_id']
        else:
            self.visual_id = "unknown character"
        
        print(f"✓ Built profiles for {len(self.character_profiles)} character(s)")
        print(f"Primary visual ID: {self.visual_id}")
        
        self.history = []
        self.current_media = None

    def generate_transition_frame(self, new_location_prompt, beat_num):
        """Generate a clean background for a new location and composite characters onto it."""
        print(f"  → Generating new background for: {new_location_prompt}")
        
        # Step 1: Generate clean background for the new location
        bg_path = self.output_dir / f"trans_bg_{beat_num:03d}.png"
        
        CreateBackground(
            prompt=new_location_prompt,
            output=str(bg_path),
            seed=self.seed + beat_num + 1000
        )
        
        # Step 2: Composite ALL characters onto this new background
        comp_path = self.output_dir / f"trans_comp_{beat_num:03d}.png"
        print(f"  → Compositing {len(self.character_refs)} character(s) into new location...")
        
        CompositeScene(
            background_path=str(bg_path),
            characters=self.character_refs,
            shot_type="medium" if len(self.character_refs) == 1 else "two_shot",
            action=f"Characters positioned in {new_location_prompt}. Clear frontal or 3/4 view, faces fully recognizable, ready for action.",
            output=str(comp_path),
            width=self.width,
            height=self.height,
            seed=self.seed + beat_num + 2000
        )
        
        return str(comp_path)
    
    def _build_combined_description(self):
        """Build a combined description for all characters."""
        descriptions = []
        for i, profile in enumerate(self.character_profiles):
            char_desc = profile.get_full_description()
            if char_desc:
                descriptions.append(f"CHARACTER {i+1}:\n{char_desc}")
        
        return '\n\n'.join(descriptions) if descriptions else "No character profiles available"

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
        
        # Step 2: Composite ALL characters back onto clean background
        composite_path = self.output_dir / f"recreated_{beat_num:03d}.png"
        print(f"  → Compositing {len(self.character_refs)} character(s) onto clean background...")
        
        # ENFORCE SAFE ANGLE: frontal or 3/4 view
        composite_action = f"{current_state}"
        
        CompositeScene(
            background_path=str(clean_bg_path),
            characters=self.character_refs,  # ← Pass ALL character references
            shot_type="medium" if len(self.character_refs) == 1 else "two_shot",
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


    def compare_and_decide(self, intended_action, actual_reality, story_context, force_transition=False):
        """Generate next action with renderer-ready components and cinematic rules."""
        history_text = "\n".join([f"- {a}" for a in self.history[-3:]]) if self.history else "First beat."
        
        setup_context = ""
        if hasattr(self, 'pending_setup') and self.pending_setup:
            setup_context = f"\nPREVIOUS SETUP: {self.pending_setup}\nThis was set up in the last beat and should now pay off or escalate."
        
        # Force transition if character walked away
        transition_directive = ""
        if force_transition:
            transition_directive = """
CRITICAL: The character has walked away or turned their back. You MUST generate a "CUT TO:" that transitions to a NEW LOCATION or NEW CAMERA ANGLE where the character is clearly visible from the front or 3/4 view. Do NOT continue the current shot."""
        
        prompt = f"""STORY CONTEXT: {story_context}

    PREVIOUS INTENTION: {intended_action}
    ACTUAL SCENE STATE: {actual_reality}
    RECENT ACTIONS: {history_text}{setup_context}{transition_directive}

    TASK: Apply "Yes, And..." improv logic with CAUSAL ESCALATION.
    1. YES: Accept the ACTUAL SCENE STATE as absolute truth.
    2. AND: Generate the next physical action that ADVANCES the story context.
    3. CRITICAL: This action must SET UP the next beat.

    Output format (STRICTLY follow this, no extra text):
    MATCH: [YES/PARTIAL/NO]
    ISSUES: [none, or specific problem]
    LOCATION: [brief location]
    CHARACTERS: [brief descriptions]
    NEXT_ACTION: [1-2 sentences of pure story action]
    CAMERA_FRAMING: [1 sentence of strict visual direction: lens, angle, lighting, movement]
    SETUP: [what this sets up]"""
        
        result = llm_analyze_media(
            media="", prompt=prompt,
            system="You are a film director and screenwriter. Every action must setup the next beat through causal chains. Use cinematic cuts to solve visibility issues.",
            max_tokens=250, temperature=0.7
        )['analysis']
        
        return result.strip()

    def parse_decision(self, decision_text):
        """Parse the decision output into components."""
        lines = decision_text.split('\n')
        match = "UNKNOWN"
        issues = "none"
        location = ""
        characters = ""
        next_action = ""
        camera_framing = "static shot, medium framing"
        setup = ""
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("MATCH:"):
                match = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()
            elif line.upper().startswith("LOCATION:"):
                location = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CHARACTERS:"):
                characters = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NEXT_ACTION:"):
                next_action = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CAMERA_FRAMING:"):
                camera_framing = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SETUP:"):
                setup = line.split(":", 1)[1].strip()
        
        return match, issues, location, characters, next_action, camera_framing, setup

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
        """Two-tier visibility check with anime-friendly leniency and detailed reasoning."""
        img, check_path = self._extract_frame_for_check(media_path)
        
        # Tier 1: RetinaFace (fast) - is ANY face visible?
        detector = RetinaFace()
        faces = detector.detect(img)
        del detector
        cleanup()
        
        if not faces:
            return False, "no_face", "No faces detected in frame"
        
        # Tier 2: AnalyzeImage identity check with detailed reasoning
        prompt = f"""Looking for a character matching this description: {self.visual_id}

    Analyze the image and answer:

    1. Is there a character visible that generally matches this description? (YES/NO)
    2. What is their orientation relative to the camera?
    - "facing_camera" - front or 3/4 view, face clearly visible
    - "walking_away" - showing back, moving away from camera
    - "turned_away" - side view or back, not moving
    - "partially_visible" - partially obscured or at extreme angle
    3. Brief reason for your assessment (1 sentence)

    Output format (STRICTLY follow this):
    MATCH: [YES/NO]
    ORIENTATION: [facing_camera/walking_away/turned_away/partially_visible]
    REASON: [brief explanation]"""
        
        result = AnalyzeImage(check_path, prompt)
        response = result['analysis'].strip()
        
        # Parse structured response
        match = "NO"
        orientation = "unknown"
        reason = "Unknown"
        
        for line in response.split('\n'):
            line = line.strip()
            if line.upper().startswith("MATCH:"):
                match = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("ORIENTATION:"):
                orientation = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        
        if match == "YES" and orientation == "facing_camera":
            return True, "visible", reason
        elif orientation == "walking_away":
            return False, "walking_away", reason
        elif orientation == "turned_away":
            return False, "turned_away", reason
        elif orientation == "partially_visible":
            return False, "partially_visible", reason
        elif match == "NO":
            return False, "wrong_character", reason
        else:
            return False, "unknown", reason
    
    def run(self, initial_media, story_context, max_beats=8):
        # Check if initial media needs characters composited
        print("\n🔍 Checking initial media...")
        visible, reason_code, reason_text = self.is_character_adequately_visible(initial_media)
        
        while not visible:
            print(f"⚠️ Initial media has no visible characters ({reason})")
            print(f"  → Compositing {len(self.character_refs)} character(s) into scene...")
            
            # Composite all characters into the initial media
            initial_media = self.recreate_frame(initial_media, story_context)
            print(f"  ✓ Characters composited into initial scene")
            visible, reason_code, reason_text = self.is_character_adequately_visible(initial_media)
        
        self.current_media = initial_media
        self.history = []
        self.pending_setup = None
        beat_count = 0
        needs_transition = False # Track if we need to build a new scene
        
        while beat_count < max_beats:
            print(f"\n{'='*60}\nBEAT {beat_count + 1}/{max_beats}\n{'='*60}")
            
            # Check visibility
            visible, reason_code, reason_text = self.is_character_adequately_visible(self.current_media)
            
            if not visible:
                print(f"⚠️ Character not visible ({reason_code}): {reason_text}")
                
                if reason_code in ["walking_away", "turned_away"]:
                    print("  → Recognized intentional exit/turn. Will force cinematic CUT TO in next beat.")
                    needs_transition = True
                else:
                    print("  → Unintended loss of visibility. Recreating frame...")
                    current_state = f"{self.visual_id} is now visible in the scene, facing the camera in a frontal or 3/4 view."
                    self.current_media = self.recreate_frame(self.current_media, current_state)
                    needs_transition = False
            
            # Get previous intention
            intended_action = self.history[-1] if self.history else "Initial scene setup"
            
            # Analyze reality
            print(f"\n🔍 Analyzing reality...")
            raw_reality = self.analyze_reality(self.current_media, intended_action)
            actual_reality = self._extract_scene_description(raw_reality)
            
            # Compare and decide
            print(f"\n🤔 Comparing intention vs reality...")
            decision = self.compare_and_decide(intended_action, actual_reality, story_context, force_transition=needs_transition)
            match, issues, location, characters, next_action, camera_framing, setup = self.parse_decision(decision)

            print(f"Match: {match}, Issues: {issues}")
            print(f"Location: {location}")
            print(f"Characters: {characters}")
            print(f"Next Action: {next_action}")
            print(f"Camera Framing: {camera_framing}")
            print(f"Setup for next beat: {setup}")
            
            self.pending_setup = setup
            
            # If major issues (and NOT a planned transition), rebuild frame
            if "NO" in match or "drift" in issues.lower() or "repeating" in issues.lower():
                if not needs_transition: # Don't rebuild if we are already transitioning
                    print(f"\n⚠️ Major issues detected - rebuilding frame to current state...")
                    self.current_media = self.recreate_frame(self.current_media, actual_reality)
            
            # *** CRITICAL: Handle the Cinematic Cut ***
            if needs_transition and "CUT TO" in next_action.upper():
                print(f"\n🎬 Executing Cinematic Transition to: {location}")
                self.current_media = self.generate_transition_frame(location, beat_count)
                needs_transition = False # Reset flag, transition is complete
            
            # Generate video
            output_path = self.output_dir / f"beat_{beat_count+1:03d}.mp4"
            i2v_prompt = self._format_video_prompt(location, characters, next_action, camera_framing)
            
            print(f"\n🎬 Generating video...")
            print(f"Prompt preview: {i2v_prompt[:200]}...")
            
            GenerateVideo(
                prompt=i2v_prompt, 
                media=self.current_media, # This is now the NEW composited frame if a transition occurred!
                output=str(output_path),
                duration_sec=10 if WGP or LTX else 5, 
                seed=self.seed + beat_count
            )
            
            self.current_media = str(output_path)
            self.history.append(next_action)
            beat_count += 1
            
            print(f"\n✅ Beat {beat_count} complete")
        
        return self.history

    def _extract_scene_description(self, raw_analysis):
        """Extract clean scene description from analysis output."""
        # Remove analysis artifacts
        lines = raw_analysis.split('\n')
        clean_lines = []
        
        skip_keywords = ['analysis', 'discrepancies', 'issues', 'unexpected', 'summary', 'based on image']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            clean_lines.append(line)
        
        return ' '.join(clean_lines)

    def _format_video_prompt(self, location, characters, next_action, camera_framing):
        """Format video prompt in renderer-optimized structure: location → characters → action."""
        
        # If no characters provided by improv, fall back to brief visual IDs
        if not characters:
            brief_chars = []
            for i, profile in enumerate(self.character_profiles):
                if profile.get_character_count() > 0:
                    char = profile.get_character(0)
                    brief_chars.append(char.get('visual_id', f'character {i+1}'))
            characters = ", ".join(brief_chars) if brief_chars else "a person"
        
        # If no location provided, use generic
        if not location:
            location = "indoor location"
        
        return f"""{location}

    {characters}

    Action: {next_action}
    Camera: {camera_framing}
    """

if __name__ == "__main__":
    import argparse
    from decomposer import decompose_scene
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--ref', type=str, action='append', default=[], help="Character reference image (can specify multiple)")
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
    refs = args.ref
    if not initial:
        if not args.prompt:
            print("Error: --initial or --prompt required")
            sys.exit(1)
        GenerateImage(prompt=args.prompt, output=f'{args.output}/improv.png', width=args.width, height=args.height, seed=args.seed)
        initial = f'{args.output}/improv.png'
        decompose_scene(
            input_image=initial,
            output_dir=args.output,
            width=args.width,
            height=args.height,
            seed=args.seed
        )
    if not refs:
        for p in ['character_1.png', 'character_2.png']:
            if Path(f'{args.output}/{p}').exists():
                refs.append(f'{args.output}/{p}')
    print(f"REFERENCES: {refs}")
        # Use provided refs, or fall back to initial image
    
    loop = FeedbackLoop(
        character_refs=refs,  # ← Pass list of refs
        output_dir=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    history = loop.run(initial, args.context, args.beats)
    
    print("\nAction history:")
    for i, action in enumerate(history, 1):
        print(f"{i}. {action}")