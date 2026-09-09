from pathlib import Path
import os
from PIL import Image

from plan10.lib.config import load_environ

load_environ()

from plan10.lib.compositor import CompositeScene
from plan10.lib.util import video_to_img


WGP = os.environ.get("WGP","False") != "False"
LTX = os.environ.get("LTX","False") != "False"
MMH3 = os.environ.get("MMH3", "False") != "False"

from plan10.emergent.vision import VisibilityChecker
from plan10.emergent.director import Director
from plan10.lib.image_analysis import AnalyzeMedia

class Pipeline:
    def __init__(self, character_refs, output_dir, width, height, seed, visual_ids, scene_mode=False, goal=None):
        self.character_refs = character_refs
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.seed = seed
        self.visual_ids = visual_ids
        self.scene_mode = scene_mode
        self.goal = goal
        self.initial_media = None
        
    def recreate_frame(self, media_path, bg_path, current_state, beat_num):
        """Recreate frame by analyzing background, generating description, creating fresh background, then compositing."""
        media_path = Path(media_path)
        ext = media_path.suffix.lower()
        
        # Step 1: Extract last frame if video
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            last_frame = video_to_img(str(media_path), self.width, self.height, True, True)
        else:
            last_frame = Image.open(media_path)
        
        last_frame_path = self.output_dir / f"last_frame_{beat_num:03d}.png"
        last_frame.save(str(last_frame_path))
        
        # Step 2: Analyze the background/environment to create a description
        print(f"  → Analyzing background environment...")
        from plan10.lib.image_analysis import AnalyzeImage
        
        bg_analysis_prompt = """Analyze this image and describe ONLY the background/environment in detail. Ignore any people or characters present.

    Provide a comprehensive description of:
    - LOCATION: Type of setting (indoor/outdoor, specific place)
    - ARCHITECTURE: Buildings, structures, room layout, furniture
    - LIGHTING: Time of day, light sources, shadows, mood
    - ATMOSPHERE: Weather, season, time period, overall feel
    - COLORS: Dominant color palette, materials, textures
    - DETAILS: Notable objects, decorations, props, environmental elements

    Focus on creating a description that could be used to recreate this exact environment without any people in it.

    Output format:
    LOCATION: [detailed description]
    ARCHITECTURE: [detailed description]
    LIGHTING: [detailed description]
    ATMOSPHERE: [detailed description]
    COLORS: [detailed description]
    DETAILS: [detailed description]

    COMBINED_DESCRIPTION: [combine all above into one flowing paragraph suitable for image generation]"""
        
        result = AnalyzeImage(str(last_frame_path), bg_analysis_prompt)
        bg_description = result['analysis']
        
        # Extract the combined description
        combined_desc = ""
        for line in bg_description.split('\n'):
            if 'COMBINED_DESCRIPTION:' in line:
                combined_desc = line.split(':', 1)[1].strip()
                break
        
        if not combined_desc:
            combined_desc = bg_description
        
        print(f"  → Background description: {combined_desc[:150]}...")
        
        # Step 3: Generate a fresh, clean background from the description
        clean_bg_path = self.output_dir / f"clean_bg_{beat_num:03d}.png"
        print(f"  → Generating fresh background from description...")
        
        from plan10.lib.image_gen import CreateBackground
        CreateBackground(
            prompt=combined_desc,
            output=str(clean_bg_path),
            seed=self.seed + beat_num + 500,
            override=(768,448)
        )
        
        # Step 4: Composite characters onto the fresh background
        composite_path = self.output_dir / f"recreated_{beat_num:03d}.png"
        print(f"  → Compositing {len(self.character_refs)} character(s) onto fresh background...")
        
        CompositeScene(
            background_path=str(clean_bg_path),
            characters=self.character_refs,
            shot_type="medium" if len(self.character_refs) == 1 else "two_shot",
            action=current_state,
            output=str(composite_path),
            width=self.width,
            height=self.height,
            seed=self.seed + beat_num
        )
        
        return str(composite_path), str(clean_bg_path)

    def recreate_frame_passthrough(self, media_path, bg_path, current_state, beat_num):
        """Scene mode: just return the original image without modification."""
        print(f"  → Scene mode: returning original image")
        return str(media_path), str(bg_path)

    def generate_transition_frame(self, new_location_prompt, beat_num):
        from image_gen import CreateBackground
        
        print(f"  → Generating new background for: {new_location_prompt}")
        bg_path = self.output_dir / f"trans_bg_{beat_num:03d}.png"
        
        CreateBackground(
            prompt=new_location_prompt,
            output=str(bg_path),
            seed=self.seed + beat_num + 1000,
            override=(768,448)
        )
        
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
        
        return str(comp_path), str(bg_path)

    def verify_location_change(self, current_media, expected_new_location):
        """Check if the current frame actually shows the new location."""
        prompt = f"""Does this {"video" if current_media.endswith(('.mp4', '.avi', '.mov')) else "image"} show "{expected_new_location}"?
        
    Answer YES if the background/environment clearly matches the new location.
    Answer NO if it still shows the previous location or is ambiguous.

    Respond with only YES or NO."""
        
        result = AnalyzeMedia(current_media, prompt)['analysis']
        return "YES" in result.upper()

    def execute_step(self, current_media, current_bg, story_context, beat_count, history, pending_setup, needs_transition):
        """Executes one creative step. Returns updated state dict with video job queued."""
        
        print(f"\n{'='*60}\nBEAT {beat_count + 1}\n{'='*60}")

        # FIRST BEAT: Store initial media and animate directly
        if not history:
            self.initial_media = current_media
            print("🎬 First beat - animating initial scene...")
            
            output_path = self.output_dir / f"beat_{beat_count+1:03d}.mp4"
            
            video_prompt = self._format_video_prompt(
                location="",
                characters=f"{' and '.join([x for x in self.visual_ids])} ",
                next_action=story_context,
                camera_framing="maintain current framing, natural movement"
            )
            
            print(f"\n📝 Queuing first video generation...")
            print(f"Prompt preview: {video_prompt[:200]}...")
            
            video_job = {
                "beat": beat_count + 1,
                "prompt": video_prompt,
                "input_media": current_media,
                "output_path": str(output_path),
                "seed": self.seed + beat_count,
                "status": "pending"
            }
            
            new_history = [story_context]
            setup = f"Initial scene: {story_context}"
            
            print(f"\n✅ Beat {beat_count + 1} planned. Video queued for rendering.")
            
            return {
                "beat_count": beat_count + 1,
                "current_media": current_media,
                "history": new_history,
                "pending_setup": setup,
                "needs_transition": False,
                "video_job": video_job,
                "current_bg": current_bg,
            }

        # SUBSEQUENT BEATS
        if self.scene_mode:
            recreate = self.recreate_frame_passthrough
        else:
            recreate = self.recreate_frame

        # 1. Get previous intention
        intended_action = history[-1]

        # 2. Analyze reality FIRST
        direct = Director()
        print(f"\n🔍 Analyzing reality...")
        raw_reality = direct.analyze_reality(current_media, intended_action, self.width, self.height, self.output_dir)
        actual_reality = direct._clean_analysis(raw_reality)
        
        # 3. Compare and decide (director makes decision including scene transitions)
        print(f"\n🤔 Comparing intention vs reality...")
        
        if self.scene_mode:
            location_constraint = "Character must remain in the current room/location. All actions must be physically possible within this space. No transitions or location changes."
        else:
            location_constraint = None
        
        decision = direct.compare_and_decide(
            intended_action, actual_reality, story_context, 
            history, pending_setup, goal=self.goal, 
            force_transition=needs_transition,
            location_constraint=location_constraint
        )
        
        # 4. Parse decision with NEW 10-value signature
        match, issues, location, characters, next_action, camera_framing, setup, goal_progress, scene_transition, new_location = direct.parse_decision(decision)

        print(f"Goal Progress: {goal_progress}")
        print(f"Match: {match}, Issues: {issues}")
        print(f"Location: {location}")
        print(f"Scene Transition: {scene_transition}, New Location: {new_location}")
        print(f"Characters: {characters}")
        print(f"Next Action: {next_action}")
        print(f"Camera Framing: {camera_framing}")
        print(f"Setup for next beat: {setup}")
        
        # 5. Check if director planned a scene transition
        if not self.scene_mode and scene_transition == "YES" and new_location:
            print(f"\n🎬 Director planned transition to: {new_location}")
            # Verify it actually happened
            location_changed = self.verify_location_change(current_media, new_location)
            
            if not location_changed:
                print(f"⚠️ Transition didn't happen. Forcing transition...")
                current_media, current_bg = self.generate_transition_frame(new_location, beat_count)
                needs_transition = False
                history.append(new_location)
            else:
                print(f"✓ Scene transition confirmed")
                history.append(new_location)
        
        # 6. Otherwise, check visibility (fallback)
        elif not self.scene_mode:
            visible = True
            reason_code = ''
            
            for visual_id in self.visual_ids:
                vcheck = VisibilityChecker(visual_id, self.width, self.height)
                visability, reason_code, reason_text = vcheck.check(current_media, self.output_dir)
                visible &= visability
                if not visible:
                    break

            if not visible:
                print(f"⚠️ Character not visible ({reason_code}): {reason_text}")
                
                if reason_code == "empty_background":
                    print("  → Background is empty/black. Generating new background...")
                    location_hint = history[-1] if history else "detailed environment, realistic lighting"
                    current_media, current_bg = self.generate_transition_frame(location_hint, beat_count)
                    needs_transition = False
                    
                elif reason_code == "walking_away":
                    print("  → Character is leaving the scene. Forcing cinematic CUT TO new location/angle.")
                    needs_transition = True
                    
                elif not MMH3 and reason_code == "turned_away":
                    print("  → Character is turned away. Recreating frame to face camera (same location)...")
                    current_state = f"{' and '.join([x for x in self.visual_ids])} turns around to face the camera in a frontal or 3/4 view, maintaining the exact same environment."
                    current_media, current_bg = recreate(current_media, current_bg, current_state, beat_count)
                    needs_transition = False

                elif reason_code == "wrong_character":
                    print("  → Character identity issue (props/weapons drifted). Accepting for now...")
                    # Don't recreate - just continue and hope it stabilizes
                    needs_transition = False
                    
                else:
                    print("  → Unintended loss of visibility. Recreating frame...")
                    current_state = f"{' and '.join([x for x in self.visual_ids])} is now visible in the scene, facing the camera in a frontal or 3/4 view."
                    current_media, current_bg = recreate(current_media, current_bg, current_state, beat_count)
                    needs_transition = False

        else:
            # scene_mode
            current_media, current_bg = recreate(current_media, current_bg, "", beat_count)

        # 7. Handle major issues (skip in scene_mode)
        if not self.scene_mode:
            if "NO" in match or "drift" in issues.lower() or "repeating" in issues.lower():
                if not needs_transition:
                    print(f"\n⚠️ Major issues detected - rebuilding frame to current state...")
                    current_media, current_bg = recreate(current_media, current_bg, actual_reality, beat_count)
        
        # 8. Handle cinematic transition (skip in scene_mode)
        if not self.scene_mode:
            combined_text = f"{next_action} {camera_framing}".upper()
            if needs_transition and "CUT TO" in combined_text:
                print(f"\n🎬 Executing Cinematic Transition to: {location}")
                current_media, current_bg = self.generate_transition_frame(location, beat_count)
                needs_transition = False
        
        # 9. Format video prompt and queue it
        output_path = self.output_dir / f"beat_{beat_count+1:03d}.mp4"
        video_prompt = self._format_video_prompt(location, characters, next_action, camera_framing)
        
        print(f"\n📝 Queuing video generation...")
        print(f"Prompt preview: {video_prompt[:200]}...")
        
        input_media = current_media

        video_job = {
            "beat": beat_count + 1,
            "prompt": video_prompt,
            "input_media": input_media,
            "output_path": str(output_path),
            "seed": self.seed + beat_count,
            "status": "pending"
        }
        
        # 10. Update history
        new_history = history + [next_action]
        
        print(f"\n✅ Beat {beat_count + 1} planned. Video queued for rendering.")
        
        return {
            "beat_count": beat_count + 1,
            "current_media": current_media,
            "current_bg": current_bg,
            "history": new_history,
            "pending_setup": setup,
            "needs_transition": needs_transition,
            "video_job": video_job
        }

    def _format_video_prompt(self, location, characters, next_action, camera_framing):
        if not characters:
            characters = f"{' and '.join([x for x in self.visual_ids])} " 
        
        if not location:
            location = "current location"
        
        return f"""{location}

{characters}

Action: {next_action}
Camera: {camera_framing}
"""