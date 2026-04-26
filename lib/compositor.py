from PIL import Image, PngImagePlugin
from image_edit import EditImage
from image_analysis import AnalyzeImage
import os

def CompositeScene(
    background_path: str,
    characters: list[str],
    shot_type: str = "medium",
    action: str = "hair swaying gently",
    output: str = "composite.png",
    seed: int = -1,
    width: int = 832,
    height: int = 480
):
    # 1. Validate
    if not os.path.exists(background_path): raise FileNotFoundError(f"Background not found: {background_path}")
    for c in characters:
        if not os.path.exists(c): raise FileNotFoundError(f"Character not found: {c}")

    # 2. Extract metadata (source of truth)
    bg_desc = Image.open(background_path).info.get('Description', 'cinematic environment')
    
    # Build character descriptions
    descriptions = []
    for c in characters:
        desc = Image.open(c).info.get('Description', 'character')
        descriptions.append(f"{desc}. Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour.")

    if shot_type not in ("two_shot", "ots"):
        descriptions = [descriptions.pop(0)] 
    
    if len(descriptions) > 1:
        chars_desc = f"Character 1: {descriptions[0]}. Character 2: {descriptions[1]}. "
    else:
        chars_desc = f"Character 1: {descriptions[0]}. " 

    # 3. Explicit Framing Dictionary
    # Note: Profile uses "Position on X looking Y" to force direction
    framing = {
        "closeup": "EXTREME FACE CLOSE-UP. Face fills frame. Crop below chin. Zero shoulders. High focus on facial expression details and eyes.",
        "profile_left": "STRICT PROFILE FACING LEFT. Character positioned on RIGHT side of frame looking off-screen LEFT. Large empty negative space on left.",
        "profile_right": "STRICT PROFILE FACING RIGHT. Character positioned on LEFT side of frame looking off-screen RIGHT. Large empty negative space on right.",
        "ots": "over-the-shoulder",
        "wide": "Full body shot, character small in frame.",
        "two_shot": "Tight waist-up framing. Characters standing shoulder-to-shoulder with minimal gap. Neutral stance.",
        "medium": "Waist-up, centered."
    }.get(shot_type, "medium shot")

    # 4. Route to specific prompt logic
    if shot_type == 'ots':
        # OTS requires specific depth cues and REF separation
        # FIX: Removed "8K Photorealistic" to prevent cartoon/style clash
        task = (
            f"REF 1: {bg_desc}. Background source. "
            "Cinematic close-up, camera is eye level, over-the-shoulder shot of "
            f"REF 2: Character 1 (foreground character) {descriptions[0]} blurred, face is away from the camera and "
            "focusing on "
            f"REF 3: Character 2 (background character) {descriptions[1]}, clear shot, face towards camera, shoulders squared, visible from shoulders up. "
            f"Action: {action}. "
            f"Lighting: Bright cinematic key + rim light on REF 3. REF 2 stays darker/blurred. "
            f"Match REF 1 color temperature. Preserve EXACT rendering style from REF 2 and REF 3. "
            f"NO flat lighting, NO foreground sharpness, NO cartoon shading. --no dark faces, no merged depth"
        )
    else:
        # Standard logic for other shots
        task = (
            f"REF 1: {bg_desc}. "
            f"REF 2: {chars_desc} "
            f"Action: {action}. "
            f"Framing: {framing}. "
            f"Lighting: CHARACTERS ARE BRIGHTLY LIT AND SEPARATED FROM BACKGROUND. "
            f"Match lighting, color temperature, and atmosphere of REF 1 exactly. "
            f"Preserve EXACT rendering style, proportions, and details from REF 2. "
            f"NO extras, NO text, NO blur. --no cartoon, no flat colors, no photorealistic skin"
        )

    print(f"\n📝 PROMPT ({len(task.split())} words):\n{task}\n")

    # 5. Generate
    ref_paths = [background_path] + characters
    status = EditImage(task, ref_paths, output, width, height, seed)

    # 6. Embed metadata for I2V handoff
    img = Image.open(output)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Prompt", task)
    meta.add_text("Action", action)
    meta.add_text("ShotType", shot_type)
    img.save(output, pnginfo=meta)

    status.update({"action": action, "prompt": task})
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    status['prompt'] = task
    return status

def CompositeSceneSchema():
    return {
        "type": "function",
        "function": {
            "name": "composite_scene",
            "description": "Composes 1 or 2 characters into a background reference for storyboarding. Automatically handles lighting matching, framing, and action-aware posing (Frame 0 anticipation) for downstream video generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "background_path": {
                        "type": "string",
                        "description": "Path to the reference background image (must contain 'Description' metadata)."
                    },
                    "characters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 2,
                        "description": "List of 1 or 2 paths to character reference images (must contain 'Description' metadata)."
                    },
                    "shot_type": {
                        "type": "string",
                        "enum": ["medium", "closeup", "profile_left", "profile_right", "ots", "two_shot", "wide"],
                        "default": "medium",
                        "description": "Camera framing and composition type."
                    },
                    "action": {
                        "type": "string",
                        "description": "Description of the 'anticipation pose' (Frame 0) for the characters. This defines the initial state for video motion (e.g., 'hair swaying back', 'weight shifted to step')."
                    },
                    "output": {
                        "type": "string",
                        "default": "composite.png",
                        "description": "Output filename."
                    },
                    "seed": {
                        "type": "integer",
                        "default": -1,
                        "description": "Random seed for reproducibility. -1 for random."
                    },
                    "width": {
                        "type": "integer",
                        "default": 832,
                        "description": "Output image width."
                    },
                    "height": {
                        "type": "integer",
                        "default": 480,
                        "description": "Output image height."
                    }
                },
                "required": ["background_path", "characters", "action"]
            }
        }
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cinematic Image Pipeline')
    parser.add_argument('-W', '--width', type=int, default=832)
    parser.add_argument('-H', '--height', type=int, default=480)
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-B', '--background', type=str, help='Background path')
    parser.add_argument('-C', '--chars', action='append', default=[], help='Character paths (1-2)')
    parser.add_argument('-S', '--shot-type', type=str, default='medium_single')
    parser.add_argument('-A', '--action', type=str, help='Action to complete')
    args = parser.parse_args()
    CompositeScene(args.background, args.chars, args.shot_type, args.action, args.output, args.seed, args.width, args.height)