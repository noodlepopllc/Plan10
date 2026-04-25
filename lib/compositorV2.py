from PIL import Image, PngImagePlugin
from image_edit import EditImage
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
    return status

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