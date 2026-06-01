from PIL import Image, PngImagePlugin
from image_edit import EditImage
from image_gen import GenerateImage, add_metadata_char, add_metadata_loc
from image_analysis import AnalyzeImage
import os

from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

def CompositeScene(
    background_path: str,
    characters: list[str],
    shot_type: str = "medium",
    action: str = "hair swaying gently",
    output: str = "composite.png",
    seed: int = -1,
    width: int = WIDTH,
    height: int = HEIGHT
):
    # 1. Validate
    if not os.path.exists(background_path): raise FileNotFoundError(f"Background not found: {background_path}")

    # 2. Extract metadata (source of truth)
    img = Image.open(background_path)
    desc = img.info.get("Description")

    if desc is None:
        desc = add_metadata_loc(background_path, '', seed)

    bg_desc = desc


    # Establishing shot mode (no characters)
    if len(characters) == 0:
        task = (
            f"REF 1: {bg_desc}. "
            f"{shot_type.upper()} SHOT of the environment. "
            f"Camera focus instruction: {action}. "
            "No characters, no silhouettes, no human forms. "
            "Preserve exact rendering style of REF 1. "
            "ALLOW CROPPING of background elements naturally."
        )

        print(f'\n📝 PROMPT (establishing shot):\n{task}\n')

        status = EditImage(task, [background_path], output, width, height, seed)

        img = Image.open(output)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Prompt", task)
        meta.add_text("ShotType", "establishing")
        img.save(output, pnginfo=meta)

        status.update({"action": action, "prompt": task})
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        status['prompt'] = task
        return status

    for c in characters:
        if not os.path.exists(c): 
            print(f"Character not found: {c}")
            raise FileNotFoundError(f"Character not found: {c}")
    
    # Build character descriptions
    descriptions = []
    for c in characters:
        desc = Image.open(c).info.get('Description')
        if not desc:
            desc = add_metadata_char(c, '', seed)
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
        "closeup": (
            "EXTREME FACE CLOSE-UP. Face fills 80% of frame. "
            "Crop at chin. No shoulders. Camera distance: very close."
        ),
        "medium": (
            "WAIST-UP FRAMING. Camera distance: tight medium. "
            "Subject scale: large. Face occupies upper third of frame. "
            "Anchor face at vertical 0.32. Centered horizontally."
        ),
        "two_shot": (
            "Tight waist-up framing of two characters shoulder-to-shoulder. "
            "Camera distance: medium close. Subject scale: large. "
        ),
        "wide": (
            "Full body shot. Character small in frame."
        ),
        "profile_left": (
            "STRICT PROFILE FACING LEFT. Character on RIGHT side of frame. "
            "Camera distance: medium close. Waist-up only."
        ),
        "profile_right": (
            "STRICT PROFILE FACING RIGHT. Character on LEFT side of frame. "
            "Camera distance: medium close. Waist-up only."
        ),
        "ots": "over-the-shoulder"
    }.get(shot_type, "WAIST-UP FRAMING. Camera distance: tight medium.")


    # 4. Route to specific prompt logic
    # In the prompt construction section, replace the "REF 1" handling:

    # 🆕 SPATIAL INTEGRITY RULES (prevents clipping through objects)
    spatial_rules = (
        "SPATIAL RULES: "
        "1. Characters MUST maintain clean spatial boundaries—NO clipping through furniture, walls, tables, or objects. "
        "2. Characters must be properly grounded on floor surfaces with visible foot/leg contact. "
        "3. If furniture (tables, desks, counters) is present, characters are either CLEARLY IN FRONT OF it (occluding it) or CLEARLY BEHIND it (partially occluded by it)—NEVER merged through. "
        "4. Maintain consistent depth layering: foreground elements > characters > midground objects > background. "
        "5. NO floating, NO intersecting geometry, NO transparency through solid objects."
    )

    Lighting = '''Soft cinematic key light from camera-left, gentle fill from camera-right, consistent color temperature across all shots. Maintain identical lighting direction and intensity between OTS and medium shots.'''


    if shot_type == 'ots':
        task = (
            f"REF 1: {bg_desc}. Background source. "
            # 🆕 Add crop permission
            "ALLOW CROPPING: Background elements may be partially cropped or extend off-frame to maintain composition. DO NOT force-fit entire objects. "
            + spatial_rules +
            "Cinematic close-up, camera is eye level, over-the-shoulder shot of "
            f"REF 2: Character 1 (foreground character) {descriptions[0]} blurred, face is away from the camera and "
            "focusing on "
            f"REF 3: Character 2 (background character) {descriptions[1]}, clear shot, face towards camera, shoulders squared, visible from shoulders up. "
            f"Action: {action}. "
            f"Lighting: {Lighting} Foreground character is blurred and slightly darker. "
            f"Match REF 1 color temperature. Preserve EXACT rendering style from REF 2 and REF 3. "
            f"NO flat lighting, NO foreground sharpness, NO cartoon shading. --no dark faces, no merged depth"
        )
    else:
        task = (
            f"REF 1: {bg_desc}. "
            # 🆕 Add crop permission + priority
            "COMPOSITION RULE: Characters are the focal point. Background elements may be cropped, truncated, or extend beyond frame edges naturally. NEVER shrink background or foreground objects to fit—allow natural cropping instead. "
            + spatial_rules + 
            f"REF 2: {chars_desc} "
            f"Action: {action}. "
            f"Framing: {framing}. "
            f"Lighting: {Lighting} Characters are fully lit and sharp. "
            f"Match lighting, color temperature, and atmosphere of REF 1 exactly. "
            f"Preserve EXACT rendering style, proportions, and details from REF 2. "
            f"NO extras, NO text, NO blur. --no cartoon, no flat colors, no photorealistic skin"
        )

    print(f"\n📝 PROMPT ({len(task.split())} words):\n{task}\n")

    # 5. Generate
    if framing == "wide":
        ref_paths = [background_path] + characters
    else:
        ref_paths = characters
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
            "description": (
                "Composes 0, 1, or 2 characters into a background reference for storyboarding. "
                "When characters are provided, shot_type controls character framing and action "
                "describes anticipation posing. When no characters are provided, shot_type "
                "controls environmental camera distance (e.g., wide, closeup) and action "
                "describes the camera's focus target within the environment (e.g., 'focus on a can on the ground')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "background_path": {
                        "type": "string",
                        "description": (
                            "Path to the reference background image (must contain 'Description' metadata). "
                            "Always required."
                        )
                    },
                    "characters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 2,
                        "description": (
                            "List of 0 to 2 paths to character reference images. "
                            "If 1–2 characters are provided, the shot is character‑focused. "
                            "If 0 characters are provided, the shot becomes an environmental establishing shot "
                            "and shot_type/action are reinterpreted accordingly."
                        )
                    },
                    "shot_type": {
                        "type": "string",
                        "enum": [
                            "medium",
                            "closeup",
                            "profile_left",
                            "profile_right",
                            "ots",
                            "two_shot",
                            "wide"
                        ],
                        "default": "medium",
                        "description": (
                            "Camera framing. "
                            "With characters: defines how characters are framed (e.g., medium, closeup, two_shot). "
                            "Without characters: defines environmental camera distance (e.g., closeup of an object, wide establishing shot)."
                        )
                    },
                    "action": {
                        "type": "string",
                        "description": (
                            "With characters: describes the anticipation pose or micro‑action (Frame 0). "
                            "Without characters: describes the camera's focus target or emphasis within the environment "
                            "(e.g., 'focus on a can on the ground', 'focus on the neon sign')."
                        )
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
                "required": ["background_path", "action"]
            }
        }
    }

def GenerateBackdropSchema():
    return {
        "type": "function",
        "function": {
            "name": "generate_backdrop",  # Matches function name for direct routing
            "description": "Take a master environment image and generate a repositioned viewpoint of a specific zone within the same room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "media": {
                        "type": "string", 
                        "description": "Absolute or relative file path to the source master environment image."
                    },
                    "zone": {
                        "type": "string",
                        "description": "Text description of the specific area to frame within the same room. Example: 'the opposite side of the room near the arched window'"
                    },
                    "output": {"type": "string", "default": "zone_backdrop.png"},
                    "width": {"type": "integer", "description": "Output image width in pixels"},
                    "height": {"type": "integer", "description": "Output image height in pixels"},
                    "seed": {"type": "integer", "description": "Random seed for reproducibility (-1 for random)"},
                    "char_image": {"type": "string", "description": "character to inject into the new backdrop"}
                },
                "required": ["media", "zone"]
            }
        }
    }

def _classify_scene(media: str) -> str:
    """Quick heuristic: check for sky/horizon dominance vs. enclosed geometry."""
    # Option A: Simple color/edge heuristic
    # Option B: Call a tiny VLM classifier: "Is this indoor or outdoor? Reply one word."
    analysis = AnalyzeImage(media, "Is this scene indoor or outdoor? Reply with one word: 'indoor' or 'outdoor'.")
    return analysis['analysis'].strip().lower()

def GenerateZoneBackdrop(
    media: str,
    zone: str,
    output: str = "zone_backdrop.png",
    width: int = 1328,
    height: int = 1328,
    seed: int = -1,
    char_image: str = None,
):
    """Generate a zone backdrop via structured image analysis + targeted prompt construction."""
    
    if not os.path.exists(media):
        raise FileNotFoundError(f"Source not found: {media}")
    
    # ========================================================================
    # 🔍 STAGE 1: ANALYZE SOURCE IMAGE (structured extraction)
    # ========================================================================
    scene_type = _classify_scene(media)

    if scene_type == "outdoor":
        analysis_prompt = (
            "Describe only the permanent environmental foundation of this outdoor scene. Focus on:\n"
            "1. TERRAIN & GROUND: Sand, rock, grass, water, soil texture and base colors.\n"
            "2. VEGETATION & NATURAL FEATURES: Tree types, foliage density, rock formations, dunes, cliffs.\n"
            "3. SKY & ATMOSPHERE: Cloud type, haze, humidity, time-of-day lighting quality. "
            "DO NOT describe sun position, shadow direction, or specular highlight locations.\n"
            "4. OPTICAL CHARACTERISTICS: Horizon line position, atmospheric perspective, lens style.\n"
            "Keep under 80 words. Describe only what is inherent to the location. EXCLUDE transient objects and light-source geometry."
        )
    else:  # indoor fallback
        analysis_prompt = (
            "Describe only the permanent architectural shell and environmental foundation of this image. Focus on:\n"
            "1. STRUCTURAL BOUNDARIES: Walls, ceiling, floor. DO NOT describe doors, windows, arches, or openings.\n"
            "2. SURFACE MATERIALS: Stone, wood, metal, plaster, fabric textures and their base colors.\n"
            "3. AMBIENT LIGHTING QUALITY: Overall color temperature, atmospheric density (haze/dust), soft fill grade.\n"
            "4. OPTICAL CHARACTERISTICS: Depth of field, lens style, aspect ratio, highlight behavior.\n"
            "Keep under 80 words. Describe only what is built into the room itself. EXCLUDE all doors, windows, arches, and focal landmarks."
        )
    
    analysis = AnalyzeImage(media, analysis_prompt)
    analysis_text = analysis['analysis'].strip()
    
    # ========================================================================
    # 🎨 STAGE 2: CONSTRUCT ZONE PROMPT (preserve + swap + remove)
    # ========================================================================
    
    # Character injection (optional, weak influence via prompt only)
    char_desc = ""
    if char_image and os.path.exists(char_image):
        with Image.open(char_image) as img:
            raw_desc = img.info.get('Description', 'character')
            char_desc = (
                f"A single {raw_desc}. "
                "Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour. "
                "Position naturally within the space, matching environmental lighting and perspective. "
            )
    
    if scene_type == "outdoor":
        prompt = (
            f"{analysis_text}\n\n"
            f"Generate a cinematic environment shot of: {zone}.\n"
            # In the outdoor branch of your prompt construction:
            "LIGHTING CONTINUITY (CRITICAL):\n"
            "• Preserve overall lighting QUALITY: color temperature, atmospheric density, time-of-day mood\n"
            "• BUT vary the apparent sun angle slightly (±15°) so reflections, shadow fall, and water speculars feel naturally shifted\n"
            "• If water is present: vary wave-angle reflections; avoid replicating exact highlight patterns from source\n"
            "• Prefer overcast or diffuse lighting if the zone description doesn't specify direct sun\n\n"
            "This is a DISTINCT AREA within the same natural environment as the analyzed scene.\n\n"
            "PRESERVE EXACTLY:\n"
            "• Biome type, terrain composition, vegetation style, color palette\n"
            "• Sky conditions, lighting direction, color temperature, atmospheric haze\n"
            "• Time of day, weather, environmental mood\n"
            #"• Optical traits: Panavision 70mm, cinematic depth of field, horizon placement\n\n"
            "NATURAL LANDMARK RULES:\n"
            "• ONLY include specific natural features (rock formations, tree clusters, water edges) if EXPLICITLY mentioned in the zone description.\n"
            "• If a feature is NOT mentioned, vary it naturally while staying within the same ecosystem.\n"
            "• The new area must feel geographically continuous (same coastline, forest type, desert region)\n\n"
            "COMPOSITION:\n"
            "• NO characters (unless specified), NO text, NO style drift\n"
            f"{char_desc}"
            #"• Photorealistic cinematic environment shot, compositing-ready"
        )
        if "water" in analysis_text.lower() or "ocean" in analysis_text.lower():
            prompt += (
                "\n• Water surface: maintain same wave scale and foam texture, "
                "but vary reflection angles and specular highlight distribution. "
                "No identical sun-path glints."
            )
    else:
        prompt = (
            f"{analysis_text}\n\n"
            f"Generate a cinematic environment shot of: {zone}.\n"
            "This is a DISTINCT AREA within the same overall location as the analyzed scene.\n\n"
            "PRESERVE EXACTLY:\n"
            "• Environment type, architectural style, material textures, color palette\n"
            "• Lighting quality, direction, color temperature, volumetric atmosphere\n"
            "• Time of day, weather conditions, atmospheric mood (haze, dust, smoke)\n"
            #"• Optical traits: Panavision 70mm, cinematic depth of field\n\n"
            "LANDMARK RULES (CRITICAL):\n"
            "• ONLY include doors, windows, arches, or openings if they are EXPLICITLY mentioned in the zone description above.\n"
            "• If a door/window is NOT mentioned in the zone string, it must NOT appear in the output.\n"
            "• The new focal element must feel physically connected to the original space (same building, same era, same design language)\n\n"
            "COMPOSITION:\n"
            "• NO characters (unless specified below), NO text overlay, NO style drift\n"
            f"{char_desc}"
            #"• Photorealistic cinematic environment shot, compositing-ready"
        )

    
    # ========================================================================
    # 🎯 STAGE 3: GENERATE (pure text-to-image, no reference image)
    # ========================================================================
    status = GenerateImage(
        prompt=prompt,
        output=output,
        width=width,
        height=height,
        seed=seed)
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    return status


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-B', '--background', type=str, help='Background path')
    parser.add_argument('-C', '--chars', action='append', default=[], help='Character paths (1-2)')
    parser.add_argument('-S', '--shot-type', type=str, default='medium_single')
    parser.add_argument('-A', '--action', type=str, help='Action to complete')
    parser.add_argument('-R', '--gen-reverse', action='store_true', help='Generate reverse-angle background (T2I)')
    parser.add_argument('-Z', '--zone', type=str, default=None, help='Request different zone other than reverse')
    args = parser.parse_args()
    if args.gen_reverse:
        if not args.background: print("ERROR: -I required for reverse gen"); exit(1)
        if args.zone:
            if args.chars:
                char_image = args.chars.pop()
            else:
                char_image = None
            print(GenerateZoneBackdrop(args.background, args.zone, args.output, 1328, 1328, args.seed, char_image))
        else:
            print(GenerateReverseBackground(args.background, args.output, 1328, 1328, args.seed))
    else:
        print(CompositeScene(args.background, args.chars, args.shot_type, args.action, args.output, args.seed, args.width, args.height))