from PIL import Image, PngImagePlugin
from image_edit import EditImage
from image_analysis import AnalyzeImage
import os

from config import load_environ

if os.environ.get('ANIME','False') != 'False':
    from anime_gen import GenerateImage, add_metadata_char, add_metadata_loc
else:
    from image_gen import GenerateImage, add_metadata_char, add_metadata_loc

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

from PIL import Image

def CompositeBackground(
    background_path: str,
    shot_type: str = "middle",  # "middle", "left", "right"
    output: str = "composite.png",
    seed: int = -1,
    width: int = WIDTH,   # Target video width (e.g., 1280 or 720)
    height: int = HEIGHT  # Target video height (e.g., 720 or 1280)
):
    """Generate left/right/middle backdrop from wide background image at target resolution."""
    
    # 1. Validate
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background not found: {background_path}")

    # 2. Load wide image (1664x928 base)
    wide = Image.open(background_path)
    wide_width, wide_height = wide.size
    
    # 3. Crop based on shot_type
    if shot_type == "left":
        # Left half
        cropped = wide.crop((0, 0, wide_width // 2, wide_height))
        crop_desc = "LEFT SIDE of environment"
    elif shot_type == "right":
        # Right half
        cropped = wide.crop((wide_width // 2, 0, wide_width, wide_height))
        crop_desc = "RIGHT SIDE of environment"
    else:  # middle
        cropped = wide
        crop_desc = "FULL WIDE SHOT of environment"
    
    # 4. Save cropped version temporarily
    crop_path = output.replace('.png', '_crop.png')
    #cropped.save(crop_path)
    cropped.save(output)
    
    '''
    # 5. Analyze the cropped version
    analysis = AnalyzeImage(crop_path, "Describe this environment in detail, focusing on architectural elements, lighting, materials, and spatial layout. 100-150 words.")
    bg_desc = analysis['analysis']
    
    # 6. Build prompt for regeneration at TARGET resolution
    # CRITICAL: Prevent the model from outpainting or hallucinating elements from the rest of the room
    task = (
        f"REF 1: {bg_desc}. "
        f"{crop_desc}. "
        "CRITICAL INSTRUCTION: This is a tightly cropped view of a specific section. "
        "DO NOT outpaint, DO NOT expand the frame, and DO NOT hallucinate or add elements from the rest of the room/location that are not visible in REF 1. "
        "Strictly limit the generated content to only what is explicitly visible in the reference image. "
        "ALLOW CROPPING of background elements naturally at frame edges. "
        "No characters, no silhouettes, no human forms. "
        "Preserve exact rendering style, lighting, and atmosphere of REF 1."
    )
    
    print(f'\n📝 PROMPT ({shot_type} shot, target {width}x{height}):\n{task}\n')
    
    # 7. Regenerate at TARGET resolution using cropped version as reference
    status = EditImage(task, [crop_path], output, width, height, seed)
    '''

    # 8. Clean up temporary crop
    if os.path.exists(crop_path):
        os.remove(crop_path)

    desc = add_metadata_loc(output, '', seed)

    status = {"status": "success", "output_path": output, "prompt": f"crop {shot_type}", "description": desc}
    status.update({"shot_type": shot_type, "resolution": f"{cropped.width}x{cropped.height}"})
    if os.environ.get('BATCH', 'False') == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    return status


def CompositeScene(
    background_path: str,
    characters: list[str],
    shot_type: str = "medium",
    action: str = "hair swaying gently",
    output: str = "composite.png",
    seed: int = -1,
    width: int = WIDTH,
    height: int = HEIGHT,
):
    style = "anime" if os.environ.get('ANIME', 'False') != 'False' else "realistic"
    # 1. Validate
    if not os.path.exists(background_path): 
        raise FileNotFoundError(f"Background not found: {background_path}")

    # 2. Extract metadata (source of truth)
    img = Image.open(background_path)

    if shot_type in ['closeup','medium','ots']:
        desc = img.info.get("Brief")
    else:
        desc = img.info.get("Description")

    if desc is None:
        if shot_type in ['closeup','medium','ots']:
            desc = add_metadata_loc(background_path, '', seed, True)
        else:
            desc = add_metadata_loc(background_path, '', seed)

    bg_desc = desc

    # 🆕 STYLE-SPECIFIC CONFIGURATION
    if style == "anime":
        lighting_desc = (
            "Anime cel-shaded lighting: soft key light from camera-left with clean shadow edges, "
            "gentle fill from camera-right, consistent color temperature. "
            "Maintain identical lighting direction and cel-shading between shots. "
            "Flat color fills with minimal gradients, bold outlines preserved."
        )
        char_preserve = (
            "Preserve anime proportions: large expressive eyes, minimal nose (dot or line), "
            "V-line face shape, cel-shaded skin with cel shadows, clean lineart, "
            "stylized hair with gravity-defying volume. "
            "Maintain exact anime art style from reference images."
        )
        analysis_style = "anime illustration"
    else:
        lighting_desc = (
            "Soft cinematic key light from camera-left, gentle fill from camera-right, "
            "consistent color temperature across all shots. "
            "Maintain identical lighting direction and intensity between OTS and medium shots. "
            "Natural skin texture with subsurface scattering, realistic shadows."
        )
        char_preserve = (
            "Preserve adult facial proportions, light cheekbone definition, "
            "subtle jawline contour, realistic skin texture with pores and natural imperfections, "
            "photorealistic rendering. Maintain exact realistic style from reference images."
        )
        analysis_style = "photorealistic image"

    # Establishing shot mode (no characters)
    if len(characters) == 0:
        task = (
            f"REF 1: {bg_desc}. "
            f"{shot_type.upper()} SHOT of the environment. "
            f"Camera focus instruction: {action}. "
            "No characters, no silhouettes, no human forms. "
            f"Preserve exact rendering style of REF 1 ({style} style). "
            "ALLOW CROPPING of background elements naturally."
        )

        print(f'\n📝 PROMPT (establishing shot):\n{task}\n')

        status = EditImage(task, [background_path], output, width, height, seed)

        img = Image.open(output)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Prompt", task)
        meta.add_text("ShotType", "establishing")
        meta.add_text("Style", style)
        img.save(output, pnginfo=meta)

        status.update({"action": action, "prompt": task})
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(
                output, 
                f"Briefly describe this {analysis_style}, no more than 100 words"
            )
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
        if not os.path.exists(c): 
            raise FileNotFoundError(f"Character not found: {c}")
    
    # Build character descriptions with explicit identity markers
    descriptions = []
    for i, c in enumerate(characters):
        desc = Image.open(c).info.get('Description')
        if not desc:
            desc = add_metadata_char(c, '', seed)
        #descriptions.append(f"Character reference sheet showing one character from multiple angles. Render only ONE instance of this character in the scene. {desc}. {char_preserve}")
        descriptions.append(f"{desc}. {char_preserve}")

    if shot_type not in ("two_shot", "ots"):
        descriptions = [descriptions.pop(0)] 
    
    spatial_rules = (
        "SPATIAL RULES: "
        "1. Characters MUST maintain clean spatial boundaries—NO clipping through furniture, walls, tables, or objects. "
        "2. Characters must be properly grounded on floor surfaces with visible foot/leg contact. "
        "3. If furniture (tables, desks, counters) is present, characters are either CLEARLY IN FRONT OF it (occluding it) or CLEARLY BEHIND it (partially occluded by it)—NEVER merged through. "
        "4. Maintain consistent depth layering: foreground elements > characters > midground objects > background. "
    )

    if shot_type == 'ots' and len(descriptions) > 1:
        task = (
            f"REF 1: {bg_desc}. Background source. "
            "ALLOW CROPPING: Background elements may be partially cropped or extend off-frame to maintain composition. DO NOT force-fit entire objects. "
            + spatial_rules +
            f"Cinematic close-up, camera is eye level, over-the-shoulder shot of "
            f"REF 2: Character 1 (foreground character) {descriptions[0]} blurred, face is away from the camera and "
            "focusing on "
            f"REF 3: Character 2 (background character) {descriptions[1]}, clear shot, face towards camera, shoulders squared, visible from shoulders up. "
            f"Action: {action}. "
            f"Lighting: {lighting_desc} Foreground character is blurred and slightly darker. "
            f"Match REF 1 color temperature. Preserve EXACT rendering style from REF 2 and REF 3. "
        )
    elif shot_type == 'two_shot' and len(descriptions) > 1:
        task = (            
            f"REF 1: {bg_desc}. "
            "COMPOSITION RULE: Characters are the focal point. Background elements may be cropped naturally. "
            #+ spatial_rules +
            
            "TWO-SHOT COMPOSITION: "
            f"REF 2: {descriptions[0]} "
            "POSITION: LEFT SIDE OF FRAME ONLY. "
            
            f"REF 3: {descriptions[1]} "
            "POSITION: RIGHT SIDE OF FRAME ONLY. "
            
            
            f"Action: {action}. "
            f"Framing: Tight waist-up framing of two DISTINCT characters. Camera distance: medium close. "
            f"Lighting: {lighting_desc} Both characters are fully lit and sharp. "
            f"Match lighting, color temperature, and atmosphere of REF 1 exactly. "
            f"Preserve EXACT rendering style from REF 2 and REF 3. "
        )
    else:
        # Single character shots (same as before)
        chars_desc = f"Character 1: {descriptions[0]}. "
        framing = {
            "closeup": "EXTREME FACE CLOSE-UP. Face fills 80% of frame. Crop at chin. Camera distance: very close.",
            "medium": "WAIST-UP FRAMING. Camera distance: tight medium. Subject scale: large. Face occupies upper third of frame. Anchor face at vertical 0.32. Centered horizontally.",
            "wide": "Full body shot. Character small in frame.",
            "profile_left": "STRICT PROFILE FACING LEFT. Character on RIGHT side of frame. Camera distance: medium close. Waist-up only.",
            "profile_right": "STRICT PROFILE FACING RIGHT. Character on LEFT side of frame. Camera distance: medium close. Waist-up only.",
        }.get(shot_type, "WAIST-UP FRAMING. Camera distance: tight medium.")
        
        task = (
            f"REF 1: {bg_desc}. "
            "COMPOSITION RULE: Characters are the focal point. Background elements may be cropped, truncated, or extend beyond frame edges naturally. NEVER shrink background or foreground objects to fit—allow natural cropping instead. "
            + spatial_rules + 
            f"REF 2: {chars_desc} "
            f"Action: {action}. "
            f"Framing: {framing}. "
            f"Lighting: {lighting_desc} Character is fully lit and sharp. "
            f"Match lighting, color temperature, and atmosphere of REF 1 exactly. "
            f"Preserve EXACT rendering style, proportions, and details from REF 2. "
        )


    print(f"\n📝 PROMPT ({len(task.split())} words):\n{task}\n")

    # 5. Generate
    # 🆕 For two-shot, pass background + all character images as references
    if shot_type in ["wide", "two_shot"]:
        ref_paths = [background_path] + characters
    else:
        ref_paths = [background_path] + characters
    
    status = EditImage(task, ref_paths, output, width, height, seed)

    # 6. Embed metadata for I2V handoff
    img = Image.open(output)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Prompt", task)
    meta.add_text("Action", action)
    meta.add_text("ShotType", shot_type)
    meta.add_text("Style", style)  # 🆕 Track style in metadata
    img.save(output, pnginfo=meta)

    status.update({"action": action, "prompt": task})
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(
            output, 
            f"Briefly describe this {analysis_style}, no more than 100 words"
        )
        status['description'] = analysis['analysis']
    status['prompt'] = task
    return status

def CompositeBackgroundSchema():
    return {
        "type": "function",
        "function": {
            "name": "composite_background",
            "description": (
                "Generates cropped background variants from a wide reference image at target resolution. "
                "Takes a wide background (e.g., 1664x928) and creates left/right/wide versions for different "
                "camera positions within a zone. The cropped version is used as a reference to regenerate "
                "the background at the target video resolution, maintaining visual consistency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "background_path": {
                        "type": "string",
                        "description": (
                            "Path to the wide reference background image (must contain 'Description' metadata). "
                            "This is the master wide shot of the zone that will be cropped."
                        )
                    },
                    "shot_type": {
                        "type": "string",
                        "enum": ["wide", "left", "right"],
                        "default": "wide",
                        "description": (
                            "Which portion of the wide background to use. "
                            "'wide' = full image (for Two-Shot). "
                            "'left' = left half (for character positioned on left side of zone). "
                            "'right' = right half (for character positioned on right side of zone)."
                        )
                    },
                    "output": {
                        "type": "string",
                        "default": "composite_bg.png",
                        "description": "Output filename for the generated background."
                    },
                    "seed": {
                        "type": "integer",
                        "default": -1,
                        "description": "Random seed for reproducibility. -1 for random."
                    },
                    "width": {
                        "type": "integer",
                        "default": 1280,
                        "description": "Target output width (should match video generation resolution)."
                    },
                    "height": {
                        "type": "integer",
                        "default": 720,
                        "description": "Target output height (should match video generation resolution)."
                    }
                },
                "required": ["background_path"]
            }
        }
    }

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
                        "default": WIDTH,
                        "description": "Output image width."
                    },
                    "height": {
                        "type": "integer",
                        "default": HEIGHT,
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