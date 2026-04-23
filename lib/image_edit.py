from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import torch
import os, gc
from typing import Union
from image_analysis import AnalyzeImage, EnhancePrompt
from image_gen import GenerateImage
from config import load_environ
import numpy as np
from pathlib import Path

load_environ()

# ─────────────────────────────────────────────────────────────
# EXPRESSION MAPPING
# ─────────────────────────────────────────────────────────────
_EXPR_MAP = {
    "neutral": "relaxed features, natural resting expression",
    "smile": "gentle closed-mouth smile",
    "smiling": "soft smile, slight crinkle at eyes",
    "laughing": "open mouth laugh, natural eye squint",
    "frown": "downturned mouth, relaxed brow",
    "angry": "furrowed brow, tense jaw, narrowed eyes",
    "worried": "slight frown, raised inner eyebrows, tense lips",
    "surprised": "raised eyebrows, slightly parted lips, widened eyes",
    "sad": "downturned corners, heavy eyelids, subtle frown",
    "determined": "focused gaze, set jaw, relaxed but alert posture",
    "smirk": "asymmetrical raised eyebrow, slight one-sided smile",
    "exhausted": "heavy eyelids, relaxed facial muscles, slight slump",
}
def _normalize_expr(expr: str) -> str:
    return _EXPR_MAP.get(expr.strip().lower(), expr)

# ─────────────────────────────────────────────────────────────
# IMAGE EDIT PIPELINE
# ─────────────────────────────────────────────────────────────
class ImageEdit(object):
    def __init__(self, vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        vram_config = {
            "offload_dtype": "disk", "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn, "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn, "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16, "computation_device": "cuda",
        }
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
                ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
                ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
            vram_limit=vrlimit,
        )

    def generate(self, prompt, images, output, width, height, seed):
        # Safely handle empty/character-only lists
        edit_image = [Image.open(x) for x in images] if images else []
        if seed == -1: seed = random.randint(0, 1000000)

        image = self.pipe(
            prompt, edit_image=edit_image, seed=seed, num_inference_steps=4,
            height=height, width=width, cfg_scale=1.0,
        )
        image.save(output)
        os.utime(output, None) 
        status = {"status": "success", "output_path": output, "prompt": prompt, "description": ''}
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        return status

    def __del__(self):
        del self.pipe 
        gc.collect()
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
def EditImageSchema():
    return  {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit or composite up to 3 existing images into a single result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "A list of 1 - 3 images to be edited or combined in different ways, remove a person, add a person, add two people to a location, change poses, actions, etc"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Detailed composition instructions. Refer to the images from the 'images' array by their order: 'first image' (index 0), 'second image' (index 1), 'third image' (index 2)."
                    },
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "seed": {"type": "integer"}
                },
                "required": ["images", "prompt"]
            }
        }
    }

def EditImage(prompt='', images=[''], output='tmp_edit.png', width=1328, height=1328, seed=42):
    edit = ImageEdit()
    status = edit.generate(prompt, images, output, int(width), int(height), int(seed))
    del edit
    return status

def CompositeSceneSchema():
    return {
        "type": "function",
        "function": {
            "name": "composite_scene",
            "description": "Compose 1 or 2 characters into a scene. Lighting is inherited from background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "background_path": {"type": "string"},
                    "characters": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                    "shot_type": {"type": "string", "enum": ["wide_single", "medium_single", "closeup_single", "profile_single_left", "profile_single_right", "two_shot_wide", "two_shot_medium", "two_shot_close", "over_shoulder", "over_shoulder_closeup", "split_closeup"], "default": "medium_single"},
                    "gaze": {"type": "string", "enum": ["forward", "at_each_other", "a_to_b", "b_to_a", "off_camera"], "default": "forward"},
                    "poses": {"type": "array", "items": {"type": "string"}, "description": "Body pose for each character. Max 2."},
                    "expressions": {"type": "array", "items": {"type": "string"}, "description": "Expression for each character. Max 2."},
                    # RENAMED & CLARIFIED
                    "interaction": {"type": "string", "description": "Relational energy & compositional proximity (e.g., 'intimate', 'distant', 'casual', 'tense'). Adjusts character spacing & framing tension. Does NOT affect lighting."},
                    "width": {"type": "integer", "default": 1024},
                    "height": {"type": "integer", "default": 1024},
                    "output": {"type": "string", "default": "composite.png"},
                    "seed": {"type": "integer", "default": -1},
                    "character_lighting": {
                        "type": "string",
                        "description": "Override default background-inherited lighting with subject-prioritized descriptors. Format: comma-separated lighting keywords. Use when characters appear underexposed, flat, or lack edge separation. Examples: 'bright subject-keyed rim light, balanced warm ambient fill, high-key facial exposure' | 'soft beauty dish key, subtle eye catchlight, porcelain-flat skin exposure' | 'dramatic cool rim from camera-left, minimal fill, high-contrast'. Note: closeup shot types auto-append facial boost keywords; omit redundant terms. Does not alter background lighting."
                    }
                },
                "required": ["background_path", "characters"]
            }
        }
    }




import tempfile
from PIL import Image, ImageFilter

def _clean_expr(expr):
    return expr.strip() if expr and expr.strip() else "neutral"

def CompositeScene(background_path: str, characters: list[str], shot_type: str = "medium_single", gaze: str = "forward", poses: Union[str, list[str]] = None, expressions: Union[str, list[str]] = "neutral", interaction: str = "tense", width: int = 1024, height: int = 1024, output: str = "composite.png", seed: int = -1, character_lighting: str = None ):
    width, height, seed = int(width), int(height), int(seed)

    # Closeup expressions: lead with eye descriptors to force tight framing
    _EYE_FIRST_MAP = {
        "angry": "intense eyes, direct gaze, sharp iris detail, furrowed brows",
        "surprised": "wide eyes, focused pupils, high-contrast catchlight, raised brows",
        "determined": "steady gaze, focused eyes, subtle brow tension",
        "worried": "searching eyes, slight brow furrow, soft focus",
        "sad": "heavy eyelids, glistening eyes, downturned gaze",
        "neutral": "soft eye contact, relaxed gaze, natural resting expression",
    }

    def _get_eye_first_expr(emotion: str) -> str:
        return _EYE_FIRST_MAP.get(emotion, _EYE_FIRST_MAP["neutral"])


    if not os.path.exists(background_path): raise FileNotFoundError(f"Background not found: {background_path}")
    if not (1 <= len(characters) <= 2): raise ValueError("characters must be 1 or 2 paths")
    for c in characters:
        if not os.path.exists(c): raise FileNotFoundError(f"Character not found: {c}")

    # ─────────────────────────────────────────────────────────────
    # QUICK ANALYSIS (ALL SHOT TYPES)
    # ─────────────────────────────────────────────────────────────
    img_with_meta = Image.open(background_path)
    if 'Description' in img_with_meta.info:
        bg_desc = img_with_meta.info['Description']
    else:
        bg_analysis = AnalyzeImage(background_path, "Description, Style, lighting, weather in <15 words.")
        bg_desc = bg_analysis['analysis'].strip()

    if isinstance(expressions, str): expr_list = [expressions] * len(characters)
    else: expr_list = list(expressions) + ["neutral"] * (len(characters) - len(expressions))

    if poses is None or poses == "": 
        pose_list = ["natural standing"] * len(characters)
    elif isinstance(poses, str):
        pose_list = [poses] * len(characters)
    else:
        # Pad with default if fewer poses than characters provided
        pose_list = list(poses) + ["natural standing"] * (len(characters) - len(poses))

    # NORMALIZE INTERACTION
    interaction_block = ""
    if interaction.strip():
        interaction_block = f"Interaction/Composition: {interaction.strip()}. Adjust character proximity, spacing, and framing tension accordingly. "
    
    # Default subject-prioritized lighting template
    DEFAULT_CHARACTER_LIGHTING = (
        "bright subject-keyed rim light, "
        "balanced warm ambient fill, "
        "high-key facial exposure, "
        "even global illumination, "
        "no harsh spot-only contrast"
    )
    
    # Closeup-specific facial lighting boost
    CLOSEUP_FACIAL_BOOST = (
        "soft beauty dish key light on face, "
        "subtle catchlight in eyes, "
        "porcelain-flat skin exposure locked, "
        "zero gradient skin rendering"
    )
    # Resolve final lighting string
    if character_lighting:
        base_lighting = character_lighting
    else:
        base_lighting = DEFAULT_CHARACTER_LIGHTING
    
    # Add closeup boost if needed
    is_closeup = any(kw in shot_type.lower() for kw in ["closeup", "split", "over_shoulder"])
    if is_closeup:
        final_lighting = f"{base_lighting}, {CLOSEUP_FACIAL_BOOST}"
    else:
        final_lighting = base_lighting
    
    # Inject into lighting instruction
    lighting_instruction = (
        f"STRICT: Match lighting, color temperature, contact shadows, and atmosphere exactly to REFERENCE IMAGE 1. "
        f"Character lighting: {final_lighting}."
    )


    is_ots = shot_type.lower() in ["over_shoulder", "over_shoulder_closeup"]
    is_ots_tight = shot_type.lower() == "over_shoulder_closeup"  
    is_profile = shot_type.lower().startswith("profile_single")

    # ─────────────────────────────────────────────────────────────
    # CHARACTER DESCRIPTORS (VISUAL, NOT ABSTRACT)
    # ─────────────────────────────────────────────────────────────
    char_descriptors = []
    identity_keywords = []
    
    for char_path in characters:
        img_with_meta = Image.open(char_path)
        if 'Description' in img_with_meta.info:
            clean_string = img_with_meta.info['Description']
        else:
            combined_prompt = (
                '''
                Describe ONLY clearly visible traits. Return a single comma-separated string in this exact order: 
                age, ethnicity, skin tone, face shape, jawline, cheekbones, eyes, eyebrows, nose, lips, 
                hair length/color/texture, hair style, hairline, eyewear, clothing.
                Rules:
                - Be accurate. Do NOT guess. If a trait isn't obvious, write 'neutral'.
                - Age: child, youth, young adult, adult, elderly, neutral
                - Ethnicity: east asian, south asian, middle eastern, african, european, latinx, neutral
                - Skin tone: fair, light, medium, tan, deep, neutral
                - Face shape: oval, round, heart, square, long, neutral
                - Jawline: soft, defined, sharp, angular, neutral
                - Cheekbones: low, medium, high, neutral
                - Eyes: almond, round, narrow, wide-set, neutral
                - Eyebrows: straight, arched, thick, thin, neutral
                - Nose: small, medium, large, narrow, wide, neutral
                - Lips: thin, medium, full, neutral
                - Hair length/color/texture: short/medium/long + color + straight/wavy/curly, or 'neutral'
                - Hair style: ponytail, bun, braid, tied-back, loose, half-up, bob, pixie, or 'neutral'
                - Hairline: straight, widow's peak, rounded, neutral
                - Eyewear: 'preserve glasses' if clearly wearing glasses, otherwise 'none'
                - Clothing: yellow sundress, white tshirt, blue jeans, red sneakers, etc.

                Example:
                "adult, european, light, oval, defined jawline, high cheekbones, almond eyes, arched brows, 
                medium nose, full lips, long brown wavy hair, low ponytail, straight hairline, none, navy uniform"

                Respond ONLY with the string.
                ''')

            analysis = AnalyzeImage(char_path, combined_prompt)
            raw = analysis['analysis'].strip().strip('"').strip("'")
            
            # Clean & filter without regex
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            # Remove "none"/"no glasses" so diffusion doesn't accidentally render them
            cleaned = [p for p in parts if p.lower() not in ["none", "no glasses"]]
            clean_string = ", ".join(cleaned)
        identity_keywords.append(clean_string + " Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour. ")

    # ─────────────────────────────────────────────────────────────
    # REFERENCE IMAGE PREP (OTS: BLUR FOREGROUND)
    # ─────────────────────────────────────────────────────────────
    ref_paths = []
    ref_paths.append(background_path)
    ref_paths += characters

    # ─────────────────────────────────────────────────────────────
    # PROMPT BUILDING (EDIT-STYLE, <100 WORDS)
    # ─────────────────────────────────────────────────────────────
    framing_map = {
        "wide_single": "Full body, character small.",
        "medium_single": "Waist-up, centered.",
        
        "profile_single_left": "Body turned 90° to the left, head facing left, eyes looking left. Standing on the right third of the frame with empty space on the left.",
        "profile_single_right": "Body turned 90° to the right, head facing right, eyes looking right. Standing on the left third of the frame with empty space on the right.",


        "two_shot_wide": "Both full body, balanced.",
        "two_shot_medium": "Both waist-up, side-by-side.",
        "two_shot_close": "Both chest-up, intimate.",
        
        "closeup_single": "Tight headshot.",  # Unused, but keep for consistency
        "over_shoulder": "Foreground in left third, background in center. Clear spatial division.",
        "over_shoulder_closeup": "Tight over-the-shoulder. Left foreground anchor, right face dominates. Crop at collarbone.",

        "split_closeup": "Both faces side-by-side, equal framing."
    }
    framing = framing_map.get(shot_type, framing_map["medium_single"])

    char1_desc = identity_keywords[0]
    if len(characters) > 1:
        char2_desc = identity_keywords[1]
    else:
        char2_desc = ''

    ref1 = "Person in REFERENCE IMAGE 2"
    ref2 = "Person in REFERENCE IMAGE 3"
    
    # 🔑 GAZE WITH VISUAL DESCRIPTORS
    if len(characters) == 1:
        if is_profile:
            gaze_str = "Looking naturally off-frame."
        else:
            gaze_str = {"forward": "Looking forward.", "off_camera": "Looking off-frame."}.get(gaze, "Looking forward.")
    else:
        gaze_map = {
            "forward": "Both looking forward.",
            "at_each_other": f"{ref1} and {ref2} making eye contact.",
            "a_to_b": f"{ref1} looks at {ref2}.",
            "b_to_a": f"{ref2} looks at {ref1}.",
            "off_camera": "Both looking off-frame."
        }
        gaze_str = gaze_map.get(gaze, "Both looking forward.")

    char1_pose = pose_list[0]
    char2_pose = pose_list[1] if len(characters) > 1 else None

    face_expr = _get_eye_first_expr(expr_list[0])

    if is_ots and len(characters) == 2:
        task = (
            f"REF 1: {bg_desc}. Background source. "
            f"REF 2: Character 1 (foreground character) {identity_keywords[0]}, back to camera, back of head and shoulder visible. "
            f"REF 3: Character 2 (background character) {identity_keywords[1]}, face towards camera, shoulders squared, visible from shoulders up. "
            f"Over-the-shoulder of Character 1 facing Character 2. {framing}. "
            "Each character has completely different clothing, no matching colors or styles. "
            f"Expression: {_clean_expr(expr_list[1])}. {lighting_instruction} NO extras."
            " 8K, Photorealistic, Realistic Skin and Textures with pores"
        )

    elif is_closeup:
        
        task = (
            f"REF 1: {bg_desc}, FULL-FRAME BACKGROUND. "
            f"REF 2: {identity_keywords[0]}, EXTREME FACE CLOSE-UP ONLY. "
            f"Crop just below chin. Face fills 95% of frame. Zero shoulders. "
            f"Expression: {face_expr}. {lighting_instruction} NO extras."
        )

    else:
        # Standard shots can afford slightly more detail
        p1 = f"REF 2: {identity_keywords[0]}. {char1_pose}. {expr_list[0]}. "
        if len(characters) > 1:
            p2 = f"REF 3: {identity_keywords[1]}. {char2_pose}. {_clean_expr(expr_list[1])}. "
            people_desc = f"{p1}{p2}"
        else:
            people_desc = p1

        task = (
            f"REFERENCE IMAGE 1: {bg_desc}. {people_desc}"
            f"Integrate into background. {framing}. {gaze_str}. {lighting_instruction} NO extras."
        )
        # Word count: ~55-65
    task += " Maintain the exact facial structure, eye shape, jawline, and hair geometry from the reference image. "
    task += " 8K, Photorealistic, Realistic Skin and Textures with pores"

    # Debug log
    print("\n" + "="*60)
    print(f"📝 PROMPT ({len(task.split())} words):")
    print(task)
    print(f"📎 ref_paths: {ref_paths}")
    print(f"🎲 seed: {seed}")
    print("="*60 + "\n")

    #Path(output.replace('.png','.txt')).write_text(task)
    status = EditImage(task, ref_paths, output, width, height, seed)

    target_image = Image.open(output)
    metadata = PngInfo()
    metadata.add_text("Prompt", task)
    metadata.add_text("Seed", str(seed))
    target_image.save(output, pnginfo=metadata)

    status['prompt'] = task
    return status

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cinematic Image Pipeline')
    parser.add_argument('-I', '--images', action='append', default=[], help='Input images')
    parser.add_argument('-P', '--prompt', type=str, default='remove text', help='Edit prompt')
    parser.add_argument('-W', '--width', type=int, default=1024)
    parser.add_argument('-H', '--height', type=int, default=1024)
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-C', '--compose', action='store_true', help='Enable composite mode')
    parser.add_argument('-BG', '--background', type=str, help='Background path')
    parser.add_argument('-CHARS', '--chars', action='append', default=[], help='Character paths (1-2)')
    parser.add_argument('-SHOT', '--shot_type', type=str, default='medium_single')
    parser.add_argument('-Z', '--pose', type=str, default='')
    parser.add_argument('-GAZE', '--gaze', type=str, default='forward')
    parser.add_argument('-EXPR', '--expressions', action='append', default=[])
    parser.add_argument('-T', '--interaction', type=str, default='intimate')
    args = parser.parse_args()

    if args.compose:
        if not args.background or not args.chars: print("ERROR: -BG and -CHARS required"); exit(1)
        print(CompositeScene(args.background, args.chars, args.shot_type, args.gaze, args.pose, args.expressions, args.interaction, args.width, args.height, args.output))
    else:
        # Standard Edit Image Fallback
        print(EditImage(args.prompt, args.images, args.output, args.width, args.height, args.seed))
