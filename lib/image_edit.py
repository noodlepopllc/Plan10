from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from PIL import Image
import random
import torch
import os, gc
from image_analysis import AnalyzeImage
from config import load_environ
import numpy as np
from pathlib import Path
from util import video_to_img

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
class ImageEditQwen(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None
    
    def __enter__(self):
        if not self.pipe:
            vram_config = {
                "offload_dtype": "disk", "offload_device": "disk",
                "onload_dtype": torch.float8_e4m3fn, "onload_device": "cpu",
                "preparing_dtype": torch.float8_e4m3fn, "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16, "computation_device": "cuda",
            }
            self.pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16, device="cuda",
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
                vram_limit=self.vrlimit,
            )
            self.pipe.load_lora(self.pipe.dit, "./loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", alpha=1.0)
            self.pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
            return self

    def generate(self, prompt, images, output, width, height, seed):
        if not self.pipe:
            self.__enter__(self)
        # Safely handle empty/character-only lists
        edit_images = []
        for item in images:
            if isinstance(item, Image.Image):
                # Already a PIL image → use directly
                edit_images.append(item)
            elif isinstance(item, str):
                # File path → load it
                edit_images.append(Image.open(item))
            else:
                raise TypeError(f"Unsupported image type: {type(item)}")
        if seed == -1: seed = random.randint(0, 1000000)

        image = self.pipe(
            prompt, edit_image=edit_images, seed=seed, num_inference_steps=8,
            height=height, width=width, edit_image_auto_resize=True,
            zero_cond_t=True, cfg_scale=1.0,
        )
        image.save(output)
        os.utime(output, None) 
        status = {"status": "success", "output_path": output, "prompt": prompt, "description": ''}
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        return status

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda and torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()
    
class ImageEditKlein(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        self.vrlimit = vrlimit
        self.pipe = None

    def __enter__(self):
        if not self.pipe:
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
                vram_limit=self.vrlimit,
            )
        return self

    def generate(self, prompt, images, output, width, height, seed):
        if not self.pipe:
            self.__enter__()

        edit_images = []

        for item in images:
            if isinstance(item, Image.Image):
                # Already a PIL image → use directly
                edit_images.append(item)
            elif isinstance(item, str):
                # File path → load it
                edit_images.append(Image.open(item))
            else:
                raise TypeError(f"Unsupported image type: {type(item)}")
    
        if seed == -1: seed = random.randint(0, 1000000)

        image = self.pipe(
            prompt, edit_image=edit_images, seed=seed, num_inference_steps=4,
            height=height, width=width, cfg_scale=1.0,
        )
        image.save(output)
        os.utime(output, None) 
        status = {"status": "success", "output_path": output, "prompt": prompt, "description": ''}
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        return status

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        gc.collect()
        if torch.cuda and torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
            torch.cuda.empty_cache()

if os.environ.get("IMAGE_EDIT", "KLEIN") == "KLEIN":
    ImageEdit = ImageEditKlein
else:
    ImageEdit = ImageEditQwen

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
                    "output": {"type": "string", "default": "edited.png"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "seed": {"type": "integer"}
                },
                "required": ["images", "prompt"]
            }
        }
    }

def EditImage(prompt='', images=[''], output='tmp_edit.png', width=1328, height=1328, seed=42, img_edit=None):
    if not img_edit:
        edit = ImageEdit()
    else:
        edit = img_edit
    status = edit.generate(prompt, images, output, int(width), int(height), int(seed))
    if not img_edit:
        del edit
    return status

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
                    "image": {"type": "string", "description": "character to inject into the new backdrop"}
                },
                "required": ["media", "zone"]
            }
        }
    }

import os
from PIL import Image

def GenerateZoneBackdrop(
    media: str,
    zone: str,
    output: str = "zone_backdrop.png",
    width: int = 1328,
    height: int = 1328,
    seed: int = -1,
    char_image: str = None,
):
    """Generate a harmonized sibling environment for a specific zone.
    Shares atmospheric DNA with source media but introduces distinct landmarks."""
    
    if not os.path.exists(media):
        raise FileNotFoundError(f"Environment source not found: {media}")

    background = video_to_img(media)

    images = [background]
    char_desc = ""

    # 🔑 CONDITIONAL CHARACTER INJECTION
    if char_image and os.path.exists(char_image):
        with Image.open(char_image) as img:
            images.append(char_image)  # Pass path for EditImage compatibility
            raw_desc = img.info.get('Description', 'character')
            char_desc = (
                f"A single {raw_desc}. "
                "Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour. "
                "Position naturally within the space, matching environmental lighting and perspective. "
            )

    # 🎨 ATMOSPHERIC CONTINUITY + NEW LANDMARKS PROMPT
    # Key shift: Don't ask for camera movement. Ask for a harmonized sibling shot.
    prompt_parts = [
        f"Generate a cinematic environment shot of {zone}.",
        "This is a DISTINCT AREA within the same overall location as the reference image.",
        "ATMOSPHERIC CONTINUITY (MANDATORY): Match the reference's color grading, lighting temperature, volumetric atmosphere, architectural style, and material textures exactly.",
        "NEW LANDMARKS (MANDATORY): Introduce 2-3 focal architectural features unique to this zone (e.g., 'polished mahogany bar counter with brass rail', 'vaulted alcove with flickering sconces', 'floor-to-ceiling stained glass window').",
        "SPATIAL LOGIC: The new landmarks must feel physically connected to the reference space (same building, same era, same design language) but occupy a different compositional zone.",
        char_desc,
        "COMPOSITION: Wide 16:9 Panavision 70mm framing, cinematic depth of field, clear mid-ground for character placement, NO foreground occlusions.",
        "NO characters (unless specified via char_image), NO text, NO style drift. Photorealistic cinematic environment shot."
    ]

    if char_image:
        prompt_parts.append("NO additional characters beyond the specified subject.")

    prompt = " ".join([p.strip() for p in prompt_parts if p.strip()])

    return EditImage(
        prompt=prompt,
        images=images,  # Reference image for atmospheric/style guidance
        output=output,
        width=width,
        height=height,
        seed=seed
    )



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
    args = parser.parse_args()

    print(EditImage(args.prompt, args.images, args.output, args.width, args.height, args.seed))
