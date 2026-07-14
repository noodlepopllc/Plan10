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
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

from diffsynth.diffusion.template import TemplatePipeline
from image_analysis import AnalyzeImage

class FrameDetailer:
    def __init__(self, pipe=None):
        """
        Initialize the frame detailer/upscaler.
        
        Args:
            pipe: Existing Flux2ImagePipeline (optional, will create if not provided)
        """
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        if pipe is None:
            print("Loading FLUX.2 Klein pipeline for frame enhancement...")
            self.pipe = Flux2ImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
                    ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
                    ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
                ],
                tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
            )
        else:
            self.pipe = pipe
        
        print("Loading upscaler template...")
        self.template = TemplatePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Upscaler")],
        )
    
    def enhance(self, image, description=None, output_path="enhanced.png", seed=42, width=None, height=None):
        """
        Enhance/upscale an image by restoring details.
        
        Args:
            image: PIL Image or path to image
            description: Optional description to guide enhancement
            output_path: Optional path to save enhanced image
            seed: Random seed for reproducibility
            width: Output width (optional, will use input width if not specified)
            height: Output height (optional, will use input height if not specified)
            
        Returns:
            Enhanced PIL Image
        """
        if isinstance(image, str):
            media = Image.open(image)
        
        # Default to input dimensions if not specified
        if width is None:
            width = media.width
        if height is None:
            height = media.height
        
        # Generate a brief description if not provided
        if not description:
            description = AnalyzeImage(output_path, "Briefly describe this image, no more than 100 words")['analysis']
        
        # Use the upscaler template with the input image
        enhanced = self.template(
            self.pipe,
            prompt=description,
            seed=seed,
            cfg_scale=4,
            num_inference_steps=50,
            template_inputs=[{
                "image": media,
                "prompt": description,
            }],
            negative_template_inputs=[{
                "image": media,
                "prompt": "",
            }],
        )
        
        enhanced.save(output_path)

        os.utime(output_path, None) 
        status = {"status": "success", "output_path": output_path, "prompt": description, "description": ''}
        if os.environ['BATCH'] == 'False':
            analysis = AnalyzeImage(output_path, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis['analysis']
        return status

    def __del__(self):
        gc.collect()
        if torch.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def get_pipe(self):
        if not self.pipe:
            self.__enter__()
        return self.pipe
    
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
            self.__enter__()
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

def EditImage(prompt='', images=[''], output='tmp_edit.png', width=WIDTH, height=HEIGHT, seed=42, img_edit=None):
    if not img_edit:
        edit = ImageEdit()
    else:
        edit = img_edit
    status = edit.generate(prompt, images, output, int(width), int(height), int(seed))
    if not img_edit:
        del edit
    return status

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, os
    os.environ['BATCH'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--images', action='append', default=[], help='Input images')
    parser.add_argument('-P', '--prompt', type=str, default='remove text', help='Edit prompt')
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    args = parser.parse_args()

    print(EditImage(args.prompt, args.images, args.output, args.width, args.height, args.seed))
