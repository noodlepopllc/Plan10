from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import torch
import os, gc
from typing import Union
from image_analysis import AnalyzeImage
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

    def __del__(self):
        del self.pipe 
        gc.collect()
        if torch.cuda:
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
