from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
import torch, os, gc, random
from typing import Dict, Any, Tuple
from PIL import Image, ImageFilter
from image_analysis import AnalyzeImage
from config import load_environ

load_environ()

# ─────────────────────────────────────────────────────────────
# STRICT RESOLUTION CONSTRAINTS (ERNIE-Image-Turbo Native)
# ─────────────────────────────────────────────────────────────
VALID_RESOLUTIONS = [
    (1024, 1024), (848, 1264), (1264, 848),
    (768, 1376), (896, 1200), (1376, 768), (1200, 896)
]

def _resolve_resolution(width: int, height: int) -> Tuple[int, int]:
    """Map requested dimensions to the closest valid native resolution by aspect ratio."""
    target = (int(width), int(height))
    if target in VALID_RESOLUTIONS:
        return target
    target_ar = width / height
    return min(VALID_RESOLUTIONS, key=lambda res: abs((res[0] / res[1]) - target_ar))

def _parse_video_size(size_str: str) -> Tuple[int, int]:
    """Parse 'WxH' string to tuple. Returns (0,0) if invalid."""
    try:
        w, h = map(int, size_str.split('x'))
        return (w, h)
    except:
        return (0, 0)

# ─────────────────────────────────────────────────────────────
# VIDEO FRAME PADDING / BORDER EXTENSION
# ─────────────────────────────────────────────────────────────
def _pad_to_video_frame(src_path: str, target_w: int, target_h: int, style: str = "blur") -> str:
    """Resize and pad graphic to exact video frame size without distortion."""
    img = Image.open(src_path).convert("RGB")
    
    # Scale to fit within target frame
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create background
    if style == "blur":
        bg = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=40))
        canvas = bg
    elif style == "solid":
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    elif style == "gradient_edge":
        # Average edge color for seamless fade
        edge_color = img.resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
        canvas = Image.new("RGB", (target_w, target_h), edge_color)
    else:
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        
    # Center paste
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    canvas.save(src_path)
    return src_path

# ─────────────────────────────────────────────────────────────
# GRAPHIC GENERATION PIPELINE
# ─────────────────────────────────────────────────────────────
class GraphicGen(object):
    def __init__(self, vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
            
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        self.model_id = "baidu/ERNIE-Image-Turbo"
        #self.model_id = "baidu/ERNIE-Image"

        self.pipe = ErnieImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=self.model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                ModelConfig(model_id=self.model_id, origin_file_pattern="text_encoder/model.safetensors", **vram_config),
                ModelConfig(model_id=self.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ],
            tokenizer_config=ModelConfig(model_id=self.model_id, origin_file_pattern="tokenizer/"),
            vram_limit=vrlimit,
        )

    def generate(self, prompt, output, width, height, seed):
        if seed == -1: 
            seed = random.randint(0, 1000000)

        if 'Turbo' in self.model_id:
            image = self.pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=8,
                cfg_scale=1.0,
                height=height,
                width=width,
                sigma_shift=4.0
            )
        else:
            image = self.pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=50,
                cfg_scale=4.0,
                height=height,
                width=width
            )
        image.save(output)
        os.utime(output, None)
        
        status = {"status": "success", "output_path": output, "prompt": prompt, "description": ''}
        if os.environ.get('BATCH', 'True') == 'False':
            analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
            status['description'] = analysis.get('analysis', '')
            
        return status

    def __del__(self):
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────
def GenerateGraphicSchema():
    return {
        "type": "function",
        "function": {
            "name": "generate_graphic",
            "description": "Generate 2D graphics, title screens, promotional banners, or UI mockups. Automatically maps to valid ERNIE resolutions and can pad to exact video frame sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the graphic. Include style keywords: 'minimalist', 'bold typography', 'gradient background', 'flat design', 'cinematic title card'."
                    },
                    "output": {
                        "type": "string",
                        "description": "Output file path."
                    },
                    "width": {"type": "integer", "default": 1024},
                    "height": {"type": "integer", "default": 1024},
                    "seed": {"type": "integer", "default": 42, "description": "Use -1 for random."},
                    "target_video_size": {
                        "type": "string",
                        "default": "",
                        "description": "Optional. Target video resolution like '1920x1080'. If provided, graphic will be padded to this exact size without distortion."
                    },
                    "padding_style": {
                        "type": "string",
                        "default": "blur",
                        "enum": ["blur", "solid", "gradient_edge"]
                    }
                },
                "required": ["prompt"]
            }
        }
    }

def GenerateGraphic(prompt='', output='tmp_graphic.png', width=1024, height=1024, seed=42, target_video_size='', padding_style='blur'):
    # 1. Enforce model constraints
    w, h = _resolve_resolution(width, height)
    
    # 2. Generate
    gen = GraphicGen()
    status = gen.generate(prompt, output, w, h, int(seed))
    del gen
    
    # 3. Pad to video frame if requested
    if status['status'] == 'success' and target_video_size:
        tv_w, tv_h = _parse_video_size(target_video_size)
        if tv_w > 0 and tv_h > 0:
            try:
                _pad_to_video_frame(output, tv_w, tv_h, padding_style)
                status['padded_to'] = target_video_size
            except Exception as e:
                status['padding_warning'] = str(e)
                
    return status

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ERNIE Graphic Generation + Video Frame Padding')
    parser.add_argument('-P', '--prompt', type=str, default='cinematic title card, bold sans-serif text, dark gradient background')
    parser.add_argument('-W', '--width', type=int, default=1376)
    parser.add_argument('-H', '--height', type=int, default=768)
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output_graphic.png')
    parser.add_argument('-V', '--video', type=str, default='', help='Target video size like 1920x1080')
    parser.add_argument('-S', '--padding', type=str, default='blur', choices=['blur','solid','gradient_edge'])
    args = parser.parse_args()

    print(GenerateGraphic(args.prompt, args.output, args.width, args.height, args.seed, args.video, args.padding))