import torch, os, gc, time, numpy as np
from PIL import Image
from pathlib import Path
import tqdm

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
import random
from plan10.lib.config import load_environ
load_environ()

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))
DURATION = 1

model_id =  f"alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP" 

def _ensure_pipeline(vrlimit=14):
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    _pipe = None

    configs = [
        ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
    ]

    _pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=configs,
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=vrlimit,
    )
    return _pipe

def Splice(prompt='',first='', last='', output='output.mp4', duration_sec=1, width=WIDTH, height=HEIGHT, seed=-1):

    original_prompt = prompt

    width = int(width)
    height = int(height)
    seed = int(seed)
    duration_sec = int(duration_sec)
    fps = 16

    if seed == -1:
        seed = random.randint(0,1000000)

    total_frames = (duration_sec * fps) + 1
    _pipe = _ensure_pipeline()

    video = None
    try:
        video = _pipe(
            prompt=prompt,
            input_image=Image.open(first),
            end_image=Image.open(last),
            width=width, height=height,
            tiled=True,
            num_frames=total_frames,
            seed=seed,
        )

        save_video(video, output, fps=fps, quality=5)
            
        # Post-processing
        
        return {
            "status": "success",
            "output_path": output,
            "frames": len(video),
            "description": '',
            "prompt": eprompt
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        del video
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prompt', type=str, default='transitions smoothly from first to last frame', required=False)
    parser.add_argument('-F', '--first', type=str, default='', help='First frame')
    parser.add_argument('-L', '--last', type=str, default='', help='Last frame')
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-D', '--duration', type=float, default=DURATION)
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-S', '--seed', type=int, default=SEED)
    args = parser.parse_args()
    Splice(prompt=args.prompt, first=args.first, last=args.last, output=args.output, duration_sec=args.duration, width=args.width, height=args.height, seed=-args.seed)
    

if __name__ == '__main__':
    main()