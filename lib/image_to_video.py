import torch, os, gc, time, numpy as np
from PIL import Image
from pathlib import Path
import tqdm
from util import video_to_img
from image_analysis import AnalyzeImage, EnhancePrompt
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
from qwen_llm import llm_analyze_media
import random
from config import load_environ
load_environ()
from image_gen import GenerateImage


enhancer = '''You are a cinematic prompt optimizer for image‑to‑video inpainting. Expand the user’s text into a clear, stable prompt inspired by late‑1960s to early‑1970s Panavision 60 cinematography.

By default, the camera must remain completely static: locked‑off camera, static tripod, no pan, no tilt, no zoom, no handheld drift, no micro‑movement, no sway, no breathing motion, zero horizontal or vertical drift. Only allow camera movement if the user explicitly requests it.

Requirements:
1. Always preserve the user’s requested action or pose change exactly as stated. Do not remove or weaken any movement.
2. By default, the camera must remain completely static: locked‑off camera, static tripod, no pan, no tilt, no zoom, fixed composition. Only allow camera movement if the user explicitly requests it.
3. Add subtle cinematic details that do not contradict the source image: warm late‑afternoon or tungsten light, soft contrast, mild halation, gentle grain.
4. Add only small natural movements that support the user’s requested action.
5. Do not add new objects, characters, or major scene changes.
6. Output in English, 35–55 words, concise and cinematic.

Directly output the rewritten prompt.


'''

model_id = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP"
model_id = "alibaba-pai/Wan2.1-Fun-V1.1-14B-InP"

def _ensure_pipeline(vrlimit=14):

    # === Global Pipeline Setup (Wan 2.1 I2V) ===
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

    _pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
            ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=vrlimit,
    )

    if '1.3B' in model_id:
        _pipe.load_lora(_pipe.dit, './loras/loras_accelerators/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors', alpha=1.0)
    else:
        _pipe.load_lora(_pipe.dit, './loras/wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors', alpha=1.0)
    return _pipe


def GenerateVideo(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=832, height=480, seed=-1):

        
        if isinstance(prompt, list):
            prompt = prompt.pop()
        
        start_image = ''
        end_image = None

        if not media:
            GenerateImage(prompt = prompt, output='first_frame.png', width=width, height=height, seed=seed)
            media='first_frame.png'

        if isinstance(media, list):
            start_image = media.pop(0)
            if len(media) > 0:
                end_image = video_to_img(media.pop(), width, height)
        else:
            start_image = media

        width = int(width)
        height = int(height)
        seed = int(seed)
        duration_sec = int(duration_sec)
        fps = 15

        if seed == -1:
            seed = random.randint(0,1000000)

        total_frames = (duration_sec * 16) + 1

        print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
        print(f"   Resolution: {width}x{height}")

        current_source = video_to_img(start_image, width, height)
        current_source.save('tmp.png')

        if not prompt:
            prompt = "The characters stand and act naturally. "

        eprompt = prompt #llm_analyze_media(start_image, prompt, enhancer)['analysis']

        print("CURRENT PROMPT: ",eprompt)

        _pipe = _ensure_pipeline()

        num_steps = 8  if '1.3B' in model_id else 4


        
        try:
            video = _pipe(
                prompt=eprompt,
                input_image=current_source,
                end_image=end_image,
                width=width, height=height,
                tiled=True,
                num_frames=total_frames,
                cfg_scale=1.0,
                num_inference_steps=num_steps,
                seed=seed,
            )

            save_video(video, output, fps=15, quality=5)
            description = ''
                
            # Post-processing
            if os.environ['BATCH'] == 'False':
                tmp_img = video_to_img(output, width, height)
                tmp_img.save('tmp.png')
                description = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")['analysis']
            
            return {
                "status": "success",
                "output_path": output,
                "frames": len(video),
                "description": description,
                "prompt": prompt
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        finally:
            del video
            gc.collect()
            torch.cuda.empty_cache()


def GenerateVideoSchema():
    return {
        "type": "function",
        "function": {
            "name": "image_to_video",
            "description": "Create a video from an image and a prompt using Wan 2.1 I2V. Pass a list of prompts for multi-window control (1 prompt per ~5s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt for what should be happening in the video"},
                    "media": {"type": "string", "description": "Path to starting image or video."},
                    "width": {"type": "integer", "default": 864},
                    "height": {"type": "integer", "default": 480},
                    "seed": {"type": "integer", "default": 42},
                    "duration_sec": {"type": "integer", "description": "Total length in seconds", "default": 10}
                },
                "required": ["prompt", "media"]
            }
        }
    }

def GenerateI2VPromptSchema():
    return {
        "type": "function",
        "function": {
            "name": "generate_i2v_prompt",
            "description": "Analyze a static image and output a short, action-oriented prompt for Image-to-Video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the generated image."},
                    "motion_style": {
                        "type": "string", 
                        "enum": ["subtle_cinematic", "dynamic", "ambient_only", "camera_movement"],
                        "default": "subtle_cinematic"
                    }
                },
                "required": ["image_path"]
            }
        }
    }

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prompt', type=str, default='', required=False)
    parser.add_argument('-I', '--images', action='append', default=[], help='Input images')
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-D', '--duration', type=float, default=5)
    parser.add_argument('-W', '--width', type=int, default=832)
    parser.add_argument('-H', '--height', type=int, default=480)
    parser.add_argument('-S', '--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Allow JSON array prompts from CLI
    try:
        prompt_input = json.loads(args.prompt)
    except:
        prompt_input = args.prompt
        
    result = GenerateVideo(
        prompt=prompt_input,
        media=args.images,
        output=args.output,
        duration_sec=args.duration,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(result)
