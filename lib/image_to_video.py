import torch, os, gc, time, numpy as np
from PIL import Image
from pathlib import Path
import tqdm
from util import video_to_img
from image_analysis import AnalyzeImage, EnhancePrompt
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
import random
from config import load_environ
load_environ()

# === Global Pipeline Setup (Wan 2.1 I2V) ===
_vram_config = {
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
_STEPS, _CFG = 4, 1.0

def _ensure_pipeline(vrlimit=14):
    global _pipe, _STEPS, _CFG
    if _pipe is not None:
        return
    if "VRAM" in os.environ:
        vrlimit = int(os.environ["VRAM"])

    # === Wan 2.1 I2V Model Config (Single DiT) ===
    _pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            # DiT - Safetensors (DiffSynth-Studio repo)
            ModelConfig(
                model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                origin_file_pattern="Wan2.1-I2V-14B-480P/diffusion_pytorch_model*.safetensors",
                **_vram_config
            ),
            
            # T5 Encoder - PTH (Original Wan-AI repo, no safetensors version)
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **_vram_config
            ),
            
            # VAE - Safetensors (DiffSynth-Studio repo)
            ModelConfig(
                model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                origin_file_pattern="Wan2.1-I2V-14B-480P/Wan2.1_VAE.safetensors",
                **_vram_config
            ),
            
            # CLIP - PTH (Original Wan-AI repo, no safetensors version)
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                **_vram_config
            ),
        ],
        
        # Tokenizer - Usually from a separate T5 repo
        tokenizer_config=ModelConfig(
            model_id="google/umt5-xxl",
            origin_file_pattern="**/*"
        ),
        
        vram_limit=vrlimit,
    )
   lora = ModelConfig(model_id="lightx2v/Wan2.1-Distill-Loras", origin_file_pattern="wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors")

    _pipe.load_lora(
        _pipe.dit, 
        lora, 
        alpha=1.0
    )

def GenerateVideo(prompt='', media='', output='output.mp4', 
                  duration_sec=10, width=832, height=480, seed=-1):

        
        if isinstance(prompt, list):
            prompt = prompt.pop()

        width = int(width)
        height = int(height)
        seed = int(seed)
        duration_sec = int(duration_sec)
        fps = 15
        sliding_window_size=81
        sliding_window_stride=32

        if seed == -1:
            seed = random.randint(0,1000000)

        total_frames = (duration_sec * 16) + 1

        print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
        print(f"   Sliding window: size={sliding_window_size}, stride={sliding_window_stride}")
        print(f"   Resolution: {width}x{height}")

        current_source = video_to_img(media, width, height, True)
        current_source.save('tmp.png')

        if not prompt:
            prompt = GenerateI2VPrompt('tmp.png')['analysis']

        print(prompt)

        _ensure_pipeline()
        
        try:
            video = _pipe(
                prompt=prompt,
                input_image=current_source,
                width=width, height=height,
                num_frames=total_frames,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                cfg_scale=1.0,
                num_inference_steps=4,
                seed=seed,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )

            save_video(video, output, fps=15, quality=5)
                
            # Post-processing
            tmp_img = video_to_img(output, width, height)
            tmp_img.save('tmp.png')
            analysis = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")
            
            return {
                "status": "success",
                "output_path": output,
                "frames": len(video),
                "description": analysis['analysis'],
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

def GenerateI2VPrompt(image_path: str, motion_style: str = "subtle_cinematic", primary_subject: int = 1) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Detect character count
    count_analysis = AnalyzeImage(image_path, "Count visible people/characters. Reply with ONLY a number: 1 or 2.")
    try:
        char_count = int(count_analysis['analysis'].strip())
        if char_count not in [1, 2]: char_count = 1
    except:
        char_count = 1

    # Facial action verbs that actually work in I2V (tested)
    facial_actions = [
        "eyes track movement", "gaze shifts", "eyebrows raise slightly", 
        "lips part", "smile forms", "expression softens", "blinks naturally",
        "head tilts", "chin lifts", "eyes narrow slightly"
    ]
    
    if char_count == 1:
        style_guides = {
            "subtle_cinematic": (
                "POSED still of ONE character. They BREAK the pose with BODY + FACIAL motion. "
                f"Body: shifts weight, turns head, adjusts stance. Face: {random.choice(facial_actions)}. "
                "BOTH required. BANNED: static face, frozen expression, only body movement."
            ),
            "dynamic": (
                "POSED still of ONE character. BOLD action with expressive face. "
                f"Body: spins, gestures, steps forward. Face: {random.choice(facial_actions)}. "
                "Strong verbs. Face MUST animate."
            ),
            "ambient_only": "Ignore character motion. Focus ONLY on environment: wind, light, particles.",
            "camera_movement": (
                "Slow push-in/orbit. Character: small body shift + {random.choice(facial_actions)}. "
                "Camera + subject + face all move."
            )
        }
    else:  # char_count == 2
        style_guides = {
            "subtle_cinematic": (
                "POSED still of TWO characters. They interact with BODY + FACIAL motion. "
                f"Character {primary_subject}: shifts stance + {random.choice(facial_actions)}. "
                "Other character: reacts with subtle facial change. BOTH faces animate. "
                "BANNED: frozen expressions, only one face moves."
            ),
            "dynamic": (
                "POSED still of TWO characters. Coordinated bold action with expressive faces. "
                f"Character {primary_subject}: gestures/turns + {random.choice(facial_actions)}. "
                "Other: reacts visibly with facial expression change. Strong interaction."
            ),
            "ambient_only": "Ignore characters. Focus ONLY on environmental motion.",
            "camera_movement": (
                "Slow pan across both. They perform small coordinated action + facial animation: "
                f"turn heads together + {random.choice(facial_actions)}. Faces MUST animate."
            )
        }
    
    guide = style_guides.get(motion_style, style_guides["subtle_cinematic"])

    subject_clause = "the character" if char_count == 1 else f"both characters (Character {primary_subject} leads)"
    analysis_prompt = (
        f"Analyze this POSED image with {char_count} character(s). Generate EXACTLY ONE short sentence (14-20 words) for Image-to-Video. "
        f"{guide} "
        f"Describe what {subject_clause} DOES next. Include BOTH body motion AND facial animation. "
        f"Use strong active verbs. Output ONLY the motion prompt. No quotes, no intro."
    )

    result = AnalyzeImage(image_path, analysis_prompt)
    raw = result['analysis'].strip().strip('"').strip("'")
    
    # Safety trim
    words = raw.split()
    if len(words) > 24:
        raw = " ".join(words[:24]).rstrip(".,;:")
    
    # Hard ban + facial fallback
    passive_words = {"breathes", "blinks", "breathing", "static", "poses", "frozen"}
    facial_keywords = {"eyes", "gaze", "smile", "lips", "expression", "eyebrows", "tilts", "narrows"}
    
    # If passive OR missing facial motion, force injection
    if any(p in raw.lower() for p in passive_words) or not any(f in raw.lower() for f in facial_keywords):
        if char_count == 1:
            raw = f"Turns head slightly, {random.choice(facial_actions)}."
        else:
            raw = f"Character {primary_subject} turns toward other, {random.choice(facial_actions)}."
    
    prompt = EnhancePrompt(image_path, raw, 'system/Wan_I2V.txt')
    return prompt


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prompt', type=str, default='', required=False)
    parser.add_argument('-I', '--image', type=str, required=True)
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-D', '--duration', type=float, default=10)
    parser.add_argument('-W', '--width', type=int, default=768)
    parser.add_argument('-H', '--height', type=int, default=448)
    parser.add_argument('-S', '--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Allow JSON array prompts from CLI
    try:
        prompt_input = json.loads(args.prompt)
    except:
        prompt_input = args.prompt
        
    result = GenerateVideo(
        prompt=prompt_input,
        media=args.image,
        output=args.output,
        duration_sec=args.duration,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(result)
