#!/usr/bin/env python3
"""
Batch S2V lip-sync: Match CU images with audio and generate videos.
FIXED: No audio padding - preserves lip-sync accuracy

Usage:
    python batch_s2v_closeups.py ./cu_output/ ./audio_output/ ./s2v_output/
"""

import torch, torchaudio, librosa, sys, os, gc
from PIL import Image
import numpy as np
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from glob import glob
from pathlib import Path
from omnivoice import OmniVoice
import torchaudio.functional as F
from util import video_to_img
from image_analysis import AnalyzeImage
import random

from config import load_environ
load_environ()

# =============================================================================
# 1. LOAD S2V PIPELINE
# =============================================================================
def load_s2v_pipe():
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
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
        vram_limit=14,
    )

    pipe.load_lora(pipe.dit, './loras/wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors', alpha=1.0)


    return pipe

def speech_to_video2(
    pipe,
    prompt,
    input_image,
    input_audio,
    sample_rate,
    max_frames_per_clip=80,
    height=832,
    width=448,
    cfg_scale=1.0,
    num_inference_steps=4,
    fps=16,
    motion_frames=73,
    chunk_size=16,
    save_path=None,
    add_smile_outro=False,
    outro_duration=0.5,
    seed=-1
):

    # Speech-to-video
    video = pipe(
        prompt=prompt,
        input_image=input_image,
        seed=seed,
        num_frames=97,
        height=height,
        width=width,
        audio_sample_rate=sample_rate,
        input_audio=input_audio,
        num_inference_steps=4,
    )
    save_video_with_audio(video[1:], save_path, "temp.wav", fps=16, quality=5)
    video_to_img(save_path).save('tmp.png')
    analysis = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")
    return {"status":"success", "output_path": save_path, "frames": len(video), "description": analysis['analysis'], "prompt": prompt }


# =============================================================================
# 2. S2V GENERATION FUNCTION (NO AUDIO PADDING)
# =============================================================================
def speech_to_video(
    pipe,
    prompt,
    input_image,
    input_audio,
    sample_rate,
    max_frames_per_clip=80,
    height=768,
    width=448,
    cfg_scale=1.0,
    num_inference_steps=4,
    fps=16,
    motion_frames=73,
    chunk_size=16,
    save_path=None,
    add_smile_outro=False,
    outro_duration=0.5,
    seed=-1
):
    import math
    import numpy as np
    
    original_duration = len(input_audio) / sample_rate
    
    # --- APPEND SILENCE FOR SMILE OUTRO ---
    if add_smile_outro:
        silence_samples = int(outro_duration * sample_rate)
        input_audio = np.concatenate([input_audio, np.zeros(silence_samples)])
        print(f"    → Added {outro_duration}s silence for smile outro")
    
    # Calculate frames: ROUND UP TO NEXT FULL SECOND
    audio_duration = len(input_audio) / sample_rate
    duration_seconds = math.ceil(audio_duration)
    required_frames = duration_seconds * fps
    
    # Round to chunk size for model compatibility
    required_frames = ((required_frames + chunk_size - 1) // chunk_size) * chunk_size
    
    print(f"    Audio duration: {original_duration:.2f}s + {outro_duration}s outro → {duration_seconds}s total")
    print(f"    Required frames: {required_frames} (chunk size: {chunk_size})")
    
    # Pre-calculate embeddings from EXTENDED audio (with silence)
    with torch.no_grad():
        audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
            pipe=pipe,
            input_audio=input_audio,  # Includes silence for outro
            audio_sample_rate=sample_rate,
            s2v_pose_video=None,
            num_frames=required_frames,
            height=height,
            width=width,
            fps=fps,
        )
    
    print(f"    Num repeats from model: {num_repeat}")
    
    # Trust model's num_repeat for clip count
    num_clips = num_repeat if num_repeat > 0 else 1
    
    motion_video = None
    video = []
    
    for clip_idx in range(num_clips):
        print(f"    Generating clip {clip_idx + 1}/{num_clips}...")
        
        clip_audio_embeds = audio_embeds[clip_idx] if isinstance(audio_embeds, list) and clip_idx < len(audio_embeds) else audio_embeds
        
        if clip_idx == num_clips - 1:
            frames_generated = clip_idx * (max_frames_per_clip - motion_frames)
            clip_frames = required_frames - frames_generated
        else:
            clip_frames = max_frames_per_clip
        
        clip_frames = max(1, clip_frames)
        clip_frames = ((clip_frames + chunk_size - 1) // chunk_size) * chunk_size
        
        print(f"    Clip {clip_idx + 1}: {clip_frames} frames")
        
        # Optional: Switch prompt for outro portion
        if add_smile_outro and clip_idx == num_clips - 1:
            current_prompt = '' + ", neutral expression"
        else:
            current_prompt = prompt
        
        current_clip_tensor = pipe(
            prompt=current_prompt,
            input_image=input_image,
            negative_prompt="",
            seed=seed,
            num_frames=clip_frames,
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            audio_embeds=clip_audio_embeds,
            s2v_pose_latents=None,
            motion_video=motion_video,
            num_inference_steps=num_inference_steps,
            output_type="floatpoint",
        )
        
        if clip_idx == 0:
            overlap_frames_num = min(motion_frames, current_clip_tensor.shape[2])
            motion_video = current_clip_tensor[:, :, -overlap_frames_num:, :, :].clone()
        else:
            overlap_frames_num = min(motion_frames, current_clip_tensor.shape[2])
            motion_video = torch.cat(
                (motion_video[:, :, overlap_frames_num:, :, :], 
                 current_clip_tensor[:, :, -overlap_frames_num:, :, :]), 
                dim=2
            )
        
        current_clip_quantized = pipe.vae_output_to_video(current_clip_tensor)
        video.extend(current_clip_quantized)
        
        torch.cuda.empty_cache()

    # Save video with ORIGINAL audio (without trimming video)
    if save_path:
        import tempfile
        import soundfile as sf
    
        # Save extended audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_audio = tmp.name
            sf.write(temp_audio, input_audio, sample_rate)  # input_audio includes silence
    
        # Now save with extended audio
        save_video_with_audio(video, save_path, temp_audio, fps=fps, quality=5)

        video_to_img(save_path).save('tmp.png')
        analysis = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")
    
        # Clean up
        os.remove(temp_audio)
    
        print(f"    Output: {len(video)} frames, {len(video)/fps:.2f}s")
    return {"status":"success", "output_path": save_path, "frames": len(video), "description": analysis['analysis'], "prompt": prompt }

def create_audio_and_free_vram(
    text, 
    instruct='female, low pitch, british accent', 
    max_retries=2,
    max_duration_seconds=5.0,  # ← New configurable limit
    target_sr=16000,            # ← Your workflow's target sample rate
    seed=-1
):
    """
    Generate audio via OmniVoice with automatic duration enforcement.
    If output exceeds max_duration_seconds, regenerate with duration= parameter.
    """
    start_silence_ms=300    # ← NEW: Silence before speech (ms)
    end_silence_ms=500      # ← NEW: Silence after speech (ms)
    speed = 0.85
    for attempt in range(1, max_retries + 1):
        torch.cuda.empty_cache()
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32)
        if seed != -1:
            torch.cuda.manual_seed(seed)
        
        try:
            with torch.no_grad():
                # First attempt: natural generation (no duration limit)
                audio = model.generate(text=text, instruct=instruct, speed=speed)
            
            # Save and load for inspection
            torchaudio.save("temp.wav", audio[0], 24000)
            input_audio, sr = librosa.load("temp.wav", sr=target_sr, mono=True, dtype=np.float32)
            
            # Sanity check: weak/corrupt output
            if np.max(np.abs(input_audio)) < 0.02 or np.isnan(input_audio).any():
                print(f"⚠️ Weak/corrupt output (attempt {attempt}), retrying...")
                continue

            # ← NEW: Add silence padding
            start_samples = int((start_silence_ms / 1000) * sr)
            end_samples = int((end_silence_ms / 1000) * sr)
            
            input_audio = np.concatenate([
                np.zeros(start_samples),  # Start silence
                input_audio,
                np.zeros(end_samples)      # End silence
            ])
            
            # ← NEW: Duration enforcement
            actual_duration = len(input_audio) / sr
            if actual_duration > max_duration_seconds:
                print(f"⚠️ Audio too long ({actual_duration:.2f}s > {max_duration_seconds}s), regenerating with duration limit...")
                
                # Regenerate with explicit duration parameter [[1]][[3]]
                with torch.no_grad():
                    audio = model.generate(
                        text=text, 
                        instruct=instruct,
                        duration=max_duration_seconds  # ← Enforce hard cap
                    )
                
                # Re-save and reload the constrained output
                torchaudio.save("temp.wav", audio[0], 24000)
                input_audio, sr = librosa.load("temp.wav", sr=target_sr, mono=True, dtype=np.float32)
                
                # Final sanity check on constrained output
                if np.max(np.abs(input_audio)) < 0.02 or np.isnan(input_audio).any():
                    print(f"⚠️ Constrained output weak, retrying...")
                    continue
                    
                print(f"✅ Duration-constrained audio: {len(input_audio)/sr:.2f}s")
            else:
                print(f"✅ Clean audio generated (attempt {attempt}, {actual_duration:.2f}s)")
            
            # Cleanup and return
            del model, audio
            gc.collect()
            torch.cuda.empty_cache()
            return input_audio, sr
            
        except Exception as e:
            print(f"❌ Failed (attempt {attempt}): {e}")
        finally:
            try: del model, audio
            except: pass
            gc.collect()
            torch.cuda.empty_cache()
            
    raise RuntimeError(f"Failed to generate valid audio ≤{max_duration_seconds}s after {max_retries} retries.")


def GenerateTalkingVideoV2(
    prompt='',
    dialog='Hello, how are you, I am pleased to meet you.',
    voice='female, low pitch, british accent',
    media='',
    output='output.mp4',
    width=480,
    height=832,
    seed=-1):

    width = int(width)
    height = int(height)
    seed = int(seed)

    input_image  = video_to_img(media, width, height, True)
    input_audio, sample_rate = create_audio_and_free_vram(dialog, voice, seed=seed)
    cfg_scale = 1.0 
    num_inference_steps = 4
    fps = 16
    motion_frames = 73
    chunk_size = 16
    max_frames_per_clip = 80
    add_smile_outro=True
    outro_duration=0.5

    pipe = load_s2v_pipe()
    return speech_to_video(
        pipe,
        prompt,
        input_image,
        input_audio,
        sample_rate,
        max_frames_per_clip,
        height,
        width,
        cfg_scale,
        num_inference_steps,
        fps,
        motion_frames,
        chunk_size,
        output,
        add_smile_outro,
        outro_duration,
        seed
    )


def GenerateTalkingVideo(
    prompt='',
    audio='',
    media='',
    output='output.mp4',
    width=480,
    height=832,
    seed=-1):

    width = int(width)
    height = int(height)
    seed = int(seed)

    if seed == -1:
        seed = random.randint(0,1000000)

    input_image  = video_to_img(media, width, height, True)
    input_audio, sample_rate = librosa.load(audio, sr=16000, mono=True, dtype=np.float32)
    cfg_scale = 1.5 
    num_inference_steps = 4
    fps = 16
    motion_frames = 73
    chunk_size = 16
    max_frames_per_clip = 80
    add_smile_outro=True
    outro_duration=0.5

    pipe = load_s2v_pipe()
    return speech_to_video(
        pipe,
        prompt,
        input_image,
        input_audio,
        sample_rate,
        max_frames_per_clip,
        height,
        width,
        cfg_scale,
        num_inference_steps,
        fps,
        motion_frames,
        chunk_size,
        output,
        add_smile_outro,
        outro_duration,
        seed
    )

def GenerateTalkingVideoSchema():
    return {
        "type": "function",
        "function": {
            "name": "dialog_to_video",
            "description": "Create a talking video from an image with lip-synced audio using OmniVoice Voice Design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string", 
                        "description": "Visual action/facial expression (e.g., 'speaking warmly, slight head tilt')."
                    },
                    "media": {
                        "type": "string", 
                        "description": "Path or alias of source image/video."
                    },
                    "audio": {
                        "type": "string",
                        "description": "Path to audio file to use for audio",
                    },
                    "width": {"type": "integer", "description": "Video width (divisible by 64). Default: 480."},
                    "height": {"type": "integer", "description": "Video height (divisible by 64). Default: 832."},
                    "seed": {"type": "integer", "description": "Random seed for reproducibility. -1 for random."}
                },
                "required": ["prompt", "media", "audio"]
            }
        }
    }

    
    
    

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prompt', type=str, default='')
    parser.add_argument('-T', '--text', type=str, default='Hello, how are you, I am pleased to meet you.')
    parser.add_argument('-V', '--voice', type=str, default='female, low pitch, british accent')
    parser.add_argument('-I', '--image', type=str, required=True)
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-W', '--width', type=int, default=832)
    parser.add_argument('-H', '--height', type=int, default=480)
    parser.add_argument('-S', '--seed', type=int, default=42)
    args = parser.parse_args()
    GenerateTalkingVideoV2(args.prompt, args.text, args.voice, args.image, args.output, args.width, args.height,args.seed)

