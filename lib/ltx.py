import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from PIL import Image
from modelscope import dataset_snapshot_download

from config import load_environ
load_environ()

import logging, os, gc
import json
from time import sleep
from pathlib import Path
from util import video_to_img
from image_analysis import AnalyzeImage, EnhancePrompt
from image_gen import add_metadata_char
import random

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
ANIME = "_anime" if os.environ.get("ANIME","False") != "False" else ""
DISTILLED = "DISTILLED" in os.environ.get("LTX","False")

enhance_path = f'./system/ltx_enhancer{ANIME}.txt'

def i2v(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cuda.matmul.allow_tf32 = True

    width, height = (720, 1280) if height > width else (1280, 720)

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

    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-distilled.safetensors" if DISTILLED else "ltx-2.3-22b-dev.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.0.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config=None if DISTILLED else ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-distilled-lora-384.safetensors"),
        vram_limit=int(os.environ["VRAM"]),
    )

    # Force explicit SFX and ban melody structure in the positive prompt
    sfx_modifiers = ", realistic sound effects only, crisp SFX, ambient background noise, completely devoid of music, no BGM, no instruments"
    final_prompt = f"{prompt}{sfx_modifiers}" if prompt else "ambient sound effects, SFX, absolute no music"

    # Purged "silent or muted audio" to allow empty spaces, heavily punished music architecture
    negative_prompt = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, music, background music, BGM, melody, song, soundtrack, musical instruments, synth, "
        "singing, vocals, rhythm, beats, distorted voice, robotic voice, echo, background noise, off-sync audio, "
        "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, "
        "unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
        "cinematic oversaturation, stylized filters, or AI artifacts."
    )
    num_frames = (duration_sec * 24) + 1

    image = Image.open(media).convert("RGB").resize((width, height))
    
    # first frame
    video, audio = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        tiled=True,
        use_two_stage_pipeline=not DISTILLED,
        use_distilled_pipeline = DISTILLED,
        input_images=[image],
        input_images_indexes=[0],
        input_images_strength=1.0,
    )
    write_video_audio_ltx2(
        video=video,
        audio=audio,
        output_path=output,
        fps=24,
        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
    )
    
    del pipe
    gc.collect()
    if torch.cuda and torch.cuda.is_available():  # ✅ Was `if torch.cuda:` (always truthy)
        torch.cuda.empty_cache()

def GenerateVideo(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):

        print(f"PROMPT: {prompt}")
        
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
                end_image = video_to_img(media.pop(), width, height, True, False)
        else:
            start_image = media

        print(f"MEDIA: {start_image}")

        original_prompt = prompt

        width = int(width)
        height = int(height)
        seed = int(seed)
        duration_sec = int(duration_sec)
        fps = 24

        if seed == -1:
            seed = random.randint(0,1000000)

        total_frames = (duration_sec * fps) + 1

        print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
        print(f"   Resolution: {width}x{height}")

        current_source = video_to_img(start_image, width, height, True, True)
        current_source.save('tmp.png')

        if not prompt:
            prompt = "The characters stand and act naturally. "

        eprompt = EnhancePrompt(start_image, prompt, enhance_path)

        print("CURRENT PROMPT: ",eprompt)

        try:
            i2v(eprompt, 'tmp.png', output, 
                    duration_sec, width, height, seed)
            description = ''
                
            # Post-processing
            if os.environ.get('BATCH', 'False') == 'False':
                tmp_img = video_to_img(f'{output}', width, height)
                tmp_img.save('tmp.png')
                description = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")['analysis']
            
            return {
                "status": "success",
                "output_path": output,
                "frames": (duration_sec * fps) + 1,
                "description": description,
                "prompt": eprompt
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

import math

def count_syllables(word):
    """Count syllables in a word using heuristic rules."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    
    # Special cases
    if len(word) <= 2:
        return 1
    
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    # Adjustments for silent endings
    if word.endswith('e') and count > 1:
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if word.endswith('ed') and count > 1:
        if word[-3] not in 'td':
            count -= 1
    
    return max(1, count)


def estimate_duration(text):
    """
    Estimate TTS duration using syllable count.
    Formula: ceil((syllables × 0.37) + 1.0) seconds
    """
    words = text.split()
    total_syllables = sum(count_syllables(w) for w in words)
    duration = (total_syllables * 0.37) + 3.0
    return math.ceil(duration)


def GenerateTalkingVideo(
    prompt='',
    text='',
    audio='',
    media='',
    output='output.mp4',
    width=WIDTH,
    height=HEIGHT,
    seed=-1):
    print(f"PROMPT: {prompt}")
    
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
            end_image = video_to_img(media.pop(), width, height, True, False)
    else:
        start_image = f'{os.getcwd()}/{media}'

    ref_audio = f'{os.getcwd()}/{audio}'

    print(f"MEDIA: {start_image}")

    original_prompt = prompt

    width = int(width)
    height = int(height)
    seed = int(seed)
    duration_sec = 10 #int(estimate_duration(text))
    fps = 24

    if seed == -1:
        seed = random.randint(0,1000000)

    total_frames = (duration_sec * fps) + 1

    print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
    print(f"   Resolution: {width}x{height}")

    current_source = video_to_img(start_image, width, height, True, True)
    current_source.save('tmp.png')

    if not prompt:
        prompt = "The characters stand and act naturally. "

    desc = Image.open(start_image).info.get('Description')
    if not desc:
        desc = add_metadata_char(start_image, '', seed)

    eprompt = f'The person can be described as {desc}. The person says "{text}." {prompt}.'

    print("ORIGINAL PROMPT: ",eprompt)

    eprompt = EnhancePrompt(start_image, eprompt, enhance_path)

    print("CURRENT PROMPT: ",eprompt)

    try:
        i2v(eprompt, start_image, output, 
                duration_sec, width, height, seed)
        description = ''
            
        # Post-processing
        if os.environ.get('BATCH', 'False') == 'False':
            tmp_img = video_to_img(f'{output}', width, height)
            tmp_img.save('tmp.png')
            description = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")['analysis']
        
        return {
            "status": "success",
            "output_path": output,
            "frames": (duration_sec * fps) + 1,
            "description": description,
            "prompt": eprompt
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise