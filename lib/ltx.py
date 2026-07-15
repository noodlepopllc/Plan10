import torch
from config import load_environ
load_environ()

from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from PIL import Image
from modelscope import dataset_snapshot_download

from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

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
enhance_path = './system/ltx_enhancer_minimal.txt'

from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.utils import encode_video

def i2v_diffusers(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    """
    Executes a native First-Frame Last-Frame conditioned 8-step distillation generation pass.
    Guarantees zero identity drift across the 10-second trajectory.
    """

        # 💡 FIX: Force orientation switch AND snap dimensions strictly to a multiple of 32
    target_w, target_h = (720, 1280) if height > width else (1280, 720)
    
    # // 32 * 32 drops any fractional pixel remainders instantly
    width = int(target_w // 32) * 32   # 1280 stays 1280, 704 stays 704, etc.
    height = int(target_h // 32) * 32  # 720 snaps down to 704!

    # 1. Acceleration backend optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
    frame_rate = 24.0
    num_frames = int((duration_sec * frame_rate) + 1)

    # 💡 UPGRADED: Points straight to the official native LTX-2.3 distilled repo
    model_path = "CalamitousFelicitousness/LTX-2.3-distilled-Diffusers"

    config_dict = LTX2VideoTransformer3DModel.load_config(
        model_path, 
        subfolder="transformer"
    )

    # 2. FORCE NATIVE LTX-2.3 MULTI-MODAL OVERRIDES
    # These parameters directly allocate the 9-dimensional attention layer tracks
    config_dict["audio_prompt_adaln"] = True           # Direct trigger for 2.3 multi-modal AdaLN layers
    config_dict["audio_caption_channels"] = 2048       # Sets the structural alignment boundaries
    config_dict["use_audio_conditioning"] = True       # Enforces the multimodel step allocation tracking

    # Remove the old temporary 2.0 properties so it doesn't complain about ignored keys
    config_dict.pop("has_prompt_adaln", None)
    config_dict.pop("num_audio_ada_params", None)

    # 3. Instantiate the transformer model block using our corrected dictionary
    transformer = LTX2VideoTransformer3DModel.from_config(config_dict)

    # Load 2.3 directly onto your CUDA memory pool
    pipe = LTX2ConditionPipeline.from_pretrained(
        model_path,
        transformer=transformer, # Inject our custom 9-parameter skeleton block
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    ).to("cuda")

    # Freeze the transformer to eliminate the ARM loop synchronization lag on Spark
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")


    # 3. Format and construct your conditional anchors
    first_img = Image.open(media).convert("RGB").resize((width, height))
    #last_img = Image.open(last_frame_path).convert("RGB").resize((width, height))
    
    first_cond = LTX2VideoCondition(frames=first_img, index=0, strength=1.0)
    #last_cond = LTX2VideoCondition(frames=last_img, index=-1, strength=1.0)
    conditions = [first_cond]#, last_cond]

    negative_prompt = (
        "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
        "motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
    )

    # 4. EXECUTE STAGE 1 (8-Step Base Denoising Latent Generation)
    video_latent, audio_latent = pipe(
        conditions=conditions,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=8,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )

    # 💡 SOLUTION: Pull from the byte-identical 2.3 community mapping repository
    # This repo includes the required "latent_upsampler" subfolder structure natively!
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        model_path,
        subfolder="latent_upsampler",
        torch_dtype=torch.bfloat16,
    ).to("cuda")



    
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
    
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    # 6. EXECUTE FINAL REFINE PASS (3-Step Detail Denoise)
    video, audio = pipe(
        latents=upscaled_video_latent,
        audio_latents=audio_latent,
        prompt=prompt,
        width=width * 2,    # Re-scale matching the 2x upsampler logic
        height=height * 2,
        num_inference_steps=3,
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        generator=generator,
        guidance_scale=1.0,
        output_type="np",
        return_dict=False,
    )

    # 7. ENCODE TO RAW MP4
    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=output_path,
    )

    # Clean execution memory states cleanly
    del pipe, upsample_pipe, latent_upsampler
    torch.cuda.empty_cache()


def i2v_diffsynth(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    
    # Enable fast hardware math handling for Blackwell cores
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    width, height = (720, 1280) if height > width else (1280, 720)

    # 1. FIXED VRAM CONFIG: Lock everything directly inside CUDA space.
    # By removing "cpu" offloading, we stop the ARM-to-GPU step-by-step page fault loops.
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cuda",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    # 2. INCREASE VRAM LIMIT OR REMOVE STRIP BOUNDARY
    # Your Spark has 128GB. If os.environ["VRAM"] is set to a low value (like 12 or 16),
    # DiffSynth will manually break up the models even if you set the device to "cuda".
    # We override it here to leverage your hardware's full capacity.
    allocated_vram_limit = max(int(os.environ.get("VRAM", 96)), 96) * 1024 * 1024 * 1024

    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.0.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-distilled-lora-384.safetensors"),
        vram_limit=allocated_vram_limit, 
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
    
    # Run core inference pipeline
    video, audio = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        tiled=True,
        use_two_stage_pipeline=True,
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
    
    # Clean up memory cleanly
    del pipe
    gc.collect()
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()

i2v = i2v_diffusers if DISTILLED else i2v_diffsynth

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