import torch
from plan10.lib.config import load_environ
load_environ()

from diffsynth.utils.data.audio import read_audio

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image

import logging, os, gc
import json
from time import sleep
from pathlib import Path
from plan10.lib.util import video_to_img
from plan10.lib.image_analysis import AnalyzeImage, EnhancePrompt
from plan10.lib.image_gen import add_metadata_char
import random

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))
ANIME = "_anime" if os.environ.get("ANIME","False") != "False" else ""
VRAM = int(os.environ.get("VRAM", 96))
DURATION = 5 #5 if VRAM < 24 else 10

if ANIME:
    from plan10.lib.anime_gen import GenerateImage
else:
    from plan10.lib.image_gen import GenerateImage

BRIEF = os.environ.get("BRIEF","False") != "False"

#enhance_path = f'./system/ltx_enhancer{ANIME}.txt'
enhance_path = f'./system/ltx_enhancer_minimal{ANIME}.txt' if BRIEF else f'./system/ltx_enhancer{ANIME}.txt'


def i2v_diffsynth(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    
    # Enable fast hardware math handling for Blackwell cores
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    #width, height = (720, 1280) if height > width else (1280, 720)

    # 1. FIXED VRAM CONFIG: Lock everything directly inside CUDA space.
    # By removing "cpu" offloading, we stop the ARM-to-GPU step-by-step page fault loops.
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cpu" if VRAM < 24 else "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    # 2. INCREASE VRAM LIMIT OR REMOVE STRIP BOUNDARY
    # Your Spark has 128GB. If os.environ["VRAM"] is set to a low value (like 12 or 16),
    # DiffSynth will manually break up the models even if you set the device to "cuda".
    # We override it here to leverage your hardware's full capacity.
    allocated_vram_limit = min(VRAM, 96)

    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
        ],
        processor_config=ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )


    # Force explicit SFX and ban melody structure in the positive prompt
    sfx_modifiers = ", realistic sound effects only, crisp SFX, ambient background noise, completely devoid of music, no BGM, no instruments"
    final_prompt = f"{prompt}{sfx_modifiers}" if prompt else "ambient sound effects, SFX, absolute no music"

    num_frames = (duration_sec * 24) + 1

    image = Image.open(media).convert("RGB").resize((width, height))
    
    # Run core inference pipeline
    video, audio = pipe(
        prompt=prompt,
        height=height, width=width, num_frames=num_frames, num_inference_steps=20, seed=seed,
        keyframes=[image], keyframe_indices=[0],
    )
    
    write_video_audio(
        video=video, audio=audio,
        output_path=output, fps=24, audio_sample_rate=32000,
    )
    
    # Clean up memory cleanly
    del pipe
    gc.collect()
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()

i2v = i2v_diffsynth

def GenerateVideo(prompt='', media='', output='output.mp4', 
                  duration_sec=DURATION, width=WIDTH, height=HEIGHT, seed=-1, enhance=True):

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
    duration_sec = 5 # int(estimate_duration(text))
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
    cprompt = f'''subject_definitions:\n<Subject 1> {desc} <Audio 1> is the voice timbre reference for <Subject 1>'s voice, containing a spoken female voiceover. <Subject 1> The person says "{text}." {prompt}.'''

    print("ORIGINAL PROMPT: ",cprompt)

    eprompt = EnhancePrompt(start_image, eprompt, enhance_path)

    print("CURRENT PROMPT: ",eprompt)

    try:
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
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-ref2va-nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
            ],
            processor_config=ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
            vram_limit=64,
        )
        ref_audio, sample_rate = read_audio(audio, duration=5, resample=True, resample_rate=pipe.audio_vae.sample_rate)
        video, audio = pipe(
            prompt=cprompt,
            height=height, width=width, num_frames=124, num_inference_steps=20, seed=seed,
            references=[
                {"type": "image", "image": current_source},
                {"type": "audio", "audio": ref_audio, "sample_rate": sample_rate},
            ],
        )
        write_video_audio(
            video=video, audio=audio,
            output_path=output, fps=24, audio_sample_rate=32000,
        )
            
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

def compose_video(background='',characters=[], voices=[], text=[], action='', output='output.mp4', height=HEIGHT, width=WIDTH, seed=SEED, duration=5):
    references = []
    desc_background = ''
    if background:
        desc_background = AnalyzeImage(image=background, prompt='Describe image background in 10 - 15 words')['analysis']
        references.append({"type": "image", "image": Image.open(background)})
    character_assets = []
    for ndx in range(len(characters)):
        character_assets.append((characters[ndx], AnalyzeImage(image=characters[ndx], prompt='Briefly describe this character sheet of the front and back of the character. 10 - 15 words')['analysis'], text[ndx], voices[ndx]))

    try:
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
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-ref2va-nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
            ],
            processor_config=ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
            vram_limit=64,
        )
        frames = duration * 24
        frames = int(((frames // 17) * 17) + 5) 

        cprompt = []
        cprompt.append(f'''<Picture 1> is {desc_background}. This will be the background for [Shot 1] \n''')
        for ndx in range(len(character_assets)):
            char = character_assets[ndx]
            references.append({"type": "image", "image": Image.open(char[0])})
            ref_audio, sample_rate = read_audio(char[-1], duration=5, resample=True, resample_rate=pipe.audio_vae.sample_rate)
            references.append({"type": "audio", "audio": ref_audio, "sample_rate": sample_rate})
            tmp = f'''subject_definitions:\n<Subject {ndx+1}> Character sheet displaying the front and back of {char[1]} <Audio 1> is the voice timbre reference for <Subject {ndx+1}>'s voice, containing a spoken voiceover. \n'''
            if ndx == 0 and char[2] and len(character_assets) == 2:
                tmp += f'''<Subject 1> says "{text}." to <Subject 2>\n'''
            elif ndx == 1 and char[2] and len(character_assets) == 2:
                tmp += f'''<Subject 2> responds  "{text}." to <Subject 1>\n'''
            elif char[2]:
                tmp += f'''<Subject 1> says "{text}." \n'''
            cprompt.append(tmp)
        cprompt.append(action)
        print("FINAL PROMPT: ",''.join(cprompt))
        video, audio = pipe(
            prompt=''.join(cprompt),
            height=height, width=width, num_frames=frames, num_inference_steps=20, seed=seed,
            references=references,
        )
        write_video_audio(
            video=video, audio=audio,
            output_path=output, fps=24, audio_sample_rate=32000,
        )
            
        description = ''
        # Post-processing
        if os.environ.get('BATCH', 'False') == 'False':
            tmp_img = video_to_img(f'{output}', width, height)
            tmp_img.save('tmp.png')
            description = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")['analysis']
        
        return {
            "status": "success",
            "output_path": output,
            "frames":  frames,
            "description": description,
            "prompt": ''.join(cprompt)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--width', type=int, default=WIDTH)
    parser.add_argument('-H', '--height', type=int, default=HEIGHT)
    parser.add_argument('-E', '--seed', type=int, default=SEED)
    parser.add_argument('-D', '--duration', type=int, default=DURATION)
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-B', '--background', type=str, help='Background path')
    parser.add_argument('-A', '--action', type=str, default='', help='Action to complete')
    parser.add_argument('-C', '--chars', action='append', default=[], help='Character paths (1-2)')
    parser.add_argument('-V', '--voices', action='append', default=[], help='Voice reference paths (1-2)')
    parser.add_argument('-T', '--texts', action='append', default=[], help='Spoken dialog (1-2)')
    args = parser.parse_args()
    print(args)
    print(compose_video(background=args.background,characters=args.chars, voices=args.voices, text=args.texts, action=args.action, output=args.output, height=args.height, width=args.width, seed=args.seed, duration=args.duration))

if __name__ == '__main__':
    main()