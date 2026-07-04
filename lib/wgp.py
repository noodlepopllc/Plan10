from config import load_environ
load_environ()

from PIL import Image

import asyncio, logging, os, random, json
from fastmcp import Client
from time import sleep

from pathlib import Path

from util import video_to_img
from image_analysis import AnalyzeImage, EnhancePrompt

from image_gen import add_metadata_char

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
ANIME = "_anime" if os.environ.get("ANIME","False") != "False" else ""

enhance_path = f'./system/ltx_enhancer{ANIME}.txt'

#tool = "ltx2_22B_1_1"
tool = "ltx2_22B_distilled_1_1"

async def i2v(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    async with Client("http://localhost:7866/mcp") as client:

        r = await client.call_tool("wangp_get_default_settings", {"model_type":tool})
        results = json.dumps(r.data, indent=4)

        # Force explicit SFX and ban melody structure in the positive prompt
        sfx_modifiers = ", realistic sound effects only, crisp SFX, ambient background noise, completely devoid of music, no BGM, no instruments"
        final_prompt = f"{prompt}{sfx_modifiers}" if prompt else "ambient sound effects, SFX, absolute no music"

        # Purged "silent or muted audio" to allow empty spaces, heavily punished music architecture
        negative_prompt = (
            "text, subtitles, lyrics, captions, on-screen text, logo, " # Text
            "music, song, soundtrack, singing, talking, speech, voice, "
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

        args = r.data
        args['output_filename'] = output
        args['prompt'] = final_prompt
        args['image_prompt_type'] =  'S'
        args['image_start'] = media
        #args['guidance_scale'] = 2.0
        #args['sample_solver'] = "euler"
        args['negative_prompt'] = negative_prompt
        #args['num_inference_steps'] = 30
        #args['prompt_enhancer'] = 'TI'
        #args['audio_source'] = None
        #args['audio_prompt_type'] = 'A'

        args['resolution'] = '720x1280' if height > width else '1280x720'
        args['video_length'] = (duration_sec * 24) + 1 
        print(args)
        r = await client.call_tool("wangp_generate", {"source": args})
        print(r.data['job_id'])
        job_id = r.data['job_id']

        r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        last = ''
        while not r.data['done']:
            sleep(5)
            this = '' 
            for event in r.data['events']:
                if 'text' in event['data']:
                    if '%|' in event['data']['text']:
                        this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

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
            start_image = f'{os.getcwd()}/{media}'

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
            asyncio.run(i2v(eprompt, start_image, Path(output).name, 
                    duration_sec, width, height, seed))
            description = ''
                
            # Post-processing
            if os.environ.get('BATCH', 'False') == 'False':
                tmp_img = video_to_img(f'{os.getcwd()}/{output}', width, height)
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

async def s2v(prompt='', media='', audio='', text='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    async with Client("http://localhost:7866/mcp") as client:

        model = "ltx2_22B_distilled_1_1"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":model})
        results = json.dumps(r.data, indent=4)

        desc = Image.open(media).info.get('Description')
        if not desc:
            desc = add_metadata_char(media, '', seed)

        newprompt = f"[VISUAL]: {desc}.\n[SPEECH]: {text}.\n[SOUNDS]: {prompt}."
        print(newprompt)

        args = r.data
        args['output_filename'] = output
        args['prompt'] = newprompt
        args['image_prompt_type'] =  'S'
        args['image_start'] = media
        args['prompt_enhancer'] = 'TI'
        args['audio_prompt_type'] = 'A1OF'
        args['audio_guide'] = audio
        args['activated_loras'] = ["id-lora-celebvhq-ltx2.3.safetensors"]

        #args['audio_source'] = None
        #args['audio_prompt_type'] = 'A'

        args['resolution'] = '720x1280' if height > width else '1280x720'
        args['video_length'] = (duration_sec * 24) + 1 
        print(args)
        r = await client.call_tool("wangp_generate", {"source": args})
        print(r.data['job_id'])
        job_id = r.data['job_id']

        r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        last = ''
        while not r.data['done']:
            sleep(5)
            this = '' 
            for event in r.data['events']:
                if 'text' in event['data']:
                    if '%|' in event['data']['text']:
                        this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

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
    duration = (total_syllables * 0.37) + 1.0
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
    duration_sec = int(estimate_duration(text)) + 1
    fps = 24

    if seed == -1:
        seed = random.randint(0,1000000)

    total_frames = (duration_sec * fps) + 1

    print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
    print(f"   Resolution: {width}x{height}")

    current_source = video_to_img(start_image, width, height, True, True)
    current_source.save('tmp.png')

    if prompt: 
        desc = Image.open(start_image).info.get('Description')
        if not desc:
            desc = add_metadata_char(start_image, '', seed)

        eprompt = f'The person can be described as {desc}. The person says "{text}." {prompt}.'
        return GenerateVideo(eprompt, start_image, output, duration_sec, width, height, seed)

    if not prompt:
        prompt = "The characters stand and act naturally. "

    eprompt = prompt 

    print("CURRENT PROMPT: ",eprompt)

    try:
        asyncio.run(s2v(eprompt, start_image, ref_audio, text, Path(output).name, 
                duration_sec, width, height, seed))
        description = ''
            
        # Post-processing
        if os.environ.get('BATCH', 'False') == 'False':
            tmp_img = video_to_img(f'{os.getcwd()}/{output}', width, height)
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


def GenerateTalkingVideo_old(
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
    duration_sec = int(estimate_duration(text)) + 1
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

    eprompt = prompt 

    print("CURRENT PROMPT: ",eprompt)

    try:
        asyncio.run(s2v(eprompt, start_image, ref_audio, text, Path(output).name, 
                duration_sec, width, height, seed))
        description = ''
            
        # Post-processing
        if os.environ.get('BATCH', 'False') == 'False':
            tmp_img = video_to_img(f'{os.getcwd()}/{output}', width, height)
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
