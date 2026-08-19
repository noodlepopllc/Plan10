from plan10.lib.config import load_environ
load_environ()

from PIL import Image

import asyncio, logging, os, random, json, math
from fastmcp import Client
from time import sleep
import librosa

from pathlib import Path

from plan10.lib.util import video_to_img, fix_minimax_audio
from plan10.lib.image_analysis import AnalyzeImage, EnhancePrompt, translate_to_audio_prompt

from plan10.lib.image_gen import add_metadata_char

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "-1"))
ANIME = "_anime" if os.environ.get("ANIME","False") != "False" else ""
ARC = os.environ.get("LTX","False") == "ARC"
MMH3 = os.environ.get('MMH3','False') != 'False'

MAX_DURATION = 5 if MMH3 else 10

if ANIME:
    from plan10.lib.anime_gen import GenerateImage
else:
    from plan10.lib.image_gen import GenerateImage

BRIEF = os.environ.get("BRIEF","False") != "False"

enhance_path = f'./system/ltx_enhancer_minimal{ANIME}.txt' if BRIEF else f'./system/ltx_enhancer{ANIME}.txt'

if MMH3:
    enhance_path = './system/mmh3_enhancer.txt'

#tool = "ltx2_22B_1_1"
tool = "ltx2_22B_distilled_1_1"

async def i2v_ltx(prompt='', media='', end='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    async with Client("http://localhost:7866/mcp") as client:

        r = await client.call_tool("wangp_get_default_settings", {"model_type":tool})
        results = json.dumps(r.data, indent=4)

        desc = AnalyzeImage(media, "Briefly describe this image, background and character, no more than 50 words")['analysis']
        audio_desc = translate_to_audio_prompt(desc)

        # Force explicit SFX and ban melody structure in the positive prompt
        sfx_modifiers = ", realistic sound effects only, crisp SFX, ambient background noise, completely devoid of music, no BGM, no instruments"
        final_prompt = f"{prompt} {audio_desc} {sfx_modifiers}" if prompt else "ambient sound effects, SFX, absolute no music"

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
        args['image_prompt_type'] =  'SE' if end else 'S'
        args['image_start'] = media
        if end:
            args['image_end'] = end

        args['resolution'] = f'{width}x{height}'
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
            if r.data.get('events',[]):
                for event in r.data['events']:
                    if event and event.get('data') and 'text' in event.get('data',''):
                        if '%|' in event['data']['text']:
                            this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

async def i2v_h3(prompt='', media='', end='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    async with Client("http://localhost:7866/mcp") as client:

        tool = "minimax_h3_fl2va_pruned"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":tool})
        results = json.dumps(r.data, indent=4)


        desc = AnalyzeImage(media, "Briefly describe this image, background and character, no more than 50 words")['analysis']
        audio_desc = translate_to_audio_prompt(desc)

        # Force explicit SFX and ban melody structure in the positive prompt
        sfx_modifiers = ", realistic sound effects only, crisp SFX, ambient background noise, completely devoid of music, no BGM, no instruments"
        final_prompt = f"{prompt} {audio_desc}" # {sfx_modifiers}" if prompt else "ambient sound effects, SFX, absolute no music"

        frames = (((duration_sec * 24) // 17) * 17) + 5
        frames = 107 if frames < 107 else frames


        args = r.data
        args['output_filename'] = output
        args['prompt'] = final_prompt
        args['image_prompt_type'] =  'SE' if end else 'S'
        args['image_start'] = media
        args['resolution'] = f'{width}x{height}'
        args['video_length'] = frames
        args["activated_loras"] = ["minimax_h3_larryvrh_v4_step600_ema.safetensors"]
        args["loras_multipliers"] = "1.0|"
        args["guidance_scale"] = 1
        args["num_inference_steps"] = 4
        if end:
            args['image_end'] = end
        print(args)
        r = await client.call_tool("wangp_generate", {"source": args})
        print(r.data['job_id'])
        job_id = r.data['job_id']

        r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        last = ''
        while not r.data['done']:
            sleep(5)
            this = '' 
            if r.data.get('events',[]):
                for event in r.data['events']:
                    if event and event.get('data') and 'text' in event.get('data',''):
                        if '%|' in event['data']['text']:
                            this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

i2v = i2v_h3 if MMH3 else i2v_ltx

def GenerateVideo(prompt='', media='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1, enhance=True):

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
        current_source.save(f'{os.getcwd()}/tmp.png')

        last = None
        if end_image:
            last = f'{os.getcwd()}/tmp_end.png'
            end_image.save(last)

        if not prompt:
            prompt = "The characters stand and act naturally. "

        if enhance:
            eprompt = EnhancePrompt('tmp.png', prompt, enhance_path)
        else:
            eprompt = prompt

        print("CURRENT PROMPT: ",eprompt)

        try:
            asyncio.run(i2v(eprompt, f'{os.getcwd()}/tmp.png', last, Path(output).name, 
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

async def s2v_ltx(prompt='', media='', end_image='', audio='', text='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    async with Client("http://localhost:7866/mcp") as client:

        model = "ltx2_22B_distilled_1_1"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":model})
        results = json.dumps(r.data, indent=4)

        '''
        desc = Image.open(media).info.get('Description')
        if not desc:
            desc = add_metadata_char(media, '', seed)
        '''

        desc = AnalyzeImage(media, "Briefly describe this image, background and character, no more than 50 words")['analysis']
        audio_desc = translate_to_audio_prompt(desc)

        newprompt = f"[VISUAL]: {desc} {prompt} Lips moving in perfect sync with the audio. \n[SPEECH]: {text}.\n[SOUNDS]: {audio_desc}."
        print(newprompt)

        args = r.data
        args['output_filename'] = output
        args['prompt'] = newprompt
        args['image_prompt_type'] =  'S'
        args['image_start'] = media
        #args['prompt_enhancer'] = 'TI'
        args['audio_prompt_type'] = 'A1OF'
        args['audio_guide'] = audio
        args['activated_loras'] = ["id-lora-celebvhq-ltx2.3.safetensors"]

        #args['audio_source'] = None
        #args['audio_prompt_type'] = 'A'

        args['resolution'] = f'{width}x{height}' #'720x1280' if height > width else '1280x720'
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
                if event['data'] and 'text' in event['data']:
                    if '%|' in event['data']['text']:
                        this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

async def s2v_h3(prompt='', media='', end_image='', audio='', text='', output='output.mp4', 
                  duration_sec=5, width=WIDTH, height=HEIGHT, seed=-1):
    transcript = ''
    cam_desc = ''
    if not text:
        from plan10.lib.dialog import transcribe
        y, sr = librosa.load(audio, sr=None)
        duration_sec = int(math.ceil(librosa.get_duration(y=y, sr=sr)) + 1)
        fixed_audio = audio
        segs = transcribe(audio)
        transcript = " ".join(segs)
    else:
        fixed_audio = audio 
        cam_desc = AnalyzeImage(media, "Briefly describe camera shot framing, return either closeup shot or medium shot")['analysis']

    desc = AnalyzeImage(media, "Briefly describe this image, background and character, no more than 50 words")['analysis']
    audio_desc = translate_to_audio_prompt(desc)

    lipsync = ("subject_definitions:\n <Subject 1> is the person in <Picture 1> and appears in [Shot 1], preserving their exact identity, facial features, skin tone, hairstyle, body proportions, clothing, footwear, "
"and distinctive accessories.\n <Audio 1> provides the voice timbre, delivery, and lip-sync mapping. summary:\n"
f"spoken_text: \nThe narration spoken in <Audio 1> is: \"{transcript}\""
f''' <Picture 1> is the first frame of [Shot 1] static. The first frame of the video must match <Picture 1> exactly, including identical pose, head angle, hand position, body orientation, facial expression, and clothing folds, with zero deviation. \n'''
#f'''{cam_desc} Camera focuses on <Subject 1> as they speak with initial framing, keeping them clearly in frame.  '''
f''' The character faces the camera and speaks, with precise lip movements, jaw adjustments, and subtle facial micro-expressions perfectly synchronized to the cadence and dialogue of <Audio 1>.  \n'''
f''' After speaking, <Subject 1> {prompt} '''
f'''{"They continue to move naturally for the remainder of the video transitioning into <Picture 2> is the last frame of [Shot 1] static. After transitioning fully into <Picture 2>, the subject holds completely still with no additional motion for the remainder of the clip." if end_image else ''} \n'''
f''' overall_soundscape: {audio_desc} ''') 
    

    newprompt = ("subject_definitions:\n <Subject 1> is the person in <Picture 1> and appears in [Shot 1], preserving their exact identity, facial features, skin tone, hairstyle, body proportions, clothing, footwear, "
"and distinctive accessories.\n <Audio 1> is the voice timbre reference for <Subject 1>'s voice, containing a spoken voiceover. summary:\n"
f''' <Picture 1> is the first frame of [Shot 1] static {cam_desc} Camera focuses on <Subject 1> as they speak, keeping them clearly in frame. <Subject 1> remains stationary as they speak (S1) clearly <d>[English] {text} </d> \n'''
f''' After speaking, <Subject 1> {prompt} They continue to move naturally for the remainder of the video. \n overall_soundscape: {audio_desc} ''') 
    async with Client("http://localhost:7866/mcp") as client:

        model = "minimax_h3_ref2va_pruned"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":model})
        results = json.dumps(r.data, indent=4)
        print(newprompt if text else lipsync)

        args = r.data
        args["activated_loras"] = ["minimax_h3_larryvrh_v4_step600_ema.safetensors"]
        args["loras_multipliers"] = "1.0|"
        args['output_filename'] = output
        args['prompt'] = newprompt if text else lipsync
        args['image_refs'] = [media, end_image] if end_image else [media]
        args["audio_guide"] = fixed_audio
        args["audio_prompt_type"] = "A"
        args["video_prompt_type"] = "I"
        args["multi_prompts_gen_type"] = "PG"
        args["num_inference_steps"] = 8
        args["guidance_scale"] = 1
        args["guidance2_scale"] = 5
        args["guidance3_scale"] = 5
        args["model_switch_phase"] = 1
        args["alt_guidance_scale"] = 1
        args["audio_guidance_scale"] = 1
        args["audio_scale"] = 1
        args["sample_solver"] = "euler"
        args["embedded_guidance_scale"] = 1.5
        args['resolution'] = f'{width}x{height}'
        args['video_length'] = (((duration_sec * 24) // 17) * 17) + 5

        args['resolution'] = f'{width}x{height}'
        print(args)
        r = await client.call_tool("wangp_generate", {"source": args})
        print(r.data['job_id'])
        job_id = r.data['job_id']

        r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        last = ''
        while r.data and not r.data['done']:
            sleep(5)
            this = '' 
            if 'events' not in r.data:
                continue
            for event in r.data['events']:
                if event['data'] and 'text' in event['data']:
                    if '%|' in event['data']['text']:
                        this = event['data']['text']
            if this != last:
                last = this
                print(this)
            r = await client.call_tool("wangp_get_job", {"job_id": job_id})
        print(r.data['result'])

s2v = s2v_h3 if MMH3 else s2v_ltx

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
    duration = (total_syllables * 0.37)
    return math.ceil(duration)

def GenerateTalkingVideo(
    prompt='',
    text='',
    audio='',
    media='',
    output='output.mp4',
    width=WIDTH,
    height=HEIGHT,
    seed=-1,
    max_duration=MAX_DURATION):
    print(f"PROMPT: {prompt}")
    
    if isinstance(prompt, list):
        prompt = prompt.pop()
    
    start_image = ''
    end_image = None

    if not media:
        GenerateImage(prompt = prompt, output='first_frame.png', width=width, height=height, seed=seed)
        media='first_frame.png'

    if isinstance(media, list):
        start_image = f'{os.getcwd()}/{media.pop(0)}'
        if len(media) > 0:
            end_image = f'{os.getcwd()}/{media.pop(0)}'
    else:
        start_image = f'{os.getcwd()}/{media}'

    ref_audio = f'{os.getcwd()}/{audio}'

    print(f"MEDIA: {start_image}")

    original_prompt = prompt

    width = int(width)
    height = int(height)
    seed = int(seed)
    estimated =  int(estimate_duration(text)) + 1
    print(f"ESTIMATED DURATION: {estimated} s")
    duration_sec = 5 if estimated < 5 else max_duration
    fps = 24

    if seed == -1:
        seed = random.randint(0,1000000)

    total_frames = (duration_sec * fps) + 1

    print(f"\n🎬 Generating {total_frames/fps:.1f}s video ({total_frames} frames)")
    print(f"   Resolution: {width}x{height}")

    current_source = video_to_img(start_image, width, height, True, True)
    current_source.save('tmp.png')
    current_source_path = f'{os.getcwd()}/tmp.png'

    if ARC and prompt: 
        desc = Image.open(start_image).info.get('Description')
        if not desc:
            desc = add_metadata_char(start_image, '', seed)

        eprompt = f'The person can be described as {desc}. The person says "{text}." {prompt}.'
        return GenerateVideo(eprompt, media, output, duration_sec, width, height, seed)

    if not prompt:
        prompt = "The characters stand and act naturally. "

    eprompt = prompt 

    print("CURRENT PROMPT: ",eprompt)

    try:
        asyncio.run(s2v(eprompt, current_source_path, end_image, ref_audio, text, Path(output).name, 
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
