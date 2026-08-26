import os
import re
import base64
import mimetypes
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from fastmcp import Client
import asyncio
from time import sleep

from plan10.lib.config import load_environ
load_environ()

WIDTH = ((int(os.environ.get("WIDTH", "864")) // 32) * 32)
HEIGHT = ((int(os.environ.get("HEIGHT", "480")) // 32) * 32)
ANIME = os.environ.get("ANIME","False") != "False" 
WGP = os.environ.get("WGP","False") != "False"


from plan10.lib.image_analysis import AnalyzeImage

class SmartVideoPromptBuilder:
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Initializes the builder with a VLM client for image analysis.
        """
        self.model = model
        
        # Registry to map user labels to generated IDs and descriptions
        # Format: { "barn": {"id": 1, "type": "Picture", "desc": "...", "shots": []} }
        self.entities = {}
        self.audio_refs = {}
        self.summary = ""
        
        self.shots = []
        self.scene_style = ""
        self.soundscape = "N/A"
        self.non_diegetic_music = "N/A"
        
        self._subject_counter = 0
        self._picture_counter = 0

         # --- NEW: Timeline tracking ---
        self._current_time_ms = 0
        self._default_shot_duration_ms = 2000  # 2.0 seconds default

    @property
    def duration(self):
        """Returns the total accumulated duration of all added shots in seconds."""
        return (self._current_time_ms / 1000.0) + 1.0

    def _format_time(self, ms: int) -> str:
        """Formats milliseconds into MM:SS.mmm"""
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _load_metadata(self, image_path: str) -> str:
        """Checks if the image already has a cached VLM description."""
        try:
            if image_path.lower().endswith('.png'):
                img = Image.open(image_path)
                # Check for our custom PNG text chunk
                return getattr(img, 'text', {}).get("SubjectDescription", "")
            else:
                # Fallback to sidecar JSON for JPEGs
                meta_path = image_path.rsplit('.', 1)[0] + '.meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        return json.load(f).get("SubjectDescription", "")
        except Exception:
            pass
        return ""

    def _save_metadata(self, image_path: str, desc: str):
        """Embeds the VLM description into the image metadata or a sidecar file."""
        try:
            if image_path.lower().endswith('.png'):
                img = Image.open(image_path)
                metadata = PngInfo()
                
                # Preserve any existing metadata (like generation params from ComfyUI/A1111)
                if hasattr(img, 'text'):
                    for k, v in img.text.items():
                        metadata.add_text(k, v)
                        
                metadata.add_text("SubjectDescription", desc)
                img.save(image_path, pnginfo=metadata)
            else:
                # Sidecar JSON for non-PNGs
                meta_path = image_path.rsplit('.', 1)[0] + '.meta.json'
                with open(meta_path, 'w') as f:
                    json.dump({"SubjectDescription": desc}, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to embed metadata in {image_path}: {e}")
            # Ultimate fallback: write a plain text sidecar
            with open(image_path + ".desc.txt", "w") as f:
                f.write(desc)

    def _analyze_image(self, image_path: str, is_character: bool = False) -> str:
        if is_character:
            prompt = """Provide a single, concise sentence describing ONLY the character's physical appearance, 
            clothing, and distinguishing features. Ignore the background, setting, props, and other people. 
            Focus on: face, hair, body type, clothing, accessories. Do not include introductory phrases."""
        else:
            prompt = """Provide a single, concise sentence describing the main visual elements, lighting, 
            atmosphere, and key objects in this environment/scene. Do not include introductory phrases 
            like 'This image shows' or 'The image features'."""
        
        desc = AnalyzeImage(image_path, prompt)['analysis']
        
        # Lowercase the first letter
        if desc:
            desc = desc[0].lower() + desc[1:]
        return desc

    def add_subject(self, image_path: str, label: str, is_character: bool = False):
        self._subject_counter += 1
        sub_id = self._subject_counter
        
        pic_tag = f"<Picture {sub_id}>" if is_character else f"<Picture {sub_id}>"
            
        # Check cache first (include is_character in cache key!)
        cache_key = f"{image_path}_{'char' if is_character else 'bg'}"
        desc = self._load_metadata(cache_key)
        
        if not desc:
            print(f"Analyzing {image_path} as {'character' if is_character else 'background'}...")
            desc = self._analyze_image(image_path, is_character=is_character)
            self._save_metadata(cache_key, desc)
        else:
            print(f"Loaded cached description for {image_path}.")
        
        self.entities[label.lower()] = {
            "id": sub_id,
            "path": image_path,
            "pic_tag": pic_tag,
            "desc": desc,
            "is_character": is_character,
            "shots": set()
        }
        return self

    def set_summary(self, text: str):
        self.summary = text
        return self

    # Alias for backgrounds to match your mental model
    def add_background(self, image_path: str, label: str):
        return self.add_subject(image_path, label, is_character=False)

    def add_character(self, image_path: str, label: str):
        return self.add_subject(image_path, label, is_character=True)

    def add_audio_reference(self, audio_path: str, label: str, target_subject_label: str, extra_desc: str = ""):
        """Registers an audio file as a voice timbre reference for a specific subject."""
        target_key = target_subject_label.lower()
        if target_key not in self.entities:
            raise ValueError(f"Target subject '{target_subject_label}' not found. Add it first.")
            
        self.audio_refs[label.lower()] = {
            "id": len(self.audio_refs) + 1,
            "path": audio_path,    # <-- ADDED: Track the file path
            "target_id": self.entities[target_key]["id"],
            "extra_desc": extra_desc
        }
        return self

    def set_scene_style(self, style: str):
        self.scene_style = style
        return self

    def add_shot(self, raw_text: str, duration: float = None, start_time: float = None):
        """
        Adds a shot with automatic timestamp incrementing.
        :param raw_text: The text description of the shot.
        :param duration: Duration of this shot in seconds. Defaults to 2.0s if not provided.
        :param start_time: Optional explicit start time in seconds. Overrides auto-increment.
        """
        duration_ms = int(duration * 1000) if duration is not None else self._default_shot_duration_ms
        
        if start_time is not None:
            current_start_ms = int(start_time * 1000)
        else:
            current_start_ms = self._current_time_ms
            
        timestamp_str = self._format_time(current_start_ms)
        
        self.shots.append({
            "raw_text": raw_text, 
            "timestamp": timestamp_str,
            "duration_ms": duration_ms
        })
        
        # Advance the timeline for the next shot
        self._current_time_ms = current_start_ms + duration_ms
        return self

    def set_soundscape(self, text: str):
        self.soundscape = text
        return self

    def _substitute_labels(self, text: str) -> str:
        """Replaces all subject labels with their <Subject N> tags."""
        processed_text = text
        sorted_labels = sorted(self.entities.keys(), key=len, reverse=True)
        
        for label in sorted_labels:
            entity = self.entities[label]
            tag = f"<Subject {entity['id']}>"
            pattern = r'\b' + re.escape(label) + r'\b'
            processed_text = re.sub(pattern, tag, processed_text, flags=re.IGNORECASE)
        
        return processed_text

    def _inject_tags(self, text: str, shot_index: int) -> str:
        processed_text = self._substitute_labels(text)
        
        # Build a lookup: subject_id -> speaker_tag (if they have an audio ref)
        subject_to_speaker = {}
        for audio_data in self.audio_refs.values():
            subject_to_speaker[audio_data['target_id']] = audio_data['speaker_tag']
        
        # Track which subjects appear in this shot
        for label in self.entities.keys():
            entity = self.entities[label]
            tag = f"<Subject {entity['id']}>"
            if tag in processed_text:
                entity["shots"].add(shot_index + 1)
        
        # Inject speaker tags for any subject with a voice reference
        for subject_id, speaker_tag in subject_to_speaker.items():
            subject_tag = f"<Subject {subject_id}>"
            tagged_subject = f"<Subject {subject_id}> {speaker_tag}"
            processed_text = processed_text.replace(subject_tag, tagged_subject)
        
        return processed_text

    def load_script(self, script_text: str, base_dir: str = "", generators: dict = None):
        if generators is None:
            generators = {}
            
        for line in script_text.strip().split('\n'):
            line = line.split('#')[0].strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            cmd = parts[0].lower()
            
            try:
                if cmd in ('bg', 'char', 'item'):
                    label = parts[1]
                    path = os.path.join(base_dir, parts[2])
                    prompt = parts[3] if len(parts) > 3 else ""
                    
                    if not os.path.exists(path):
                        if cmd in generators:
                            print(f"Generating {label} at {path}...")
                            generators[cmd](prompt, path)
                        else:
                            print(f"[Warning] File not found: {path}")
                    
                    if cmd == 'bg':
                        self.add_background(path, label)
                    elif cmd == 'char':
                        self.add_character(path, label)
                    elif cmd == 'item':
                        self.add_subject(path, label, is_character=False)
                        
                elif cmd == 'summary':
                    # FIXED: Moved to its own branch
                    self.set_summary(parts[1] if len(parts) > 1 else "")
                        
                elif cmd == 'audio':
                    label = parts[1]
                    path = os.path.join(base_dir, parts[2])
                    target = parts[3] if len(parts) > 3 else ""
                    extra = parts[4] if len(parts) > 4 else ""
                    voice_prompt = parts[5] if len(parts) > 5 else "female"
                    
                    if not os.path.exists(path):
                        if 'audio' in generators:
                            print(f"Generating voice {label} at {path}...")
                            generators['audio'](voice_prompt, path, long=False)
                        else:
                            print(f"[Warning] Audio file not found: {path}")
                    
                    self.add_audio_reference(path, label, target, extra)
                    
                elif cmd == 'prompt':
                    self.set_scene_style(parts[1] if len(parts) > 1 else "")
                elif cmd == 'soundscape':
                    self.set_soundscape(parts[1] if len(parts) > 1 else "")
                elif cmd == 'shot':
                    duration = float(parts[2]) if len(parts) > 2 and parts[2] else None
                    self.add_shot(parts[1], duration=duration)
                    
            except IndexError:
                print(f"[Warning] Malformed line skipped: {line}")
                
        return self

    def generate(self) -> str:
        """Compiles everything into the final structured prompt format."""
        sections = []
        
        # 1. Subject & Audio Definitions
        sections.append("subject_definitions:")
        sub_defs = []
        for label, data in self.entities.items():
            sub_defs.append(
                f"<Subject {data['id']}> is {data['desc']} in {data['pic_tag']}."
            )
        
        audio_defs = []
        speaker_counter = 0
        for label, data in self.audio_refs.items():
            speaker_counter += 1
            speaker_tag = f"(S{speaker_counter})"
            data['speaker_tag'] = speaker_tag  # Store for shot injection
            
            extra = f", {data['extra_desc']}" if data['extra_desc'] else ""
            audio_defs.append(
                f"<Audio {data['id']}> is the voice-timbre reference for <Subject {data['target_id']}> {speaker_tag}{extra}."
            )
        sections.append("\n".join(sub_defs + audio_defs))
        
        if self.summary:
            sections.append("\nsummary:")
            sections.append(self._substitute_labels(self.summary))
        
        # 2. Process Shots & Build Detailed Description
        sections.append("\ndetailed_description:")
        if self.scene_style:
            sections.append(self.scene_style)
            
        for i, shot in enumerate(self.shots):
            processed_text = self._inject_tags(shot["raw_text"], i)
            
            # Wrap dialogue in <d> tags
            # Primary: quoted dialogue [Language] "text"
            processed_text = re.sub(
                r'(\[(?:English|Spanish|French|German|Italian)\]\s*"[^"]*")',
                r'<d>\1</d>',
                processed_text
            )
            # Fallback: unquoted dialogue up to punctuation (only if not already wrapped)
            processed_text = re.sub(
                r'(?<!<d>)(\[(?:English|Spanish|French|German|Italian)\][^"<.!?]*[.!?])',
                r'<d>\1</d>',
                processed_text
            )
            time_str = f" At {shot['timestamp']}," if shot.get("timestamp") else ""
            sections.append(f"[Shot {i+1}]{time_str} {processed_text}")
            
        # 3. Auto-Generate Retention Analysis
        sections.append("\nretention_analysis:")
        for label, data in self.entities.items():
            shots_list = sorted(list(data["shots"]))
            shots_str = ", ".join([f"[Shot {s}]" for s in shots_list])
            sections.append(
                f"<Subject {data['id']}> (appears in {shots_str}): fully_preserved - "
                f"{data['desc']} is retained."
            )
        for label, data in self.audio_refs.items():
            sections.append(
                f"<Audio {data['id']}>: reference - its vocal timbre guides the dialogue delivery."
            )
            
        # 4. Soundscape & Music
        sections.append("\noverall_soundscape:")
        sections.append(self.soundscape)
        sections.append("\nnon_diegetic_music:")
        sections.append(self.non_diegetic_music)
        
        return "\n".join(sections)

async def send(prompt, images, audio, output='output.mp4', width=768, height=448, duration=5.0):
    async with Client("http://localhost:7866/mcp") as client:

        model = "minimax_h3_ref2va_pruned"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":model})
        results = json.dumps(r.data, indent=4)
        args = r.data
        args["activated_loras"] = ["minimax_h3_larryvrh_v4_step600_ema.safetensors"]
        args["loras_multipliers"] = "1.0|"
        args['output_filename'] = output
        args['prompt'] = prompt
        args['image_refs'] = images
        if len(audio):
            args["audio_prompt_type"] = "AB" if len(audio) == 2 else "A"
            args["audio_guide"] = audio.pop()
            if len(audio):
                args["audio_guide2"] = audio.pop()
        args["video_prompt_type"] = "I"
        args["multi_prompts_gen_type"] = "FG"
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
        args['video_length'] = (((duration * 24) // 17) * 17) + 5
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

def main():
    from pathlib import Path
    import os, argparse, sys
    if ANIME:
        from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground
    else:
        from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground
    from plan10.lib.dialog import DesignVoice
    parser = argparse.ArgumentParser(description='Cinematic Director')
    parser.add_argument('-O', '--output', type=str, default='output', help='output directory')
    parser.add_argument('-I', '--input', type=str, default=None, help='input file')
    parser.add_argument('-D', '--debug', action='store_true')
    args = parser.parse_args()

    base_dir = f'{os.getcwd()}/{args.output}'
    Path(f"{base_dir}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{base_dir}/audio").mkdir(parents=True, exist_ok=True)
    
    # Pass generation functions to the parser
    generators = {
        'bg': CreateBackground,
        'char': CreateCharacterSheet,
        'item': GenerateImage,
        'audio': DesignVoice
    }

    output_filename = f"{Path(args.input).name.replace('.txt','.mp4')}" if args.input else f"{base_dir}/output.mp4"
    if args.input:
        script = Path(args.input).read_text()
    else:
        # --- THE COMPLETE SELF-CONTAINED SCRIPT ---
        script = """
        # --- ASSETS (with generation prompts) ---
        bg   | barn    | images/rustic_barn.png    | the inside of a rustic barn
        char | blondie | images/blonde_woman.png   | a medium shot of a blonde woman, white sundress, white tennis shoes
        char | red     | images/red_woman.png      | a medium shot of a red haired woman, blue jeans, tshirt, cowboy boots
        char | dog     | images/samoyed_dog.png    | a samoyed dog
        
        # --- AUDIO REFERENCES (with voice generation prompts) ---
        # audio | label   | path                    | target  | extra_desc                          | voice_prompt
        audio | voice_b | audio/voice_sample.wav  | blondie | containing a spoken English vocal layer | female
        audio | voice_r | audio/voice_sample2.wav | red     | containing a spoken English vocal layer | female
        
        # --- SCENE CONTEXT ---
        prompt      | The target video uses a realistic cinematic style with warm golden hour lighting.
        soundscape  | Ambient wind and soft acoustic guitar music.
        
        # --- SHOTS ---
        shot | The camera pushes on the inside of the barn. blondie is standing near the door holding a treat. | 3.0
        shot | The dog runs into the barn and jumps up to grab the treat from blondie.
        shot | red walks over to blondie and speaks, [English] You spoil him. | 2.5
        shot | Closeup shot of blondie | 1.5
        shot | blondie speaks to dog [English] You are such a good boy! | 3.0
        """

    # Build and execute
    builder = SmartVideoPromptBuilder().load_script(script, base_dir=base_dir, generators=generators)
    
    final_prompt = builder.generate()
    print(final_prompt)
    if args.debug:
        sys.exit()
    
    # Extract paths dynamically from the builder instead of hardcoding
    img_refs = [data["path"] for data in builder.entities.values()]
    aud_refs = [data["path"] for data in builder.audio_refs.values()]

    #width and height must be multiples of 32, 1344x768, 864x480 minimal
    if args.input:
        Path(args.input.replace('.txt','_prompt.txt')).write_text(final_prompt)
    
    if WGP:
        asyncio.run(send(
            final_prompt, 
            img_refs, 
            aud_refs, 
            output=output_filename, 
            width=WIDTH, 
            height=HEIGHT, 
            duration=builder.duration
        ))
    else:
        from plan10.lib.mmh3 import compose_video
        print(compose_video(final_prompt, img_refs, aud_refs, output_filename, WIDTH, HEIGHT, builder.duration))

if __name__ == '__main__':
    main()