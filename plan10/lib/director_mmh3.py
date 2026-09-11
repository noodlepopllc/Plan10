import os
import re
import base64
import mimetypes
from PIL import Image
import json
from fastmcp import Client
import asyncio
from time import sleep
from functools import partial

from plan10.lib.config import load_environ
load_environ()

from plan10.lib.util import load_metadata

ANIME = os.environ.get("ANIME","False") != "False" 
SEED = int(os.environ.get("SEED", "-1"))

from plan10.lib.image_analysis import AnalyzeImage

# portrait_manager.py

import os
from plan10.lib.image_analysis import AnalyzeImage

class PortraitReferenceManager:
    """
    Manages portrait references that attach to existing character subjects.
    Mirrors the structure of audio_refs but for visual identity reinforcement.
    """

    def __init__(self):
        # portrait_refs[label] = { id, path, desc, extra_desc, target }
        self.portrait_refs = {}

    def analyze_portrait(self, image_path: str) -> str:
        """
        Uses your existing VLM analysis pipeline to extract a clean identity description.
        """
        prompt = """Provide a single, concise sentence describing ONLY the character's
        facial features, hair, and identity-defining appearance. Ignore background,
        props, and lighting. Do not include introductory phrases."""
        desc = AnalyzeImage(image_path, prompt)['analysis']
        if desc:
            desc = desc[0].lower() + desc[1:]
        return desc

    def add_portrait_reference(self, builder, image_path: str, label: str,
                               target_subject_label: str, extra_desc: str = "",
                               generator=None):
        """
        Adds a portrait reference linked to an existing subject in the builder.

        builder: SmartVideoPromptBuilder instance
        image_path: path to portrait image
        label: name of this portrait reference
        target_subject_label: name of the character this portrait belongs to
        extra_desc: optional fallback description for generation
        generator: optional function to generate the portrait if missing
        """

        target_key = target_subject_label.lower()
        if target_key not in builder.entities:
            raise ValueError(f"Target subject '{target_subject_label}' not found. Add it first.")

        # Fallback generation if portrait file doesn't exist
        if not os.path.exists(image_path):
            if generator:
                print(f"Generating portrait {label} at {image_path}...")
                char_ref = builder.entities.get(target_subject_label.lower(), None)
                cref_path = char_ref['path'] if char_ref else ''
                generator(extra_desc, cref_path, image_path)
            else:
                print(f"[Warning] Portrait file not found: {image_path}")

        # Analyze portrait identity
        desc = self.analyze_portrait(image_path)

        target_id = builder.entities[target_key]["id"]

        self.portrait_refs[label.lower()] = {
            "id": target_id,               # SAME subject ID
            "path": image_path,
            "pic_tag": f"<Picture {target_id}>",
            "desc": desc,
            "extra_desc": extra_desc,
            "target": target_subject_label
        }

    def emit_prompt_definitions(self):
        """
        Returns a list of prompt lines describing portrait identity references.
        Called inside SmartVideoPromptBuilder.generate().
        """
        lines = []
        for label, data in self.portrait_refs.items():
            extra = f", {data['extra_desc']}" if data['extra_desc'] else ""
            lines.append(
                f"<Subject {data['id']}> face identity is reinforced by "
                f"{data['pic_tag']}, showing {data['desc']}{extra}."
            )
        return lines

    def get_paths(self):
        """
        Returns all portrait image paths for inclusion in image_refs.
        """
        return [data["path"] for data in self.portrait_refs.values()]

class SmartVideoPromptBuilder:
    def __init__(self):
        """
        Initializes the builder with a VLM client for image analysis.
        """
        self.portrait_manager = PortraitReferenceManager()
        
        # Registry to map user labels to generated IDs and descriptions
        self.entities = {}
        self.audio_refs = {}
        self.used_audio_refs = {} 
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
        
        # --- NEW: First frame tracking ---
        self.first_frame_label = None

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
                return getattr(img, 'text', {}).get("SubjectDescription", "")
            else:
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
                metadata = load_metadata(img)
                if hasattr(img, 'text'):
                    for k, v in img.text.items():
                        metadata.add_text(k, v)
                metadata.add_text("SubjectDescription", desc)
                img.save(image_path, pnginfo=metadata)
            else:
                meta_path = image_path.rsplit('.', 1)[0] + '.meta.json'
                with open(meta_path, 'w') as f:
                    json.dump({"SubjectDescription": desc}, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to embed metadata in {image_path}: {e}")
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
        if desc:
            desc = desc[0].lower() + desc[1:]
        return desc

    def add_subject(self, image_path: str, label: str, is_character: bool = False, is_environment: bool = False):
        self._subject_counter += 1
        sub_id = self._subject_counter
        pic_tag = f"<Picture {sub_id}>"
            
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
            "is_environment": is_environment,
            "shots": set()
        }
        return self

    # --- NEW: Dedicated first frame method ---
    def add_firstframe(self, image_path: str, label: str):
        self.add_subject(image_path, label, is_character=False)
        self.first_frame_label = label.lower()
        return self

    def set_summary(self, text: str):
        self.summary = text
        return self

    def add_background(self, image_path: str, label: str):
        return self.add_subject(image_path, label, is_character=False, is_environment=True)

    def add_character(self, image_path: str, label: str):
        return self.add_subject(image_path, label, is_character=True)

    def add_audio_reference(self, audio_path: str, label: str, target_subject_label: str, extra_desc: str = ""):
        target_key = target_subject_label.lower()
        if target_key not in self.entities:
            raise ValueError(f"Target subject '{target_subject_label}' not found. Add it first.")
            
        self.audio_refs[label.lower()] = {
            "id": len(self.audio_refs) + 1,
            "path": audio_path,
            "target_id": self.entities[target_key]["id"],
            "extra_desc": extra_desc
        }
        return self

    def set_scene_style(self, style: str):
        self.scene_style = style
        return self

    def add_shot(self, raw_text: str, duration: float = None, start_time: float = None):
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
        
        self._current_time_ms = current_start_ms + duration_ms
        return self

    def set_soundscape(self, text: str):
        self.soundscape = text
        return self

    def _substitute_labels(self, text: str) -> str:
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
        
        subject_to_speaker = {}
        for audio_data in self.audio_refs.values():
            subject_to_speaker[audio_data['target_id']] = audio_data['speaker_tag']
        
        for label in self.entities.keys():
            entity = self.entities[label]
            tag = f"<Subject {entity['id']}>"
            if tag in processed_text:
                entity["shots"].add(shot_index + 1)
        
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
                if cmd in ('bg', 'char', 'item', 'ff'):
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
                    elif cmd == 'ff':
                        self.add_firstframe(path, label)
                    elif cmd == 'char':
                        self.add_character(path, label)
                    elif cmd == 'item':
                        self.add_subject(path, label, is_character=False)
                        
                elif cmd == 'summary':
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
                            generators['audio'](voice_prompt, path, long=True)
                        else:
                            print(f"[Warning] Audio file not found: {path}")
                    
                    self.add_audio_reference(path, label, target, extra)
                elif cmd == 'portrait':
                    label = parts[1]
                    path = os.path.join(base_dir, parts[2])
                    target = parts[3]
                    extra = parts[4] if len(parts) > 4 else ""

                    generator = generators.get('portrait', None)
                    self.portrait_manager.add_portrait_reference(
                        self, path, label, target, extra_desc=extra, generator=generator)
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
        self.used_audio_refs = {} # Reset for safety
        
        # 1. PRE-SCAN: Find exactly which subjects are speaking
        used_subject_ids = set()
        for shot in self.shots:
            text = self._substitute_labels(shot["raw_text"])
            # CRITICAL FIX: Look for [English], [Spanish], etc. NOT <d> tags
            # because <d> tags haven't been injected yet at this stage
            matches = re.findall(r'<Subject (\d+)>(?=[^<]*\[(?:English|Spanish|French|German|Italian)\])', text)
            used_subject_ids.update(int(m) for m in matches)
        
        # 2. Build speaker tags ONLY for subjects who actually speak AND have an audio ref
        subject_to_speaker = {}
        speaker_counter = 1
        
        for label, data in self.audio_refs.items():
            target_id = data['target_id']
            speaker_tag = None

            if target_id in used_subject_ids:
                speaker_tag = f"(S{speaker_counter})"
                subject_to_speaker[target_id] = speaker_tag
                speaker_counter += 1

            data['speaker_tag'] = speaker_tag
            self.used_audio_refs[label] = data

        # 3. Subject & Audio Definitions
        sections.append("subject_definitions:")
        sub_defs = []
        for label, data in self.entities.items():
            if data.get("is_environment"):
                sub_defs.append(
                    f"<Subject {data['id']}> is the {label} environment in {data['pic_tag']}, featuring {data['desc']}."
                )
            else:
                sub_defs.append(
                    f"<Subject {data['id']}> is {data['desc']} in {data['pic_tag']}."
                )

        audio_defs = []
        for label, data in self.used_audio_refs.items():
            extra = f", {data['extra_desc']}" if data['extra_desc'] else ""
            audio_defs.append(
                f"<Audio {data['id']}> is the voice-timbre reference for <Subject {data['target_id']}> {data['speaker_tag']}{extra}."
            )
        sections.append("\n".join(sub_defs + audio_defs))
        portrait_defs = self.portrait_manager.emit_prompt_definitions()
        sections.append("\n".join(portrait_defs))

        if self.summary:
            sections.append("\nsummary:")
            sections.append(self._substitute_labels(self.summary))
        
        # 4. Process Shots & Build Detailed Description
        sections.append("\ndetailed_description:")
        
        if self.first_frame_label and self.first_frame_label in self.entities:
            pic_tag = self.entities[self.first_frame_label]["pic_tag"]
            ff_rule = f"""{pic_tag} is the first frame of [Shot 1]. The first frame must match {pic_tag} exactly for SPATIAL COMPOSITION: identical pose, head angle, hand position, body orientation, camera angle, and spatial relationships with zero deviation.
However, CHARACTER IDENTITY (facial features, clothing details, body proportions, hair texture) must be corrected and overridden by the character reference images to prevent feature degradation. The character references are the source of truth for identity; the first frame is the source of truth for composition."""
            sections.append(ff_rule)
            sections.append("")
            
        if self.scene_style:
            sections.append(self.scene_style)
            
        for i, shot in enumerate(self.shots):
            processed_text = self._substitute_labels(shot["raw_text"])
            
            # Track which entities appear in this shot for retention analysis
            for label, entity in self.entities.items():
                tag = f"<Subject {entity['id']}>"
                if tag in processed_text:
                    entity["shots"].add(i + 1)
            
            # INJECT SPEAKER TAGS: Replace <Subject X> with <Subject X> (S#) ONLY if it precedes <d>
            # NOTE: At this point, <d> tags still don't exist, so we look for [English] etc.
            def inject_speaker(match):
                subject_tag = match.group(1)
                subject_id = int(re.search(r'\d+', subject_tag).group())
                if subject_id in subject_to_speaker:
                    return f"{subject_tag} {subject_to_speaker[subject_id]}"
                return subject_tag
            
            processed_text = re.sub(r'(<Subject \d+>)(?=[^<]*\[(?:English|Spanish|French|German|Italian)\])', inject_speaker, processed_text)

            # Regex targeting: <Subject 3> speaks [English] "Stay where you are."
            pattern = r'(<Subject \d+>)([^"\[]*?)\[(?:English|Spanish|French|German|Italian)\]\s*("[^"]*")'

            def strip_lang_only(match):
                subject_tag = match.group(1)   # e.g., "<Subject 3>"
                verbs = match.group(2)         # e.g., " speaks " or " says "
                dialogue = match.group(3)      # e.g., '"Stay where you are."'
                
                # Drops the [English] chunk entirely and NEVER inserts <d></d> literal text
                return f"{subject_tag}{verbs}{dialogue}"

            processed_text = re.sub(pattern, strip_lang_only, processed_text)

            '''

            # Wrap dialogue in <d> tags if not already wrapped
            processed_text = re.sub(
                r'(\[(?:English|Spanish|French|German|Italian)\]\s*"[^"]*")',
                r'<d>\1</d>',
                processed_text
            )
            processed_text = re.sub(
                r'(?<!<d>)(\[(?:English|Spanish|French|German|Italian)\][^"<.!?]*[.!?])',
                r'<d>\1</d>',
                processed_text
            )
            '''
            
            time_str = f" At {shot['timestamp']}," if shot.get("timestamp") else ""
            sections.append(f"[Shot {i+1}]{time_str} {processed_text}")
            
        # 5. Auto-Generate Retention Analysis
        sections.append("\nretention_analysis:")
        for label, data in self.entities.items():
            shots_list = sorted(list(data["shots"]))
            shots_str = ", ".join([f"[Shot {s}]" for s in shots_list])
            sections.append(
                f"<Subject {data['id']}> (appears in {shots_str}): fully_preserved - "
                f"{data['desc']} is retained."
            )
            
        # ONLY list audio refs that were actually used
        for label, data in self.used_audio_refs.items():
            sections.append(
                f"<Audio {data['id']}>: reference - its vocal timbre guides the dialogue delivery for <Subject {data['target_id']}>."
            )
            
        # 6. Soundscape & Music
        sections.append("\noverall_soundscape:")
        sections.append(self.soundscape)
        sections.append("\nnon_diegetic_music:")
        sections.append(self.non_diegetic_music)
        
        return "\n".join(sections)

async def send(prompt, images, audio, output='output.mp4', width=768, height=448, duration=5.0, steps=4, start_image=True, upscale=False):
    async with Client("http://localhost:7866/mcp") as client:

        model = "minimax_h3_ref2va_pruned_pdd"

        r = await client.call_tool("wangp_get_default_settings", {"model_type":model})
        results = json.dumps(r.data, indent=4)
        args = r.data
        args['output_filename'] = output
        args['prompt'] = prompt
        args["seed"] = SEED
        if len(audio):
            args["audio_prompt_type"] = "AB" if len(audio) == 2 else "A"
            args["audio_guide"] = audio.pop()
            if len(audio):
                args["audio_guide2"] = audio.pop()
        if start_image:
            args["image_prompt_type"] = "S"
            args["image_start"] = images[0]
            args['image_refs'] = images[1:]
        else:
            args['image_refs'] = images
        if upscale:
            args["spatial_upsampling"] = "ltx25*2"
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

def get_builder(script, output_dir):
    if ANIME:
        from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground
    else:
        from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground
    from plan10.lib.dialog import DesignVoice
    base_dir = f'{os.getcwd()}/{output_dir}'
    generators = {
        'bg': CreateBackground,
        'char': CreateCharacterSheet,
        'ff': GenerateImage,
        'item': GenerateImage,
        'audio': partial(DesignVoice, long=True),
        'portrait': CreatePortrait
    }
    return SmartVideoPromptBuilder().load_script(script, base_dir=base_dir, generators=generators)

def main():
    from pathlib import Path
    import os, argparse, sys
    if ANIME:
        from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    else:
        from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    from plan10.lib.dialog import DesignVoice
    parser = argparse.ArgumentParser(description='Cinematic Director')
    parser.add_argument('-O', '--output', type=str, default='output')
    parser.add_argument('-I', '--input', type=str, default=None)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-W', '--width', type=int, default=int(os.environ.get("WIDTH", "864")))
    parser.add_argument('-H', '--height', type=int, default=int(os.environ.get("HEIGHT", "480")))
    parser.add_argument('-S', '--steps', type=int, default=4)
    parser.add_argument('--wangp', action="store_true")
    args = parser.parse_args()

    # Override environment first
    os.environ['WIDTH'] = str(args.width)
    os.environ['HEIGHT'] = str(args.height)

    # Recompute WIDTH/HEIGHT cleanly
    WIDTH = (args.width // 32) * 32
    HEIGHT = (args.height // 32) * 32

    base_dir = f'{os.getcwd()}/{args.output}'
    Path(f"{base_dir}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{base_dir}/audio").mkdir(parents=True, exist_ok=True)
    
    # Pass generation functions to the parser
    generators = {
        'bg': CreateBackground,
        'char': CreateCharacterSheet,
        'ff': GenerateImage,
        'item': GenerateImage,
        'audio': partial(DesignVoice, long=True),
        'portrait': CreatePortrait
    }

    output_filename = f"{Path(args.input).name.replace('.txt','.mp4')}" if args.input else f"{base_dir}/output.mp4"
    if args.input:
        script = Path(args.input).read_text()
    else:
        # --- THE COMPLETE SELF-CONTAINED SCRIPT ---
        script = """
        # --- ASSETS (with generation prompts) ---
        bg   | barn    | images/rustic_barn.png    | the inside of a rustic barn, with a large open doorway allowing light to spill in
        char | blondie | images/blonde_woman.png   | a medium shot of a blonde woman, white sundress, white tennis shoes
        char | red     | images/red_woman.png      | a medium shot of a red haired woman, blue jeans, tshirt, cowboy boots
        item | dog     | images/samoyed_dog.png    | a samoyed dog
        
        # --- AUDIO REFERENCES (with voice generation prompts) ---
        # audio | label   | path                    | target  | extra_desc                          | voice_prompt
        # audio | voice_b | audio/voice_sample.wav  | blondie | containing a spoken English vocal layer | female
        audio | voice_r | audio/voice_sample2.wav | red     | containing a spoken English vocal layer | female

        portrait | red_port | images/red_portrait.png | red | a red haired woman
        
        # --- SCENE CONTEXT ---
        prompt      | The target video uses a realistic cinematic style with warm golden hour lighting.
        soundscape  | Ambient wind and soft acoustic guitar music.
        
        # --- SHOTS ---
        shot | Sound of roosters, as The camera pushes in on blondie holding a treat and red stands beside her with her arms crossed inside barn environment
        shot | Sounds of a barking dog, Following shot The dog runs and jumps up to grab the treat from blondie inside barn environment | 1
        shot | Sounds of cows mooing, Camera pushes in on red as she speaks [English] "You spoil him." She finishes speaking standing with her mouth closed for a static shot.
        """

    # Build and execute
    builder = SmartVideoPromptBuilder().load_script(script, base_dir=base_dir, generators=generators)
    
    final_prompt = builder.generate()
    print(final_prompt)
    if args.debug:
        sys.exit()
    
    # Extract paths dynamically from the builder instead of hardcoding
    img_refs = (
        [data["path"] for data in builder.entities.values()] + builder.portrait_manager.get_paths())
    aud_refs = [data["path"] for data in builder.used_audio_refs.values()]

    #width and height must be multiples of 32, 1344x768, 864x480 minimal
    if args.input:
        Path(args.input.replace('.txt','_prompt.txt')).write_text(final_prompt)
    
    if args.wangp:
        asyncio.run(send(
            final_prompt, 
            img_refs, 
            aud_refs, 
            output=output_filename, 
            width=args.width, 
            height=args.height, 
            duration=builder.duration,
            steps=args.steps
        ))
    else:
        from plan10.lib.mmh3 import compose_video
        print(compose_video(final_prompt, img_refs, aud_refs, output_filename, args.width, args.height, builder.duration))

if __name__ == '__main__':
    main()