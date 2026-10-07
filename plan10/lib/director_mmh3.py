import os
import re
import base64
import mimetypes
import requests
from PIL import Image
import json
from time import sleep
from functools import partial
from pathlib import Path

from plan10.lib.config import load_environ
load_environ()

from plan10.lib.util import load_metadata

ANIME = os.environ.get("ANIME", "False") != "False" 
SEED = int(os.environ.get("SEED", "-1"))

from plan10.lib.image_analysis import AnalyzeImage

def director_load_metadata(image_path: str) -> str:
    """Checks if the image already has a cached VLM description."""
    try:
        if image_path.lower().endswith('.png'):
            img = Image.open(image_path)
            return getattr(img, 'info', {}).get("SubjectDescription", "")
        else:
            meta_path = image_path.rsplit('.', 1)[0] + '.meta.json'
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get("SubjectDescription", "")
    except Exception:
        pass
    return ""

def director_save_metadata(image_path: str, desc: str):
    """Embeds the VLM description into the image metadata or a sidecar file."""
    try:
        if image_path.lower().endswith('.png'):
            with Image.open(image_path) as img:
                metadata = load_metadata(img)
                if hasattr(img, 'text'):
                    for k, v in img.text.items():
                        metadata.add_text(k, v)
                metadata.add_text("SubjectDescription", desc)
                img.save(image_path, pnginfo=metadata)
        else:
            meta_path = image_path.rsplit('.', 1)[0] + '.meta.json'
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({"SubjectDescription": desc}, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to embed metadata in {image_path}: {e}")
        with open(image_path + ".desc.txt", "w", encoding='utf-8') as f:
            f.write(desc)

class PortraitReferenceManager:
    def __init__(self):
        self.portrait_refs = {}

    def analyze_portrait(self, image_path: str) -> str:
        prompt = """Provide a single, concise sentence describing ONLY the character's
        facial features, hair, and identity-defining appearance. Ignore background,
        props, and lighting. Do not include introductory phrases."""
        desc = director_load_metadata(image_path)
        if not desc:
            desc = AnalyzeImage(image_path, prompt)['analysis']
        if desc:
            desc = desc[0].lower() + desc[1:]
        return desc

    def add_portrait_reference(self, builder, image_path: str, label: str,
                               target_subject_label: str, extra_desc: str = "",
                               generator=None):
        target_key = target_subject_label.lower()
        if target_key not in builder.entities:
            raise ValueError(f"Target subject '{target_subject_label}' not found. Add it first.")

        if not os.path.exists(image_path):
            if generator:
                print(f"Generating portrait {label} at {image_path}...")
                char_ref = builder.entities.get(target_subject_label.lower(), None)
                cref_path = char_ref['path'] if char_ref else ''
                generator('', cref_path, image_path)
            else:
                print(f"[Warning] Portrait file not found: {image_path}")

        desc = self.analyze_portrait(image_path)
        target_id = builder.entities[target_key]["id"]
        picture_index = len(builder.entities) + len(self.portrait_refs) + 1

        self.portrait_refs[target_id] = {
            "id": target_id,
            "path": image_path,
            "pic_tag": f"<Picture {picture_index}>",
            "desc": desc,
            "extra_desc": extra_desc,
            "target": target_subject_label
        }

    def rewrite_with_portraits(self, sub_defs):
        rewritten = []
        for line in sub_defs:
            prefix = "<Subject "
            id_part = line.split(">")[0]
            subj_id = int(id_part.split(prefix)[1])

            portrait = self.portrait_refs.get(subj_id)
            if not portrait:
                rewritten.append(line)
                continue

            base_line = line.rstrip(".")
            portrait_line = (
                f"{base_line}. Facial identity is reinforced by {portrait['pic_tag']}. "
                f"{portrait['desc']}. "
                f"{portrait['pic_tag']} is an identity-only reference containing facial "
                f"features ONLY. Ignore any background, lighting, framing, or spatial cues "
                f"present in the portrait. Do NOT use the portrait for camera distance, "
                f"cropping, or background inference."
            )
            rewritten.append(portrait_line)
        return rewritten

    def get_paths(self):
        return [data["path"] for data in self.portrait_refs.values()]

class SmartVideoPromptBuilder:
    def __init__(self):
        self.portrait_manager = PortraitReferenceManager()
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
        self._current_time_ms = 0
        self._default_shot_duration_ms = 2000
        self.first_frame_label = None

    @property
    def duration(self):
        return (self._current_time_ms / 1000.0) + 1.0

    def add_text_subject(self, label: str, desc: str, is_character: bool = False, is_environment: bool = False):
        self._subject_counter += 1
        self.entities[label.lower()] = {
            "id": self._subject_counter, "path": None, "pic_tag": None, "desc": desc,
            "is_character": is_character, "is_environment": is_environment, "shots": set(),
        }
        return self

    def _format_time(self, ms: int) -> str:
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _analyze_image(self, image_path: str, is_character: bool = False) -> str:
        if is_character:
            prompt = """Provide a single, concise sentence describing ONLY the character's physical appearance, 
            clothing, and distinguishing features. Ignore the background, setting, props, and other people."""
        else:
            prompt = """Provide a single, concise sentence describing the main visual elements, lighting, 
            atmosphere, and key objects in this environment/scene."""
        
        desc = AnalyzeImage(image_path, prompt)['analysis']
        if desc:
            desc = desc[0].lower() + desc[1:]
        return desc

    def add_subject(self, image_path: str, label: str, is_character: bool = False, is_environment: bool = False):
        self._subject_counter += 1
        sub_id = self._subject_counter
        pic_tag = f"<Picture {sub_id}>"
            
        desc = director_load_metadata(image_path)
        if not desc:
            print(f"Analyzing {image_path} as {'character' if is_character else 'background'}...")
            desc = self._analyze_image(image_path, is_character=is_character)
            director_save_metadata(image_path, desc)
        else:
            print(f"Loaded cached description for {image_path}.")
        
        self.entities[label.lower()] = {
            "id": sub_id, "path": image_path, "pic_tag": pic_tag, "desc": desc,
            "is_character": is_character, "is_environment": is_environment, "shots": set()
        }
        return self

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
            "id": len(self.audio_refs) + 1, "path": audio_path,
            "target_id": self.entities[target_key]["id"], "extra_desc": extra_desc
        }
        return self

    def set_scene_style(self, style: str):
        self.scene_style = style
        return self

    def add_shot(self, raw_text: str, duration: float = None, start_time: float = None):
        duration_ms = int(duration * 1000) if duration is not None else self._default_shot_duration_ms
        current_start_ms = int(start_time * 1000) if start_time is not None else self._current_time_ms
        timestamp_str = self._format_time(current_start_ms)
        
        self.shots.append({"raw_text": raw_text, "timestamp": timestamp_str, "duration_ms": duration_ms})
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

    def load_script(self, script_text: str, base_dir: str = "", input_dir: str = "", generators: dict = None, low_vram=False):
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
                    path_field = parts[2].strip()
                    prompt = parts[3] if len(parts) > 3 else ""

                    if not path_field or path_field == "-":
                        if not prompt:
                            raise ValueError(f"{cmd} '{label}' uses '-' but no description was provided.")
                        self.add_text_subject(label, prompt, is_character=(cmd == "char"), is_environment=(cmd == "bg"))
                        continue

                    candidate_paths = [
                        path_field, 
                        os.path.join(input_dir, path_field) if input_dir else "", 
                        os.path.join(os.getcwd(), path_field), 
                        os.path.join(base_dir, path_field)
                    ]
                    
                    resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), os.path.join(base_dir, path_field))
                        
                    if not os.path.exists(resolved_path):
                        if cmd in generators:
                            print(f"Generating {label} at {resolved_path}...")
                            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                            generators[cmd](prompt, resolved_path)
                        else:
                            print(f"[Warning] File not found: {path_field}. Skipping {cmd} '{label}'.")
                            continue
                    
                    if cmd == 'bg': self.add_background(resolved_path, label)
                    elif cmd == 'ff': self.add_firstframe(resolved_path, label)
                    elif cmd == 'char': self.add_character(resolved_path, label)
                    elif cmd == 'item': self.add_subject(resolved_path, label, is_character=False)
                        
                elif cmd == 'summary':
                    self.set_summary(parts[1] if len(parts) > 1 else "")
                        
                elif cmd == 'audio':
                    label = parts[1]
                    path_field = parts[2].strip()
                    target = parts[3] if len(parts) > 3 else ""
                    extra = parts[4] if len(parts) > 4 else ""
                    voice_prompt = parts[5] if len(parts) > 5 else "female"
                    
                    # Assign a valid default path if '-' is used
                    if not path_field or path_field == '-':
                        path_field = f"audio/{label}.wav"

                    candidate_paths = [
                        path_field, 
                        os.path.join(input_dir, path_field) if input_dir else "", 
                        os.path.join(os.getcwd(), path_field), 
                        os.path.join(base_dir, path_field)
                    ]
                    
                    # Find the first path that exists, otherwise default to base_dir + path_field
                    resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), os.path.join(base_dir, path_field))
                    
                    # CRITICAL FIX: Always convert to an absolute path
                    resolved_path = str(Path(resolved_path).resolve())

                    if not os.path.exists(resolved_path):
                        if 'audio' in generators:
                            print(f"Generating voice {label} at {resolved_path}...")
                            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                            generators['audio'](voice_prompt, resolved_path, long=True)
                        else:
                            print(f"[Warning] Audio file not found: {path_field}. Skipping.")
                            continue
                    
                    self.add_audio_reference(resolved_path, label, target, extra)
                    
                elif not low_vram and cmd == 'portrait':
                    label = parts[1]
                    path_field = parts[2].strip()
                    target = parts[3]
                    extra = parts[4] if len(parts) > 4 else ""

                    # FIX: Assign a valid default path if '-' is used
                    if not path_field or path_field == '-':
                        path_field = f"images/{label}_portrait.png"

                    candidate_paths = [
                        path_field, 
                        os.path.join(input_dir, path_field) if input_dir else "", 
                        os.path.join(os.getcwd(), path_field), 
                        os.path.join(base_dir, path_field)
                    ]
                    resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), os.path.join(base_dir, path_field))

                    if not os.path.exists(resolved_path):
                        generator = generators.get('portrait', None)
                        if generator:
                            print(f"Generating portrait {label} at {resolved_path}...")
                            target_key = target.lower()
                            char_ref = self.entities.get(target_key, None)
                            cref_path = char_ref['path'] if char_ref else ''
                            
                            # Ensure the directory exists before generating
                            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                            generator(extra, cref_path, resolved_path)
                        else:
                            print(f"[Warning] Portrait file not found: {path_field}. Skipping.")
                            continue
                    
                    self.portrait_manager.add_portrait_reference(self, resolved_path, label, target, extra_desc=extra, generator=None)
                    
                elif cmd == 'prompt':
                    self.set_scene_style(parts[1] if len(parts) > 1 else "")
                elif cmd == 'soundscape':
                    self.set_soundscape(parts[1] if len(parts) > 1 else "")
                elif cmd.startswith('shot'):
                    # Safely parse duration, catching ValueError if it's not a number
                    duration = float(parts[2]) if len(parts) > 2 and parts[2] else None
                    self.add_shot(parts[1], duration=duration)
                    
            except (IndexError, ValueError) as e:
                # Now you will SEE exactly which line is failing and why
                print(f"[Warning] Malformed line skipped: '{line}' (Error: {e})")
                
        return self

    def generate(self, low_vram=False) -> str:
        sections = []
        self.used_audio_refs = {}
        
        used_subject_ids = set()
        for shot in self.shots:
            text = self._substitute_labels(shot["raw_text"])
            matches = re.findall(r'<Subject (\d+)>(?=[^<]*\[(?:English|Spanish|French|German|Italian)\])', text)
            used_subject_ids.update(int(m) for m in matches)
        
        subject_to_speaker = {}
        speaker_counter = 1
        for label, data in self.audio_refs.items():
            target_id = data['target_id']
            speaker_tag = f"(S{speaker_counter})" if target_id in used_subject_ids else None
            if speaker_tag:
                subject_to_speaker[target_id] = speaker_tag
                speaker_counter += 1
            data['speaker_tag'] = speaker_tag
            self.used_audio_refs[label] = data

        sections.append("subject_definitions:")
        sub_defs = []
        for label, data in self.entities.items():
            has_picture = bool(data.get("pic_tag"))
            if data.get("is_environment"):
                sub_defs.append(f"<Subject {data['id']}> is the background environment " + (f"in {data['pic_tag']}, " if has_picture else "") + f"featuring {data['desc']}.")
            else:
                sub_defs.append(f"<Subject {data['id']}> is {data['desc']} " + (f"in {data['pic_tag']}." if has_picture else "."))
        
        if low_vram:
            sub_defs = self.portrait_manager.rewrite_with_portraits(sub_defs)

        audio_defs = []
        for label, data in self.used_audio_refs.items():
            extra = f", {data['extra_desc']}" if data['extra_desc'] else ""
            speaker = data['speaker_tag'] or ""
            audio_defs.append(f"<Audio {data['id']}> is the voice-timbre reference for <Subject {data['target_id']}> {speaker}{extra}.")
        sections.append("\n".join(sub_defs + audio_defs))

        if self.summary:
            sections.append("\nsummary:")
            sections.append(self._substitute_labels(self.summary))
                
        if self.first_frame_label and self.first_frame_label in self.entities:
            pic_tag = self.entities[self.first_frame_label]["pic_tag"]
            ff_rule = f"""{pic_tag} is the first frame of [Shot 1]. The first frame must match {pic_tag} exactly for SPATIAL COMPOSITION: identical pose, head angle, hand position, body orientation, camera angle, and spatial relationships with zero deviation.
However, CHARACTER IDENTITY (facial features, clothing details, body proportions, hair texture) must be corrected and overridden by the character reference images to prevent feature degradation. The character references are the source of truth for identity; the first frame is the source of truth for composition."""
            sections.append(ff_rule)
            sections.append("")
            
        if self.scene_style:
            sections.append(self.scene_style)

        scene_shots = []
        for i, shot in enumerate(self.shots):
            processed_text = self._substitute_labels(shot["raw_text"])
            for label, entity in self.entities.items():
                if f"<Subject {entity['id']}>" in processed_text:
                    entity["shots"].add(i + 1)
            
            def inject_speaker(match):
                subject_tag = match.group(1)
                subject_id = int(re.search(r'\d+', subject_tag).group())
                return f"{subject_tag} {subject_to_speaker.get(subject_id, '')}"
            
            processed_text = re.sub(r'(<Subject \d+>)(?=[^<]*\[(?:English|Spanish|French|German|Italian)\])', inject_speaker, processed_text)
            pattern = r'(<Subject \d+>)([^"\[]*?)\[(?:English|Spanish|French|German|Italian)\]\s*("[^"]*")'
            processed_text = re.sub(pattern, lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}", processed_text)

            time_str = f" At {shot['timestamp']}," if shot.get("timestamp") else ""
            scene_shots.append(f"[Shot {i+1}]{time_str} {processed_text}")
            
        sections.append("\nretention_analysis:")
        retention = []
        for label, data in self.entities.items():
            shots_list = sorted(list(data["shots"]))
            shots_str = ", ".join([f"[Shot {s}]" for s in shots_list])
            retention.append(f"<Subject {data['id']}> (appears in {shots_str}): fully_preserved - {data['desc']} is retained.")
        for label, data in self.used_audio_refs.items():
            sections.append(f"<Audio {data['id']}>: reference - its vocal timbre guides the dialogue delivery for <Subject {data['target_id']}>.")
        sections.append("\n".join(retention))
        
        sections.append("\ndetailed_description:")
        sections.append('\n'.join(scene_shots))
        sections.append("\noverall_soundscape:")
        sections.append(self.soundscape)
        sections.append("\nnon_diegetic_music:")
        sections.append(self.non_diegetic_music)
        
        return "\n".join(sections)

def send(prompt, images, audio, output='output.mp4', width=768, height=448, duration=5.0, steps=4, start_image=True, upscale=os.environ.get('UPSCALE', 'False') != 'False', debug=False):
    model = "minimax_h3_ref2va_pruned"
    args = requests.get(f"http://127.0.0.1:8080/defaults/{model}").json()

    args['output_filename'] = Path(output).name
    args['prompt'] = prompt
    args["seed"] = SEED
    if steps <= 8:
        args["activated_loras"] = ["minimax_h3_larryvrh_v4_step600_ema.safetensors"]
        args["loras_multipliers"] = "1.0|"
    if len(audio) == 1: 
        args["audio_prompt_type"] = "A"
        args["audio_guide"] = audio[0]
    if len(audio) == 2:
        args["audio_prompt_type"] = "AB"
        args["audio_guide2"] = audio[1]
            
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
    args["num_inference_steps"] = steps
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
    if steps >= 8:
        args["custom_settings"] = { "h3_mask_mode": "grouped_rows", "audio_refinement": "enabled" }
        
    if debug:
        json_filename = output.replace('.mp4', '.json')
        with open(json_filename, 'w', encoding='utf-8') as js:
            js.write(json.dumps(args, indent=4))
            
    args['output_dir'] = str(Path(output).parent)
    if args['output_dir'][-1] == '.':
        args['output_dir'] = args['output_dir'][:-1]
        
    print(f"[Debug] Sending to API with {len(images) if images else 0} image refs and {len(audio) if audio else 0} audio refs.")
    job_id = requests.post("http://127.0.0.1:8080/run", json=args).json()
    print(f"Job ID: {job_id}")

    last = ''
    dedupe_updates = set([])
    while status := requests.get(f"http://127.0.0.1:8080/status/{job_id}").json()[-1] in ("pending", "running"):
        sleep(5)
        update = requests.get(f"http://127.0.0.1:8080/updates/{job_id}").json()
        if update:
            update = update[0].strip()
            if update not in dedupe_updates:
                dedupe_updates.add(update)
                print(update)
    print(requests.get(f"http://127.0.0.1:8080/status/{job_id}").json()[-2:])

def get_builder(script, output_dir):
    if ANIME:
        from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    elif os.environ.get('IMAGE_GEN', 'KLEIN') == 'QWEN21':
        from plan10.lib.qwen21 import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    else:
        from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    from plan10.lib.dialog import DesignVoice
    
    base_dir = str((Path.cwd() / output_dir).resolve())
    generators = {
        'bg': CreateBackground, 'char': CreateCharacterSheet, 'ff': GenerateImage,
        'item': GenerateImage, 'audio': partial(DesignVoice, long=False), 'portrait': CreatePortrait
    }
    return SmartVideoPromptBuilder().load_script(script, base_dir=base_dir, generators=generators)

def main():
    import argparse, sys
    if ANIME:
        from plan10.lib.anime_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    elif os.environ.get('IMAGE_GEN', 'KLEIN') == 'QWEN21':
        from plan10.lib.qwen21 import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    else:
        from plan10.lib.image_gen import GenerateImage, CreateCharacterSheet, CreateBackground, CreatePortrait
    from plan10.lib.dialog import DesignVoice
    
    parser = argparse.ArgumentParser(description='Cinematic Director')
    parser.add_argument('-O', '--output', type=str, default='feedback_output')
    parser.add_argument('-I', '--input', type=str, default=None)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-W', '--width', type=int, default=int(os.environ.get("WIDTH", "768")))
    parser.add_argument('-H', '--height', type=int, default=int(os.environ.get("HEIGHT", "576")))
    parser.add_argument('-S', '--steps', type=int, default=4)
    parser.add_argument('--wangp', action="store_true")
    parser.add_argument('--low-vram', action='store_true')
    args = parser.parse_args()

    os.environ['WIDTH'] = str(args.width)
    os.environ['HEIGHT'] = str(args.height)
    WIDTH = (args.width // 32) * 32
    HEIGHT = (args.height // 32) * 32

    base_dir = Path.cwd() / args.output
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "audio").mkdir(parents=True, exist_ok=True)
    
    input_dir = str(Path(args.input).parent.resolve()) if args.input else str(Path.cwd())
    base_input = str((Path.cwd() / args.input).resolve()) if args.input else ""

    generators = {
        'bg': CreateBackground, 'char': CreateCharacterSheet, 'ff': GenerateImage,
        'item': GenerateImage, 'audio': partial(DesignVoice, long=True), 'portrait': CreatePortrait
    }
    
    builder = None
    final_prompt = ''
    
    if args.input:
        if args.input.endswith('.mmh3'):
            final_prompt = Path(args.input).read_text(encoding='utf-8')
            print("[Info] Loaded pre-compiled .mmh3 prompt. Note: Image references must be extracted from a paired .txt or handled manually.")
        else:
            script = Path(args.input).read_text(encoding='utf-8')
    else:
        script = """
        # --- ASSETS (with generation prompts) ---
        bg   | barn    | images/rustic_barn.png    | the inside of a rustic barn, with a large open doorway allowing light to spill in
        char | blondie | -                         | a medium shot of a blonde woman, white sundress, white tennis shoes
        char | red     | images/red_woman.png      | a medium shot of a red haired woman, blue jeans, tshirt, cowboy boots
        item | dog     | images/samoyed_dog.png    | a samoyed dog
        
        # --- AUDIO REFERENCES ---
        audio | voice_r | audio/voice_sample2.wav | red | containing a spoken English vocal layer | female
        portrait | red_port | images/red_portrait.png | red | a red haired woman
        
        # --- SCENE CONTEXT ---
        prompt      | The target video uses a realistic cinematic style with warm golden hour lighting.
        soundscape  | Ambient wind and soft acoustic guitar music.
        
        # --- SHOTS ---
        shot | Sound of roosters, as The camera pushes in on blondie holding a treat and red stands beside her with her arms crossed inside barn environment
        shot | Sounds of a barking dog, Following shot The dog runs and jumps up to grab the treat from blondie inside barn environment | 1
        shot | Sounds of cows mooing, Camera pushes in on red as she speaks [English] "You spoil him." She finishes speaking standing with her mouth closed for a static shot.
        """

    if not final_prompt:
        builder = SmartVideoPromptBuilder().load_script(
            script, 
            base_dir=str(base_dir.resolve()), 
            input_dir=input_dir,
            generators=generators, 
            low_vram=args.low_vram
        )
        final_prompt = builder.generate(args.low_vram)
        
    print(final_prompt)
    
    if args.input and not args.input.endswith('.mmh3'):
        Path(args.input.replace('.txt', '.mmh3')).write_text(final_prompt, encoding='utf-8')

    start_image = False

    # Extract paths dynamically from the builder
    if builder is not None:
        img_refs = []
        
        # FIRST: Add first frame if it exists
        if builder.first_frame_label and builder.first_frame_label in builder.entities:
            ff_data = builder.entities[builder.first_frame_label]
            if ff_data["path"] and os.path.exists(ff_data["path"]):
                img_refs.append(ff_data["path"])
                print(f"[Debug] Added first frame: {ff_data['path']}")
                start_image=True
            else:
                print(f"[Warning] First frame path not found: {ff_data.get('path')}")
        
        # THEN: Add all other entities (bg, char, item)
        for label, data in builder.entities.items():
            if label == builder.first_frame_label:
                continue  # Skip ff, already added
            if data["path"] and os.path.exists(data["path"]):
                img_refs.append(data["path"])
                print(f"[Debug] Added entity: {label} -> {data['path']}")
        
        # Add portraits if not low_vram
        if not args.low_vram:
            for portrait_path in builder.portrait_manager.get_paths():
                if os.path.exists(portrait_path):
                    img_refs.append(portrait_path)
                    print(f"[Debug] Added portrait: {portrait_path}")
        
        aud_refs = [data["path"] for data in builder.used_audio_refs.values() if os.path.exists(data["path"])]
    else:
        img_refs = []
        aud_refs = []

    if args.low_vram:
        # SAFE RESIZE WITH FALLBACK: Prevents resize functions from silently dropping valid images
        #width and height must be multiples of 32, 1344x768, 864x480 minimal
        from plan10.lib.util import resize_low_vram_png
        img_refs_resized = []
        for ref in img_refs:
            if not ref or not os.path.exists(ref):
                print(f"[Warning] Image reference not found, skipping: {ref}")
                continue
            try:
                out = resize_low_vram_png(ref, divisor=32)
                # Validate that the resized file actually exists and was created
                if out and os.path.exists(out) and os.path.getsize(out) > 0:
                    img_refs_resized.append(out)
                    print(f"[Debug] Successfully resized: {ref} -> {out}")
                else:
                    # Fallback to original if resize failed
                    print(f"[Warning] Resize failed for {ref}, using original")
                    img_refs_resized.append(ref)
            except Exception as e:
                print(f"[Warning] Failed to resize {ref}: {e}. Using original path.")
                img_refs_resized.append(ref)
        img_refs = img_refs_resized

    print(f"\n[Debug] Final image refs to be used ({len(img_refs)}): {img_refs}")
    print(f"[Debug] Final audio refs to be used ({len(aud_refs)}): {aud_refs}\n")

    output_filename = f"{base_input.replace('.txt', '.mp4')}" if args.input else str((base_dir / "output.mp4").resolve())

    if args.debug and not args.wangp:
        sys.exit(0)

    if args.wangp:
        send(
            final_prompt, img_refs, aud_refs, output=output_filename, start_image=start_image,
            width=args.width, height=args.height, duration=builder.duration if builder else 5.0,
            steps=args.steps, debug=args.debug
        )
    else:
        from plan10.lib.mmh3 import compose_video
        print(compose_video(final_prompt, img_refs, aud_refs, output_filename, args.width, args.height, builder.duration if builder else 5.0))

if __name__ == '__main__':
    main()