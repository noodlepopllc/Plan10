from plan10.lib.config import load_config
load_config()

from auk.infer.infer_auk import AukInfer, save_audio
from plan10.lib.qwen_llm import llm_analyze_media
import sys, json, re, librosa
from plan10.lib.util import transcribe, estimate_f5_baseline_duration


from huggingface_hub import snapshot_download
import os

basepath = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH","./models")

auk_base_repo = "tencent/AuK"
auk_base_path = f"{basepath}/ckpts/AuK"

auk_flash_repo = "tencent/AuK-Flash"
auk_flash_path = f"{basepath}/ckpts/AuK-Flash"

mllm_repo = "Qwen/Qwen2.5-Omni-3B"
mllm_path = f"{basepath}/ckpts/Qwen2.5-Omni-3B"

import os
import yaml
import pathlib

BASE = pathlib.Path(os.environ["DIFFSYNTH_MODEL_BASE_PATH"])
CKPTS = BASE / "ckpts"

import torch
import gc
from auk.infer.infer_auk import AukInfer

def patch_auk_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model",{})

    # AuK-Flash uses text_encoder_path
    if "text_encoder_path" in model.get("text_encoder", {}):
        rel = model["text_encoder"]["text_encoder_path"]
        tail = pathlib.Path(rel).name
        model["text_encoder"]["text_encoder_path"] = str(CKPTS / tail)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)


def ensure_model(repo, path):
    if not os.path.exists(path):
        snapshot_download(repo, local_dir=path)
        name = pathlib.Path(path).name

        if name.startswith("AuK"):
            patch_auk_yaml(pathlib.Path(path) / "config.yaml")

# checkpoint = "ckpts/AuK/auk_base.safetensors"
# config = "ckpts/AuK/config.yaml"
# ensure_model(auk_base_repo, auk_base_path)

# Use AuK-Flash instead:
checkpoint = f"{auk_flash_path}/auk_flash.safetensors"
config = f"{auk_flash_path}/config.yaml"

ensure_model(auk_flash_repo, auk_flash_path)
ensure_model(mllm_repo, mllm_path)

class DialogSession:
    def __init__(self, config_path=config, checkpoint_path=checkpoint):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None

    def __enter__(self):
        # Load AuK here
        self.model = AukInfer(self.config_path, self.checkpoint_path, cpu_offload=True)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    def cleanup(self):
        # Clean up GPU
        try:
            del self.model
        except Exception:
            pass

        self.model = None

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def parse_omnivoice(desc: str):
    parts = [p.strip().lower() for p in desc.split(",")]
    gender = next((p for p in parts if p in ["male", "female"]), None)
    age = next((p for p in parts if p in [
        "child", "teenager", "young adult", "middle-aged", "elderly"
    ]), None)
    pitch = next((p for p in parts if "pitch" in p), None)
    accent = next((p for p in parts if "accent" in p), None)
    style = "whisper" if "whisper" in parts else None
    return gender, age, pitch, accent, style

def map_gender(g):
    return "男性" if g == "male" else "女性"

def map_age(a):
    return {
        "child": "一位十岁左右的儿童",
        "teenager": "一位十几岁的青少年",
        "young adult": "一位二十多岁的成年人",
        "middle-aged": "一位中年人",
        "elderly": "一位年长者",
    }.get(a, "一位成年人")

def map_pitch(p):
    return {
        "very low pitch": "音色极低沉",
        "low pitch": "音色低沉",
        "moderate pitch": "音色适中",
        "high pitch": "音色偏高",
        "very high pitch": "音色极高",
    }.get(p, "")

def map_accent(a):
    return {
        "british accent": "带有轻微的英式口音",
        "american accent": "带有轻微的美式口音",
        "australian accent": "带有轻微的澳洲口音",
        "canadian accent": "带有轻微的加拿大口音",
        "chinese accent": "带有轻微的中文口音",
        "indian accent": "带有轻微的印度口音",
        "japanese accent": "带有轻微的日式口音",
        "korean accent": "带有轻微的韩式口音",
        "portuguese accent": "带有轻微的葡萄牙口音",
        "russian accent": "带有轻微的俄式口音",
    }.get(a, "")

def map_style(s):
    return "以轻声耳语的方式说话" if s == "whisper" else ""


def build_auk_prompt(desc, text):
    gender, age, pitch, accent, style = parse_omnivoice(desc)

    demo = f"{map_age(age)}的{map_gender(gender)}"
    tone = f"{map_pitch(pitch)}{',' if pitch else ''}{map_style(style)}"
    tone = tone.strip(" ,")

    # Add neutral filler if accent is missing
    if accent is None:
        filler = "声音清晰，音量适中"
    else:
        filler = map_accent(accent)

    chinese_style = (
        f"{demo}，在安静的环境中，以自然、平稳的语气说话，"
        f"{tone if tone else '音色自然'}，{filler}。"
    )

    return f'请基于下面的描述: "{chinese_style}", 生成语音内容 "{text}".'

def run_auk(
    instruction,
    output_path,
    audio_path=None,
    gen_seconds=None,
    model=None
):
    content = [{"type": "text", "text": instruction}]

    if audio_path is not None:
        content.append({"type": "audio", "audio": audio_path})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    if model:
        audio, sr = model.generate(
            messages,
            gen_seconds=gen_seconds,
        )
        save_audio(audio, sr, output_path)

def CloneVoice(text, audio, output, duration=5.0, seed=-1, lengthen=True, session=None):
    # The actual prompt fed into the model

    this_session = session if session else DialogSession()

    duration=float(duration)
    seed=int(seed)

    if lengthen and len(text.split(' ')) < 5:
        text = f"{text} ... Random words added for length."

    estimate = estimate_f5_baseline_duration(text)

    # If user did NOT specify a duration, use the estimate
    if duration is None:
        duration = estimate

    # If user DID specify a duration, blend or respect it
    else:
        duration = max(duration, estimate)


    run_auk(
        f"Say the following with the same voice: '{text}",
        output,
        audio_path=audio,
        gen_seconds=duration,
        model=session.__enter__()
    )

    if not session:
        this_session.cleanup()

    transcription = " ".join(transcribe(output)) if lengthen else ''

    duration = round(librosa.get_duration(path=output), 2)

    description = (
        f"Cloned voice.\n"
        f"Reference audio: {audio}\n"
        f"Duration: {duration} seconds\n"
        f"Transcription: \"{transcription}\""
    )

    return {
        "status": "success",
        "description": description,
        "output_path": output,
        "prompt": final_prompt
    }

def DesignVoice(voice=None, output='output.wav', seed=-1, long=False):
    duration=10.0 if long else 5.0
    short_text = "The quick, anxious boy judged the rough wizard's vibrant, icy voice..." # as a huge, sharp, mellow echo drifting through the quiet, yellow forest."
    long_text = "The quick, anxious boy judged the rough wizard's vibrant, icy voice as a huge, sharp, mellow echo drifting through the quiet, yellow forest."
    text_to_speak = long_text if long else short_text
    style_desc = (
        '"A woman in her twenties, speaking softly to her partner who just arrived home. '
        'Her tone is gentle, caring, and slightly playful. '
        'Her voice is soft, intimate, with a slightly slower speech rate and moderate volume. '
        'The timbre is sweet and natural, featuring a warm, upward inflection at the end of phrases." '
        )
    language = "en"

    # 2. Calculate F5 baseline
    f5_duration = estimate_f5_baseline_duration(text_to_speak, language)
    print(f"F5 Baseline Duration: {f5_duration:.2f}s")
    llm_payload = {
        "items": [
            {
                "key": "request",
                "language": language,
                "content": text_to_speak,
                "style_instruction": style_desc,
                "f5_duration_sec": round(f5_duration, 6),
            }
        ]
    }
    final_duration = f5_duration
    instruction = build_auk_prompt(voice, text_to_speak)

    with DialogSession() as session:
        # Use the LLM-refined duration
        run_auk(instruction, output, gen_seconds=final_duration, model=session)

    duration = round(librosa.get_duration(path=output), 2)

    description = (
        f"Designed voice.\n"
        f"Voice style: {style_desc}\n"
        f"Duration: {duration:.2f} seconds\n"
        f"Transcription: \"{text_to_speak}\""
    )

    return {
        "status": "success",
        "description": description,
        "output_path": output,
        "prompt": voice
    }

def main():
    import argparse, math
    import sys, json
    from pathlib import Path
    parser = argparse.ArgumentParser(
                    prog='GenerateDialog',
                    description='Generate voices with dialog',
                    epilog='')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-I', '--instruct', type=str, default='female, low pitch, british accent', help='instructions for voice')
    parser.add_argument('-O', '--output', type=str, default='output.wav', help='output filename')
    parser.add_argument('-L', '--long', action='store_true', help='increased duration for designed voice')
    args = parser.parse_args()

    DesignVoice(args.instruct, args.output, args.seed, args.long)


if __name__ == '__main__':
    main()

def main():
    import argparse, math
    import sys, json
    from pathlib import Path
    parser = argparse.ArgumentParser(
                    prog='GenerateDialog',
                    description='Generate voices with dialog',
                    epilog='')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-T', '--text', type=str, default='hello how are you today', help='output text')
    parser.add_argument('-I', '--instruct', type=str, default='female, low pitch, british accent', help='instructions for voice')
    parser.add_argument('-R', '--ref-audio', type=str, default=None, help='audio to be cloned')
    parser.add_argument('-O', '--output', type=str, default='output.wav', help='output filename')
    parser.add_argument('-W', '--no-whisper', action='store_false', help='turn off whisper transcription')
    parser.add_argument('-D', '--duration', type=float, default=5.0, help='duration of the generated clip')
    parser.add_argument('-S', '--transcribe', action='store_true', help='transcribe the reference audio')
    parser.add_argument('-L', '--long', action='store_true', help='increased duration for designed voice')
    parser.add_argument('-P', '--plus', action='store_true', help='detailed timestamps with transcript')
    args = parser.parse_args()
    if not args.ref_audio:
        DesignVoice(args.instruct, args.output, args.seed, args.long)
    elif args.transcribe and args.ref_audio:
        output = transcribe(args.ref_audio, detailed=args.plus)
        if args.plus:
            output = str(output)
        else:
            output = ' '.join(output)
        dur = -1
        if args.output.endswith('.txt'):
            if args.ref_audio.endswith('.mp4'):
                Path(args.output).write_text(f'{str(output).strip()}')
                sys.exit()
            y, sr = librosa.load(args.ref_audio, sr=None)
            dur = librosa.get_duration(y=y, sr=sr)
            Path(args.output).write_text(f'{math.ceil(dur)}|{str(output).strip()}')
        print(f'Duration: {math.ceil(dur)} seconds, Text: "{str(output).strip()}"')
    else:
        CloneVoice(args.text, args.ref_audio, args.output, duration=args.duration, seed=args.seed, lengthen=args.long, session=None)

if __name__ == '__main__':
    main()

