from plan10.lib.config import load_config
load_config()

from auk.infer.infer_auk import AukInfer, save_audio
from plan10.lib.qwen_llm import llm_analyze_media
import sys, json, re
from plan10.lib.util import transcribe

# Constants extracted directly from pe.config.yaml
TTS_SEC_PER_UTF8_BYTE = {"en": 0.0656, "zh": 0.0803}
F5_SHORT_TEXT_BYTE_THRESHOLD = 10
F5_SHORT_TEXT_SPEED = 0.3
F5_SAMPLE_RATE = 24000
F5_HOP_LENGTH = 256

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



def estimate_f5_baseline_duration(text: str, language: str = "en") -> float:
    """Replicates _estimate_f5_instruct_duration from pe.py"""
    # Simple UTF-8 byte weight (sufficient for single-language prompts)
    byte_count = len(text.encode("utf-8"))
    weight = byte_count * TTS_SEC_PER_UTF8_BYTE.get(language, 0.0656)
    
    # Short text speed adjustment
    speed_multiplier = F5_SHORT_TEXT_SPEED if byte_count < F5_SHORT_TEXT_BYTE_THRESHOLD else 1.0
    
    frames = int(weight * F5_SAMPLE_RATE / F5_HOP_LENGTH / speed_multiplier)
    return frames * F5_HOP_LENGTH / F5_SAMPLE_RATE

# checkpoint = "ckpts/AuK/auk_base.safetensors"
# config = "ckpts/AuK/config.yaml"

# Use AuK-Flash instead:
checkpoint = "ckpts/AuK-Flash/auk_flash.safetensors"
config = "ckpts/AuK-Flash/config.yaml"

engine = AukInfer(
    config,
    checkpoint,
)

def run_auk(
    instruction,
    output_path,
    audio_path=None,
    gen_seconds=None,
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

    audio, sr = engine.generate(
        messages,
        gen_seconds=gen_seconds,
    )
    save_audio(audio, sr, output_path)

def CloneVoice(text, audio, output, duration=5.0, seed=-1, lengthen=True, session=None):
    # The actual prompt fed into the model
    final_prompt = f"{text} | cloned from: {audio}"
    duration=float(duration)
    seed=int(seed)

    if lengthen and len(text.split(' ')) < 5:
        text = f"{text} ... Random words added for length."

    estimate = estimate_f5_baseline_duration(text) 

    duration = estimate if estimate < duration else duration

    run_auk(
        f"Say the following with the same voice: '{text}",
        output,
        audio_path=audio,
        gen_seconds=duration,
    )

    transcription = " ".join(transcribe(output)) if lengthen else ''

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

    # Use the LLM-refined duration
    run_auk(instruction, output, gen_seconds=final_duration)

    description = (
        f"Designed voice.\n"
        f"Voice style: {style_desc}\n"
        f"Duration: {final_duration:.2f} seconds\n"
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

