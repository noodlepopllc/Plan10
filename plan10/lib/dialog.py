from plan10.lib.config import load_config
load_config()

from plan10.lib.qwen_llm import llm_analyze_media
import sys, json, re, torch, gc, librosa
from plan10.lib.util import transcribe, estimate_f5_baseline_duration
        
import soundfile as sf

class DialogSession:
    def __init__(self, model_type="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        self.model_type = model_type
        self.model = None

    def __enter__(self):
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_type,
            device_map="auto",
            dtype=torch.bfloat16
        )
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


def CloneVoice(text, audio, output, duration=5.0, seed=-1, lengthen=True, session=None):
    # The actual prompt fed into the model

    if not session:
        this_session = DialogSession( "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        model = this_session.__enter__()
    else:
        model = session

    duration=float(duration)
    seed=int(seed)

    if lengthen and len(text.split(' ')) < 5:
        text = f"{text} ... Random words added for length."

    estimate = estimate_f5_baseline_duration(text)

    # If user did NOT specify a duration, use the estimate
    if duration is None:
        duration = estimate

    else:
        duration = max(duration, estimate)

    ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
    ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=audio,
        ref_text=' '.join(transcribe(audio)),
    )
    sf.write(output, wavs[0], sr)

    duration = round(librosa.get_duration(path=output), 2)

    transcription = ' '.join(transcribe(output))


    if not session:
        this_session.cleanup()


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
        "prompt": text
    }

def DesignVoice(voice=None, output='output.wav', seed=-1, long=False):
    duration=10.0 if long else 5.0
    short_text = "The quick, anxious boy judged the rough wizard's vibrant, icy voice..." # as a huge, sharp, mellow echo drifting through the quiet, yellow forest."
    long_text = "The quick, anxious boy judged the rough wizard's vibrant, icy voice as a huge, sharp, mellow echo drifting through the quiet, yellow forest."
    text_to_speak = long_text if long else short_text
    language = "English"

    # 2. Calculate F5 baseline
    f5_duration = estimate_f5_baseline_duration(text_to_speak, language)

    final_duration = f5_duration
    instruction = build_auk_prompt(voice, text_to_speak)

    with DialogSession("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") as session:
        wavs, sr = session.generate_voice_design(
            text=text_to_speak,
            language=language,
            instruct=instruction,
        )
        sf.write(output, wavs[0], sr)

    duration = round(librosa.get_duration(path=output), 2)

    description = (
        f"Designed voice.\n"
        f"Voice style: {instruction}\n"
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

