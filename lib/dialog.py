import torch, torchaudio, gc, librosa, traceback
from omnivoice import OmniVoice
import numpy as np
from faster_whisper import WhisperModel


def transcribe(path):

    model_size = "large-v3"

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(path, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    segs = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        segs.append(segment.text)
    return segs

def create_audio_and_free_vram(
    text, 
    instruct='female, low pitch, british accent', 
    ref_audio='',
    ref_text='',
    output='temp.wav',
    max_retries=2,
    max_duration_seconds=5.0,
    target_sr=16000,
    seed=-1,
    use_whisper=True
):
    """
    Generate audio via OmniVoice with robust validation:
    - RMS check
    - histogram check
    - voiced-frame check
    - optional Whisper semantic verification
    """

    # Optional reference transcription
    if ref_audio and not ref_text:
        segs = transcribe(ref_audio)
        ref_text = " ".join(segs)

    start_silence_ms = 300
    end_silence_ms = 500
    speed = 0.85

    for attempt in range(1, max_retries + 1):
        torch.cuda.empty_cache()
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32)

        if seed != -1:
            torch.cuda.manual_seed(seed)

        try:
            with torch.no_grad():
                if ref_audio:
                    audio = model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text)
                else:
                    audio = model.generate(text=text, instruct=instruct, speed=speed)

            # OmniVoice returns numpy -> convert to tensor, move to CPU, fix shape
            audio_tensor = torch.from_numpy(audio[0]).cpu()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # torchaudio expects (channels, samples)
                
            torchaudio.save(output, audio_tensor, 24000)
            input_audio, sr = librosa.load(output, sr=target_sr, mono=True, dtype=np.float32)

            # --- SIGNAL QUALITY VALIDATION ---
            peak = np.max(np.abs(input_audio))
            rms = np.sqrt(np.mean(input_audio**2))
            hist, _ = np.histogram(input_audio, bins=50, range=(-1, 1))
            nonzero_bins = np.sum(hist > 10)
            voiced_frames = np.sum(np.abs(input_audio) > 0.02)

            if (
                peak < 0.02 or
                rms < 0.005 or
                nonzero_bins < 5 or
                voiced_frames < 200 or
                np.isnan(input_audio).any()
            ):
                print(f"⚠️ Low-quality audio (attempt {attempt}), retrying...")
                continue

            # --- OPTIONAL WHISPER VALIDATION ---
            if use_whisper:
                segs = transcribe(output)
                if len(segs) == 0:
                    print(f"⚠️ Whisper found no speech (attempt {attempt}), retrying...")
                    continue

                transcribed = " ".join(segs).strip()
                if len(transcribed.split()) < 2:
                    print(f"⚠️ Whisper detected too little content, retrying...")
                    continue

            # Add silence padding
            start_samples = int((start_silence_ms / 1000) * sr)
            end_samples = int((end_silence_ms / 1000) * sr)
            input_audio = np.concatenate([
                np.zeros(start_samples),
                input_audio,
                np.zeros(end_samples)
            ])

            # Duration enforcement
            actual_duration = len(input_audio) / sr
            if actual_duration > max_duration_seconds:
                print(f"⚠️ Audio too long ({actual_duration:.2f}s), regenerating with duration cap...")

                with torch.no_grad():
                    if ref_audio:
                        audio = model.generate(
                            text=text, ref_audio=ref_audio, ref_text=ref_text,
                            duration=max_duration_seconds
                        )
                    else:
                        audio = model.generate(
                            text=text, instruct=instruct,
                            duration=max_duration_seconds
                        )

                # 🔧 FIX: Convert numpy → tensor, move to CPU, ensure (1, samples) shape
                audio_tensor = torch.from_numpy(audio[0]).cpu()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                    
                torchaudio.save(output, audio_tensor, 24000)
                input_audio, sr = librosa.load(output, sr=target_sr, mono=True, dtype=np.float32)

                # Re-check RMS + histogram
                rms = np.sqrt(np.mean(input_audio**2))
                if rms < 0.005:
                    print(f"⚠️ Constrained output still weak, retrying...")
                    continue

            print(f"✅ Clean audio generated (attempt {attempt})")

            del model, audio
            gc.collect()
            torch.cuda.empty_cache()
            return input_audio, sr

        except Exception as e:
            print(traceback.format_exc())
            print(f"❌ Failed (attempt {attempt}): {e}")

        finally:
            try: del model, audio
            except: pass
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError(f"Failed to generate valid audio ≤{max_duration_seconds}s after {max_retries} retries.")


def VoiceDesignSchema():
    return {
        "type": "function",
        "function": {
            "name": "design_voice",
            "description": "Generate speech using a synthetic designed voice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak."
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice description (timbre, pitch, accent, energy, etc.)."
                    },
                    "output": {
                        "type": "string",
                        "description": "Output WAV file path."
                    },
                    "duration": {
                        "type": "number",
                        "description": "Optional target duration in seconds."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional seed for deterministic output."
                    }
                },
                "required": ["text", "voice", "output"]
            }
        }
    }

def VoiceCloneSchema():
    return {
        "type": "function",
        "function": {
            "name": "clone_voice",
            "description": "Generate speech using a cloned voice from reference audio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak."
                    },
                    "audio": {
                        "type": "string",
                        "description": "Path to reference audio file."
                    },
                    "ref_text": {
                        "type": "string",
                        "description": "Optional transcription of the reference audio."
                    },
                    "output": {
                        "type": "string",
                        "description": "Output WAV file path."
                    },
                    "duration": {
                        "type": "number",
                        "description": "Optional target duration in seconds."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional seed for deterministic output."
                    }
                },
                "required": ["text", "audio", "output"]
            }
        }
    }

def DesignVoice(text, voice, output, duration=5.0, seed=-1):
    # The actual prompt fed into the model
    final_prompt = f"{text} | voice: {voice}"
    duration=float(duration)
    seed=int(seed)

    audio, sr = create_audio_and_free_vram(
        text=text,
        instruct=voice,
        output=output,
        max_duration_seconds=duration,
        seed=seed,
        use_whisper=True
    )

    actual_duration = len(audio) / sr
    transcription = " ".join(transcribe(output))

    description = (
        f"Designed voice.\n"
        f"Voice style: {voice}\n"
        f"Duration: {actual_duration:.2f} seconds\n"
        f"Transcription: \"{transcription}\""
    )

    return {
        "status": "success",
        "description": description,
        "output_path": output,
        "prompt": final_prompt
    }

def CloneVoice(text, audio, output, duration=5.0, seed=-1):
    # The actual prompt fed into the model
    final_prompt = f"{text} | cloned from: {audio}"
    duration=float(duration)
    seed=int(seed)

    _audio, sr = create_audio_and_free_vram(
        text=text,
        ref_audio=audio,
        ref_text='',
        output=output,
        max_duration_seconds=duration,
        seed=seed,
        use_whisper=True
    )

    actual_duration = len(_audio) / sr
    transcription = " ".join(transcribe(output))

    description = (
        f"Cloned voice.\n"
        f"Reference audio: {audio}\n"
        f"Duration: {actual_duration:.2f} seconds\n"
        f"Transcription: \"{transcription}\""
    )

    return {
        "status": "success",
        "description": description,
        "output_path": output,
        "prompt": final_prompt
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog='GenerateDialog',
                    description='Generate voices with dialog',
                    epilog='')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-T', '--text', type=str, default='hello how are you today', help='output text')
    parser.add_argument('-I', '--instruct', type=str, default='female, low pitch, british accent', help='instructions for voice')
    parser.add_argument('-R', '--ref-audio', type=str, default='', help='audio to be cloned')
    parser.add_argument('-O', '--output', type=str, default='output.wav', help='output filename')
    parser.add_argument('-W', '--no-whisper', action='store_false', help='turn off whisper transcription')
    parser.add_argument('-D', '--duration', type=float, default=5.0, help='duration of the generated clip')
    args = parser.parse_args()
    create_audio_and_free_vram(args.text, args.instruct, args.ref_audio, '', args.output, 2, args.duration, 16000, args.seed, args.no_whisper)
    #print(design_voice('hello world, greetings from the empire',instruct='female,american accent',output='hello.wav'))
    #print(clone_voice('this is cool if it works', ref_audio='hello.wav', output='cool.wav'))
    #print(transcribe('hello.wav'))
    #print(transcribe('cool.wav'))
