import sys, os, time
from plan10.lib.config import load_environ
load_environ()


from plan10.lib.qwen_llm import llm_analyze_media

from plan10.lib.image_gen import GenerateImage
from plan10.lib.image_to_video import GenerateVideo
from pathlib import Path
import torch, torchaudio


from PIL import Image
from PIL.PngImagePlugin import PngInfo

if os.environ.get('WGP','False') == 'True':
    from plan10.lib.wgp import GenerateVideo
elif os.environ.get('LTX','False') != 'False':
    from plan10.lib.ltx import GenerateVideo
else:
    from plan10.lib.image_to_video import GenerateVideo


WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

output = './outputs_qwen'
count = 0
duration = 5

AUDIO_SYSTEM_PROMPT = """
Convert this visual scene into a comma-separated list of diegetic sound effects only.
Rules:
- NO music, score, mood words, or camera directions
- ONLY physical sounds present in the scene
- Keep it under 15 words
- Output format: "sound1, sound2, sound3"
"""

### Checkout woosh repo, add dependencies and uncomment
'''
sys.path.append('./Woosh')
from woosh.inference.flowmatching_sampler import flowmatching_integrate
from woosh.components.base import LoadConfig
from woosh.model.video_kontext import VideoKontext
from woosh.utils.video import SynchformerProcessor
from woosh.utils.videoio import extract_video_frames, remux_video


def translate_to_audio_prompt(visual_prompt):
    if not visual_prompt: return ""
    return llm_analyze_media('', visual_prompt, AUDIO_SYSTEM_PROMPT)["analysis"]


def add_audio(video_path, prompt):
    description = ''
    if prompt:
        description = translate_to_audio_prompt(prompt)

    fps = 16

    # Pick device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    COMPONENT_PATH = "checkpoints/Woosh-VFlow-8s"
    ldm = VideoKontext(LoadConfig(path=COMPONENT_PATH))
    ldm = ldm.eval().to(device)

    # Video feature extractor
    featuresModel = SynchformerProcessor(frame_rate=fps).eval().to(device)

    # Latent length based on duration
    latent_len = int(duration * 100) + 1
    noise = torch.randn(1, 128, latent_len).to(device)

    with torch.inference_mode():
        # Extract frames
        video_frames, video_rate, pts_arr = extract_video_frames(
            video_path,
            start_time=0,
            end_time=duration,
        )
        video_frames = video_frames.to(device)

        # Video features
        features = featuresModel(video_frames, video_rate)

        # Conditioning
        cond = ldm.get_cond(
            {
                "audio": None,
                "description": [description],
                "synch_out": features["synch_out"],
            },
            no_dropout=True,
            device=device,
        )

        # Generate audio
                # Generate audio
        start_time = time.perf_counter()
        x_fake, steps = flowmatching_integrate(
            ldm,
            noise=noise,
            cond=cond,
            cfg=2.0 if description else 4.5,
            atol=1e-3,
            rtol=1e-3,
            return_steps=True,
            device=device,
            dtype=torch.float32 if device == "mps" else torch.float64,
        )
        audio_fake = ldm.autoencoder.inverse(x_fake)
        audio_fake = audio_fake.cpu().squeeze()  # Shape: [T]

        # 🔒 Force exact length
        target_samples = int(duration * 48000)
        current = audio_fake.shape[-1]
        if current < target_samples:
            audio_fake = torch.nn.functional.pad(audio_fake, (0, target_samples - current), value=0.0)
        elif current > target_samples:
            audio_fake = audio_fake[:target_samples]

        print(f"Integrating finished in {steps + 1} steps")

        # Normalize (safe division)
        max_abs = torch.max(torch.abs(audio_fake))
        if max_abs > 1e-6:
            audio_fake = audio_fake / max_abs

        # Shape must be [1, T] for torchaudio & remux_video
        audio_2d = audio_fake.unsqueeze(0)  # [1, T]

        suffix = "1" if description else "2"

        # Define output paths
        audio_out = video_path.replace('.mp4',f'_{suffix}.wav')
        video_out = video_path.replace('.mp4',f'_audio_{suffix}.mp4')

        video_out = video_path.replace('.mp4',f'_audio.mp4')

        # Save audio
        #torchaudio.save(audio_out, audio_2d, sample_rate=48000)

        # Remux (pass 2D so remux_video can read shape[1])
        remux_video(
            output_path=video_out,
            video_path=video_path,
            audio_input=audio_2d,
            sample_rate=48000,
            audio_start=0,
            duration_seconds=duration,
        )
'''
with open('MovieGenVideoBench.txt', 'r') as mov:
    res = [(WIDTH,HEIGHT)]
    for prompt in [x for x in mov]:
        if not prompt.strip():
            continue
        count += 1
        p = Path(f'{output}/{count}_prompt.txt')
        if p.exists():
            continue
        Path(f'{output}/{count}_prompt.txt').write_text(prompt)
        this_prompt = prompt.strip()
        for w, h in res:
            t2v = f'{output}/{count}_{w}_{h}_T2V.mp4'
            i2v = f'{output}/{count}_{w}_{h}_I2V'
            print(GenerateImage(this_prompt, f'{i2v}.png', w, h, -1))
            print(GenerateVideo(this_prompt, f'{i2v}.png', f'{i2v}.mp4', duration, w, h, -1))
            #add_audio(f'{i2v}.mp4', this_prompt)
            #add_audio(f'{i2v}.mp4', '')
