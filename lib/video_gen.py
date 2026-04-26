import torch, gc, os
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from qwen_llm import llm_analyze_media
from image_analysis import AnalyzeImage
from config import load_environ
from PIL import Image
from PIL.PngImagePlugin import PngInfo

load_environ()

model = "Wan-AI/Wan2.1-T2V-1.3B"

enhancer = '''You are a cinematic prompt optimizer. Your task is to expand short user inputs into clear, stable, film‑language prompts inspired by late‑1960s to early‑1970s Panavision 60 cinematography.

Requirements:
1. Add essential cinematography details: lens (40–75mm anamorphic), shot size (wide, medium, close), camera angle (eye‑level, low), and FOV.
2. Add period‑accurate lighting: warm tungsten or late‑afternoon sun, soft contrast, mild halation, gentle film grain.
3. Add simple natural actions using direct verbs (stand, walk, turn, look).
4. Add environmental cues appropriate to the scene without inventing unrelated objects or characters.
5. Keep the tone grounded, realistic, and consistent with 60s–70s American cinema. No modern digital aesthetics.
6. Output in English, 40–60 words, concise and cinematic.
7. Do not add style labels unless the user specifies one; otherwise default to a neutral 1970s film look.

Directly output the rewritten prompt.

''' 

class VideoGen(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": "cpu",
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=model, origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
                ModelConfig(model_id=model, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
                ModelConfig(model_id=model, origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
            ],
            tokenizer_config=ModelConfig(model_id=model, origin_file_pattern="google/umt5-xxl/"),
            vram_limit=vrlimit,
        )
        self.pipe.load_lora(self.pipe.dit, './loras/loras_accelerators/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors', alpha=1.0)

    def generate(self, prompt, output, width, height, seed, duration_sec):
        eprompt = prompt #llm_analyze_media('', prompt, enhancer)['analysis']
        total_frames = (duration_sec * 16) + 1
        video = self.pipe(
                prompt=eprompt,
                seed=seed,
                num_inference_steps=6,
                cfg_scale=1.0,
                height=height,
                width=width,
                num_frames=total_frames
            )
        save_video(video, output, fps=15, quality=5)
        description = ''
            
        # Post-processing
        if os.environ['BATCH'] == 'False':
            tmp_img = video_to_img(output, width, height)
            tmp_img.save('tmp.png')
            description = AnalyzeImage('tmp.png', "Briefly describe this image, no more than 100 words")['analysis']
        
        return {
            "status": "success",
            "output_path": output,
            "frames": len(video),
            "description": description,
            "prompt": prompt
        }


    def __del__(self):
        del self.pipe
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog='GenerateImages',
                    description='Generate images from prompt',
                    epilog='')
    parser.add_argument('-W', '--width', type=int, default=512, help='width of output')
    parser.add_argument('-H', '--height', type=int, default=412, help='height of output')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-P', '--prompt', type=str, default='a beautiful woman tanning at the beach', help='prompt')
    parser.add_argument('-O', '--output', type=str, default='output.mp4')
    parser.add_argument('-D', '--duration', type=int, default=5)
    args = parser.parse_args()
    vgen = VideoGen()
    print(vgen.generate(args.prompt, args.output, args.width, args.height, args.seed, args.duration))