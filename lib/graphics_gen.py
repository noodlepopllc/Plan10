from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
import torch, os, sys, gc
sys.path.append('./lib')
from image_analysis import AnalyzeImage
from config import load_environ
load_environ()

model_id = "baidu/ERNIE-Image-Turbo"

class GraphicsGen(object):
    def __init__(self,vrlimit=14):
        if "VRAM" in os.environ:
            vrlimit = int(os.environ["VRAM"])
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        self.pipe = ErnieImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device='cuda',
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
                ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/model.safetensors", **vram_config),
                ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ],
            tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="tokenizer/"),
            vram_limit=vrlimit,
        )

    def generate(self, prompt, output, width, height, seed):
        image = self.pipe(
                prompt=prompt,
                seed=seed,
                num_inference_steps=8,
                cfg_scale=1.0,
                height=height,
                width=width,
                sigma_shift=4.9
            )
        image.save(output)
        return {"status":"success", "output_path":output}


    def __del__(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

def GenerateGraphics(prompt='', output='tmp.png', width=1024, height=1024, seed=42):
    gen = GraphicsGen()
    status = gen.generate(prompt, output, int(width), int(height), int(seed))
    del gen
    status['description'] = ''
    if os.environ['BATCH'] == 'False':
        analysis = AnalyzeImage(output, "Briefly describe this image, no more than 100 words")
        status['description'] = analysis['analysis']
    status['prompt'] = prompt
    return status

RESOLUTIONS = ['1024x1024','848x1264','1264x848','768x1376','896x1200','1376x768','1200x896']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--resolution', type=str, default='1024x1024', help='resolutions')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-P', '--prompt', type=str, default='a beautiful woman tanning at the beach', help='prompt')
    parser.add_argument('-O', '--output', type=str, default='output.png')
    args = parser.parse_args()
    if args.resolution not in RESOLUTIONS:
        print('Resolution must be one of the following: ', ', '.join(RESOLUTIONS))
    width, height = args.resolution.split('x')
    print(GenerateGraphics(args.prompt, args.output, width, height, args.seed))


