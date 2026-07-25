# Plan10
Planning and execution of commands to generate images/storyboards
```
git clone https://github.com/noodlepopllc/Plan10.git
cd Plan10
conda env create -f environment.yml
conda activate plan10-local
uv pip install -e . 

uv run config

# The following are very large, test out first with existing and download as needed

# Required for Qwen models, not needed for default settings
hf download lightx2v/Qwen-Image-2512-Lightning Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors --local-dir ./loras
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors --local-dir ./loras
hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA qwen-image-edit-2511-multiple-angles-lora.safetensors --local-dir ./loras

# Required for Wan models
hf download DeepBeepMeep/Wan2.1 loras_accelerators/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors --local-dir ./loras
hf download lightx2v/Wan2.1-Distill-Loras wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors --local-dir ./loras
hf download noodlepop/Wan-Series-Converted-Safetensors --local-dir ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
```
