# Plan10
Planning and execution of commands to generate images/storyboards
```
git clone https://github.com/noodlepopllc/Plan10.git
cd Plan10
conda create python=3.12 -n plan10
conda activate plan10
pip install -r requirements.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git 
pip install -e DiffSynth-Studio
mkdir loras
hf download lightx2v/Qwen-Image-2512-Lightning Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors --local-dir ./loras
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors --local-dir ./loras
hf download lightx2v/Wan2.1-Distill-Loras wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors --local-dir ./loras
```
