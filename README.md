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
hf download DeepBeepMeep/Wan2.1 Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors --local-dir ./loras
hf download lightx2v/Wan2.1-Distill-Loras wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors --local-dir ./loras
hf download noodlepop/Wan-Series-Converted-Safetensors --local-dir ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
python lib/config.py
```
