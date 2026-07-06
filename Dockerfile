# -----------------------------------------------------------------------------
# STEP 1: OFFICIAL ANACONDA BASE INFRASTRUCTURE
# -----------------------------------------------------------------------------
# This gives you a native Python 3.12 environment with zero PEP 668 system locks.
FROM continuumio/anaconda3:latest

# Force instant log streaming to your terminal console window
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install baseline Linux systems utilities needed for your repos.
# We skip ffmpeg here because Conda will manage it natively!
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# STEP 2: CONDA ENVIRONMENT & NATIVE CUDA TOOLKIT INSTALLATION
# -----------------------------------------------------------------------------
WORKDIR /app

# Tell Conda to create your custom environment, lock in Python 3.12, 
# and pull the explicit CUDA toolkit dependencies straight from NVIDIA's channel!
# We clean the conda cache in the SAME STEP to keep the image footprint optimized.
RUN conda create -n plan10 python=3.12 "ffmpeg=6.1.1" cuda-toolkit -c nvidia -c conda-forge -y && \
    conda clean --all -f -y

# Prepend the newly created environment to the container's master PATH string.
# This forces the container to permanently use your plan10 conda env by default!
ENV PATH="/opt/conda/envs/plan10/bin:${PATH}"

# -----------------------------------------------------------------------------
# STEP 3: HIGH-SPEED NATIVE TORCH PIP INSTALLATION
# -----------------------------------------------------------------------------
RUN pip install --upgrade pip && pip install huggingface_hub[cli]

# Explicitly pull down PyTorch binaries matched for Python 3.12 and CUDA 13.x
RUN pip install --no-cache-dir torch torchvision torchaudio torchcodec
# -----------------------------------------------------------------------------
# STEP 4: PROJECT REQUIREMENTS & VERIFIED DIFFSYNTH-STUDIO
# -----------------------------------------------------------------------------
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Clones and installs DiffSynth-Studio directly from the correct source URL
RUN git clone https://github.com/modelscope/DiffSynth-Studio.git && \
    pip3 install -e DiffSynth-Studio

# -----------------------------------------------------------------------------
# STEP 5: CONFIGURATION-MATCHED MODEL DOWNLOADS
# -----------------------------------------------------------------------------
RUN mkdir -p /app/models /app/loras

# 1. Download LoRAs straight to your dedicated sibling root-level /app/loras directory
RUN hf download lightx2v/Qwen-Image-2512-Lightning Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors --local-dir /app/loras && \
    hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors --local-dir /app/loras && \
    hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA qwen-image-edit-2511-multiple-angles-lora.safetensors --local-dir /app/loras && \
    hf download DeepBeepMeep/Wan2.1 loras_accelerators/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors --local-dir /app/loras && \
    hf download lightx2v/Wan2.1-Distill-Loras wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors --local-dir /app/loras

# 2. FIXED: Explicitly target the model repository structure mapping natively.
# By forcing the destination into a specific folder block matching your config structure,
# your DiffSynth-Studio framework maps paths flawlessly.
RUN hf download noodlepop/Wan-Series-Converted-Safetensors --local-dir /app/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors

# -----------------------------------------------------------------------------
# STEP 6: ENTRYPOINT CONFIGURATION (Bound directly to bin/bot.py)
# -----------------------------------------------------------------------------
# Copy the rest of your local codebase into the container
COPY . /app

RUN python lib/config.py
ENV HF_HOME=/app/.cache

# Routes the container execution trigger straight to your script engine
ENTRYPOINT ["python", "bin/bot.py", "-F"]
CMD ["alice.txt"]
