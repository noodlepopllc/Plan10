# -----------------------------------------------------------------------------
# STEP 1: BASE INFRASTRUCTURE & UV INSTALLATION
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (official Astral script)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# -----------------------------------------------------------------------------
# STEP 2: REPOSITORY CLONE & ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
WORKDIR /app

# Clone and immediately step into the directory for clean relative paths
RUN git clone https://github.com/noodlepopllc/Plan10.git
WORKDIR /app/Plan10

# Ensure entrypoint is executable
RUN chmod +x bot.sh

# Install the package in editable mode directly into the system/conda environment
RUN uv pip install --system -e .

# Run the uv command app (and lib/config.py if still needed separately)
RUN uv run config -R
# RUN uv run python lib/config.py  # Uncomment if this is still a distinct required step

# Set Hugging Face cache directory (relative to /app)
ENV HF_HOME=/app/.cache

# -----------------------------------------------------------------------------
# STEP 3: EXECUTION
# -----------------------------------------------------------------------------
# Paths are now relative to /app/Plan10
ENTRYPOINT ["./bot.sh"]
CMD ["alice.txt"]