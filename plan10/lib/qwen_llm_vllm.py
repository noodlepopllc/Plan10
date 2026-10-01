import os
import base64
import json
import re
import requests
from pathlib import Path
from plan10.lib.config import load_environ
import random
import time
from openai import OpenAI

load_environ()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
# Reusing Ollama names for seamless config file compatibility
VLLM_URL = os.environ.get("OLLAMA_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("OLLAMA_MODEL", "Qwen/Qwen3.8-27B-FP8")  
SEED = os.environ.get("SEED", "-1")

SEED = random.randint(0, 1000000) if SEED == "-1" else int(SEED)  
THINKING = os.environ.get("THINKING", "False")

client = OpenAI(
    api_key="dummy",
    base_url=VLLM_URL
)

class LLMContext:
    def __init__(self):
        self.processor = None
        self.model = None

    def __enter__(self):
        return self.processor, self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

def _system_prompt(fn="system/bot.txt"):
    if not os.path.exists(fn):
        repo_root = Path(__file__).parent.parent
        fn = repo_root / "system" / Path(fn).name
    prompt = Path(fn).read_text().strip()
    while True:
        yield [{"role": "system", "content": prompt}]

_system_prompt_gen = _system_prompt()

def _strip_thinking(raw: str):
    if not raw:
        return "", ""
    m = re.search(
        r"<think>(.*?)</think>",
        raw,
        flags=re.DOTALL) # Fixed syntax typo
    if m:
        thinking = m.group(1).strip()
        response = raw.replace(m.group(0), "").strip()
        return thinking, response
    return "", raw.strip()

def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────
# 1) Agent / tools chat
# ─────────────────────────────────────────
def llm_chat(
    messages,
    tools=None,
    max_tokens=8192,
    temperature=0.7,
    enable_thinking=THINKING
):
    sys_msg = next(_system_prompt_gen)
    kwargs = {
        "model": VLLM_MODEL,
        "messages": sys_msg + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message # Fixed index reference
    raw_content = msg.content or ""

    thinking_content, clean_content = _strip_thinking(raw_content)

    return {
        "status": "success",
        "thinking": thinking_content,
        "response_clean": clean_content,
        "tool_calls": getattr(msg, "tool_calls", None)
    }

# ─────────────────────────────────────────
# 2) Media analysis
# ─────────────────────────────────────────
def llm_analyze_media(
    media,
    prompt="Describe this.",
    system=None,
    max_tokens=8192,
    temperature=0.1,
    processor=None,
    model=None
):
    image_b64 = _encode_image(media)

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    })

    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return {
        "status": "success",
        "analysis": response.choices[0].message.content # Fixed index reference
    }
