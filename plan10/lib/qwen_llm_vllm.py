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
    enable_thinking=THINKING  # e.g., "low", "medium", "xhigh", or "False"
):
    sys_msg = next(_system_prompt_gen)
    kwargs = {
        "model": VLLM_MODEL,
        "messages": sys_msg + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": SEED,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    #
    # Qwen 3.8 Native Reasoning Configurations
    #
    if enable_thinking and enable_thinking != "False":
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": enable_thinking  # Passes "low", "medium", or "xhigh"
            }
        }
    else:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
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
    media=None,              
    prompt="Describe this.",
    system=None,
    max_tokens=8192,
    temperature=0.1,
    processor=None,
    model=None,
    enable_thinking=THINKING  
):
    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    user_content = []

    if media:
        # Enforce strict PNG validation if media is passed
        media_path = Path(media)
        if media_path.suffix.lower() != ".png":
            raise ValueError(f"Unsupported file format: '{media_path.suffix}'. Only .png files are accepted.")
            
        image_b64 = _encode_image(media)
        user_content.append({
            "type": "image_url",
            "image_url": {
                # Updated MIME type from image/jpeg to image/png
                "url": f"data:image/png;base64,{image_b64}"
            }
        })

    # Always append the text prompt
    user_content.append({
        "type": "text",
        "text": prompt
    })

    messages.append({
        "role": "user",
        "content": user_content
    })

    # Prepare standard request options
    kwargs = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": SEED,
    }

    # Qwen Reasoning Settings
    if enable_thinking and enable_thinking != "False":
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": enable_thinking  
            }
        }
    else:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    response = client.chat.completions.create(**kwargs)
    msg = response.choices.message
    raw_content = msg.content or ""

    # Parse out thinking tags if present
    thinking_content, clean_content = _strip_thinking(raw_content)

    return {
        "status": "success",
        "thinking": thinking_content,
        "analysis": clean_content
    }
