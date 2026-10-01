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
raw_url = os.environ.get("OLLAMA_URL", "http://localhost:8000")

# Safely append /v1 if the environment variable omitted it
if not raw_url.endswith("/v1") and not raw_url.endswith("/v1/"):
    VLLM_URL = f"{raw_url.rstrip('/')}/v1"
else:
    VLLM_URL = raw_url

VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.8-27B-FP8")

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
    
    # ⚠️ vLLM / OpenAI compliance: Intercept incoming history.
    # If the last step was a tool execution, ensure the required OpenAI schema properties exist.
    sanitized_messages = []
    for msg in (sys_msg + messages):
        msg_copy = msg.copy()
        if msg_copy.get("role") == "tool" and "tool_call_id" not in msg_copy:
            # Inject a mock validation ID to keep vLLM happy without exposing it to the brain
            msg_copy["tool_call_id"] = "call_auto_generated_id"
        sanitized_messages.append(msg_copy)

    kwargs = {
        "model": VLLM_MODEL,
        "messages": sanitized_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": SEED,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # Qwen 3.8 Reasoning Parameters nested safely
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

    # Execute request against vLLM server
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    raw_content = msg.content or ""

    thinking_content, clean_content = _strip_thinking(raw_content)

    # ⚠️ Translation Layer: Standardize tool schema output to match what your brain expects.
    formatted_tool_calls = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            # Translate native OpenAI tool block to your uniform dict structure
            formatted_tool_calls.append({
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments  # Keeps string/dict format intact
                }
            })

    # Returns the exact object layout your agent loop is built to expect
    return {
        "status": "success",
        "thinking": thinking_content,
        "response_clean": clean_content,
        "tool_calls": formatted_tool_calls if formatted_tool_calls else None
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
    
    # Fixed: Access the first element [0] of the choices list
    msg = response.choices[0].message
    raw_content = msg.content or ""

    # Parse out thinking tags if present
    thinking_content, clean_content = _strip_thinking(raw_content)

    return {
        "status": "success",
        "thinking": thinking_content,
        "analysis": clean_content
    }

