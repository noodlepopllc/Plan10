import os
import base64
import json
import re
import requests
from pathlib import Path
from config import load_environ

load_environ()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:latest")  # Match your pulled model name

def _system_prompt(fn="system/bot.txt"):
    prompt = Path(fn).read_text().strip()
    while True:
        yield [{"role": "system", "content": prompt}]

_system_prompt_gen = _system_prompt()

def _strip_thinking(raw: str):
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = raw.replace(m.group(0), "").strip()
        return thinking, response
    return "", raw.strip()

def _encode_image(image_data):
    """Convert PIL Image, numpy array, or file path to base64 string for Ollama."""
    import io
    from PIL import Image
    if isinstance(image_data, (str, Path)):
        with open(image_data, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if hasattr(image_data, "save"):  # PIL.Image
        buf = io.BytesIO()
        image_data.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    if hasattr(image_data, "astype"):  # numpy array
        img = Image.fromarray(image_data.astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None

import json
import requests

def _normalize_for_ollama(messages):
    """Convert OpenAI-style messages to Ollama native format."""
    normalized = []
    for msg in messages:
        content = msg.get("content", "")
        images = []
        
        # Handle OpenAI multimodal list format
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") in ("image_url", "image"):
                    img = item.get("image_url", {}).get("url", item.get("image", ""))
                    # Strip data URI prefix if present
                    if img.startswith("data:image"):
                        img = img.split(",", 1)[1]
                    images.append(img)
            content = " ".join(text_parts)
        
        msg_dict = {"role": msg["role"], "content": content}
        if images:
            msg_dict["images"] = images
        normalized.append(msg_dict)
    return normalized

def _call_ollama(messages, max_tokens=8192, temperature=0.7, top_p=0.9, tools=None, thinking=False):
    # ✅ Convert to Ollama's expected format
    ollama_messages = _normalize_for_ollama(messages)
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "stream": False,
        "think": thinking,
        "keep_alive": "1m",
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
    }
    if tools:
        payload["tools"] = tools

    # 🔍 Debug: uncomment to see exactly what Ollama receives
    # print(json.dumps(payload, indent=2))
    
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    
    if response.status_code == 400:
        print("❌ Ollama 400 Error Response:", response.text)
        raise ValueError(f"Bad request to Ollama: {response.text}")
        
    response.raise_for_status()
    return response.json()

# ─────────────────────────────────────────
# 1) Agent / tools chat
# ─────────────────────────────────────────
def llm_chat(messages, tools=None, max_tokens=8192, temperature=0.7, enable_thinking=True):
    sys_msg = next(_system_prompt_gen)
    full_messages = sys_msg + messages

    # Note: Ollama doesn't have a native toggle for thinking models.
    # If enable_thinking=False, the model may still output <think> tags depending on the Modelfile.
    res = _call_ollama(full_messages, max_tokens, temperature, top_p=0.9, tools=tools)

    assistant_msg = res.get("message", {})
    raw_content = assistant_msg.get("content", "")
    tool_calls = assistant_msg.get("tool_calls", [])

    thinking, response_clean = _strip_thinking(raw_content)

    return {
        "status": "success",
        "thinking": thinking,
        "response_clean": response_clean,
        "tool_calls": tool_calls if tool_calls else None
    }

# ─────────────────────────────────────────
# 2) Media analysis
# ─────────────────────────────────────────
def llm_analyze_media(media, prompt="Describe this.", system=None, max_tokens=1024):
    from util import video_to_img

    image = None
    if os.path.exists(media):
        image = video_to_img(media)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    user_content = [{"type": "text", "text": prompt}]
    if image is not None:
        b64_img = _encode_image(image)
        if b64_img:
            user_content.insert(0, {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

    messages.append({"role": "user", "content": user_content})

    res = _call_ollama(messages, max_tokens=max_tokens, temperature=0.1, top_p=0.9)
    output_text = res.get("message", {}).get("content", "").strip()

    return {"status": "success", "analysis": output_text}

