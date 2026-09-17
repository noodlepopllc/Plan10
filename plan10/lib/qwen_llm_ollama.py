import os
import base64
import json
import re
import requests
from pathlib import Path
from plan10.lib.config import load_environ
import random

load_environ()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:latest")  # Match your pulled model name
SEED = os.environ.get("SEED","-1")

SEED = random.randint(0,1000000) if SEED == "-1" else int(SEED)  
THINKING = os.environ.get("THINKING", "False") != "False"

def _system_prompt(fn="system/bot.txt"):
    if not os.path.exists(fn):
        repo_root = Path(__file__).parent.parent
        fn = repo_root / "system" / Path(fn).name
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

def dummy_request():
    import time, traceback
    payload = {
        "model": "qwen3.8-next:xs",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }

    start = time.time()
    print(f"Starting at {time.strftime('%X')}...")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", 
            json=payload, 
            timeout=300,
            proxies={"http": None, "https": None}
        )
        elapsed = time.time() - start
        print(f"✅ Success in {elapsed:.2f}s")
        print(response.json()['message']['content'])
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED after {elapsed:.2f}s")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

def _call_ollama(messages, max_tokens=8192, temperature=0.5, top_p=0.9, tools=None, thinking=THINKING):
    ollama_messages = _normalize_for_ollama(messages)
    
    # Map max_tokens to num_predict
    if max_tokens <= 4096:
        num_predict = 512
    elif max_tokens <= 8192:
        num_predict = 2048
    elif max_tokens <= 16384:
        num_predict = 4096
    else:
        num_predict = 8192

    payload = {
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "stream": False,
        "keep_alive": "1m",
        "options": {
            "num_ctx": max_tokens,  # ← Output + room for input
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
            "seed": SEED
        }
    }
    
    # Set thinking parameters correctly
    if '3.8' in OLLAMA_MODEL and thinking:
        payload['think'] = "low"
        payload['options']["preserve_thinking"] = True
    
    if 'mtp' in OLLAMA_MODEL:
        payload['options']['draft_num_predict'] = 1 if thinking else 2 
    
    if tools:
        payload["tools"] = tools

    # Retry up to 3 times
    for attempt in range(3):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat", 
                json=payload, 
                timeout=(10, 600),
                proxies={"http": None, "https": None}
            )
            
            # Handle 400 errors immediately
            if response.status_code == 400:
                print("❌ Ollama 400 Error Response:", response.text)
                raise ValueError(f"Bad request to Ollama: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 500 and attempt < 2:  # ← Retry on attempts 0 and 1
                print(f"Warning: Attempt {attempt + 1} failed (USB I/O bottleneck). Warming cache and retrying...")
                time.sleep(0.5)
                continue
            else:
                raise
        except Exception as e:
            raise

# ─────────────────────────────────────────
# 1) Agent / tools chat
# ─────────────────────────────────────────
def llm_chat(messages, tools=None, max_tokens=8192, temperature=0.7, enable_thinking=THINKING):
    sys_msg = next(_system_prompt_gen)
    full_messages = sys_msg + messages

    # Note: Ollama doesn't have a native toggle for thinking models.
    # If enable_thinking=False, the model may still output <think> tags depending on the Modelfile.
    res = _call_ollama(full_messages, max_tokens, temperature, top_p=0.9, tools=tools, thinking=enable_thinking)

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
def llm_analyze_media(media, prompt="Describe this.", system=None, max_tokens=8192, temperature=0.1):
    from plan10.lib.util import video_to_img

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

    res = _call_ollama(messages, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
    output_text = res.get("message", {}).get("content", "").strip()

    return {"status": "success", "analysis": output_text}

