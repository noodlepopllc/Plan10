# qwen_llm.py
import os, gc, json, re, torch
from pathlib import Path
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
from config import load_environ
load_environ()


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

def _system_prompt(fn="system/bot.txt"):
    prompt = Path(fn).read_text()
    while prompt:
        yield [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    return None

_system_prompt_gen = _system_prompt()

def _strip_thinking(raw: str):
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = raw.replace(m.group(0), "").strip()
        return thinking, response
    return "", raw.strip()

def _load_llm():
    processor = AutoProcessor.from_pretrained(os.environ["QWEN"])

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        os.environ["QWEN"],
        torch_dtype=torch.float16,
        quantization_config=get_bnb_config() if os.environ["BITSNBYTES"] == "True" else None,
        device_map="cuda:0",
        trust_remote_code=True
    )

    model.eval()
    return processor, model

def _unload_llm(model, processor):
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ─────────────────────────────────────────
# 1) Agent / tools chat (text, tools, optional thinking)
# ─────────────────────────────────────────
def llm_chat(messages, tools=None, max_tokens=8192, temperature=0.7, enable_thinking=True):
    messages = next(_system_prompt_gen) + messages
    processor, model = _load_llm()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,          # ← REQUIRED for Qwen-VL
        tools=tools,
        enable_thinking=enable_thinking
    )

    # Move all tensors to the model's device
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(model.device)


    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    trimmed = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()

    _unload_llm(model, processor)

    thinking, response_clean = _strip_thinking(text)
    return {"status": "success", "thinking": thinking, "response_clean": response_clean}

# ─────────────────────────────────────────
# 2) Media analysis / prompt enhancement
# ─────────────────────────────────────────

def llm_analyze_media(media, prompt="Describe this.", system=None, max_tokens=1024):
    from util import video_to_img
    import torch

    image = None
    if os.path.exists(media):
        image = video_to_img(media)
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    })
    if image is not None:
        ndx = 1 if system else 0
        messages[ndx]['content'].append({"type": "image", "image": image})
    
    processor, model = _load_llm()

    # ✅ Processor handles tokenization + image preprocessing in ONE call
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,          # ← Critical: must be True
        add_generation_prompt=True,
        return_dict=True,       # ← Returns dict with input_ids, pixel_values, etc.
        return_tensors="pt",     # ← Returns PyTorch tensors
        enable_thinking=False
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=0.1, 
            top_p=0.9, 
            do_sample=True, 
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # ✅ Trim input tokens to decode ONLY the generated response
    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    _unload_llm(model, processor)
    
    return {"status": "success", "analysis": output_text.strip()}

# ─────────────────────────────────────────
# 3) Narrative → Pipeline Generator
# ─────────────────────────────────────────
def llm_generate_pipeline(story: str, system_file: str = "system/narrative.txt", max_tokens: int = 4096, output_path: str = None):
    """Generate an alias-driven pipeline script using narrative.txt as the system prompt."""
    sys_prompt = Path(system_file).read_text().strip()
    if not sys_prompt:
        raise ValueError(f"System prompt file '{system_file}' is empty or missing.")

    processor, model = _load_llm()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": f"INPUT: {story.strip()}\nOUTPUT:"}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,      # 🔑 Strict formatting
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    trimmed = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()

    _unload_llm(model, processor)

    # Strip markdown code blocks if the model wraps the output
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.lower().startswith("text"):
            text = text[4:].strip()

    if output_path:
        Path(output_path).write_text(text)
        print(f"✅ Pipeline saved to: {output_path}")

    return text






