# qwen_llm.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from plan10.lib.config import load_environ

load_environ()

# Read at module import time
BACKEND = os.environ.get("LLM_BACKEND", "transformers").lower().strip()
THINKING = os.environ.get("THINKING","False") != "False"

if BACKEND == "ollama":
    from plan10.lib.qwen_llm_ollama import (
        llm_chat,
        llm_analyze_media
    )
elif BACKEND == "transformers":

    import gc, json, re, torch
    from pathlib import Path
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig

    def get_bnb_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    class LLMContext:
        def __init__(self):
            self.processor = None
            self.model = None

        def __enter__(self):
            # 1. Clear out memory before allocating fresh weights
            gc.collect()
            torch.cuda.empty_cache()

            # 2. Initialize and load model/processor
            self.processor = AutoProcessor.from_pretrained(os.environ["QWEN"])

            # Note: Replace 'Qwen3_5ForConditionalGeneration' with your exact imported model class
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                os.environ["QWEN"],
                torch_dtype=torch.float16,
                quantization_config=get_bnb_config() if os.environ["BITSNBYTES"] == "True" else None,
                device_map="cuda:0",
                trust_remote_code=True
            )

            self.model.eval()
            
            # This returns the tuple to the 'as' variable in the 'with' block
            return self.processor, self.model

        def __exit__(self, exc_type, exc_val, exc_tb):
            # This block ALWAYS runs, even if model.generate() crashes
            print("[Memory Guard] Unloading model from memory and purging VRAM...")
            
            if self.model:
                self.model.to('cpu')
                del self.model
                self.model = None
                
            if self.processor:
                del self.processor
                self.processor = None

            # Force aggressive garbage collection and clear CUDA allocations
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Returning False ensures any underlying exceptions are raised normally 
            # instead of being silently swallowed.
            return False



    def _system_prompt(fn="system/bot.txt"):
        if not os.path.exists(fn):
            repo_root = Path(__file__).parent.parent
            fn = repo_root / "system" / Path(fn).name
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
        if model:
            model.to('cpu')
            del model
        if processor:
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

    def llm_analyze_media(media, prompt="Describe this.", system=None, max_tokens=1024, temperature=0.1, processor=None, model=None):
        from plan10.lib.util import video_to_img
        import torch

        image = None
        if os.path.exists(media):
            image = video_to_img(media)
            
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        
        if image is not None:
            ndx = 1 if system else 0
            messages[ndx]['content'].append({"type": "image", "image": image})
        
        # Inner logic execution
        def _execute_inference(p_instance, m_instance):
            inputs = p_instance.apply_chat_template(
                messages,
                tokenize=True,          
                add_generation_prompt=True,
                return_dict=True,       
                return_tensors="pt",     
                enable_thinking=False
            )
            inputs = inputs.to(m_instance.device)
            
            with torch.no_grad():
                generated_ids = m_instance.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature, 
                    top_p=0.9, 
                    do_sample=True, 
                    pad_token_id=p_instance.tokenizer.eos_token_id
                )
            
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            output_text = p_instance.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()

        # Dynamic Lifecycle Fork:
        if processor is not None and model is not None:
            # Scenario A: Bypasses model management entirely, uses your shared instance
            analysis_text = _execute_inference(processor, model)
        else:
            # Scenario B: Manages own memory context automatically
            with LLMContext() as (local_processor, local_model):
                analysis_text = _execute_inference(local_processor, local_model)
                
        return {"status": "success", "analysis": analysis_text}
else:
    raise ValueError(
        f"Invalid LLM_BACKEND='{BACKEND}'. Must be 'transformers' or 'ollama'."
    )

# Optional: log which backend is active
print(f"🤖 [qwen_llm] Active backend: {BACKEND.upper()}")

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('-M','--media', type=str, default=None, help='media to analyze')
    parser.add_argument('-P', '--prompt', type=str, default='describe this image', help='prompt')
    parser.add_argument('-O', '--output', type=str, default=None, help='optionally save to file')
    parser.add_argument('-S', '--system', type=str, default='None', help='path to system prompt')
    parser.add_argument('-T', '--max-tokens', type=int, default=1024, help='max number of tokens')
    args = parser.parse_args()
    system_prompt = None
    if Path(args.system).exists():
        system_prompt = Path(args.system).read_text().strip()
    max_tokens = 4096 if system_prompt else 1024
    max_tokens = max(max_tokens,args.max_tokens)
    out = llm_analyze_media(args.media if args.media else '', args.prompt, system_prompt, max_tokens=max_tokens)['analysis']
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(out)
    else:
        print(out)

if __name__ == '__main__':
    main()

