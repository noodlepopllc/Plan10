import os, json

CONFIG_FILE = "config.json"

def load_config():
    cfg = {
        "VRAM": "14", 
        "QWEN": "Qwen/Qwen3.5-4B", 
        "TRANSFORMERS_OFFLINE": "0", 
        "HF_HUB_OFFLINE": "0", 
        "HF_HOME": "./models", 
        "WAN21": "14B", # or 1.3B
        "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface", 
        "DIFFSYNTH_SKIP_DOWNLOAD": "False", 
        "BITSNBYTES":"False",
        "BATCH":"True", 
        "LLM_BACKEND": "transformers",
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen3.5:latest",
        "IMAGE_GEN": "KLEIN",
        "IMAGE_EDIT": "KLEIN",
        "WIDTH": "832",
        "HEIGHT": "480",
        "SEED": "122333",
        "TRANSFORMERS_CACHE":"$HF_HOME",
        "WGP": "False",
        "LTX": "False",
        "ANIME": "False"

    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: cfg.update(json.load(f))
        except: print(f'{CONFIG_FILE} is missing or broken')
    else:
        with open('config.json', 'w') as c:
            json.dump(cfg, c, indent=4)
    return cfg

additional = '''# 1. CRITICAL: Prevents PyTorch from hoarding memory and fragmentation crashes
export TORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
'''

def load_environ(replace_env=False):
    from pathlib import Path
    if "LOADED" not in os.environ:
        # If need something from modelscope, try international first, much faster
        # os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
        cfg = load_config()
        for k, v in cfg.items():
            if k not in os.environ:
                os.environ[k] = v
        os.environ["LOADED"] = "True"
    cfg = load_config()
    if replace_env:
        if os.path.exists('.env'):
            os.remove('.env')
    if not os.path.exists('.env'):
        with Path('.env').open('a') as fp:
            for k, v in cfg.items():
                fp.write(f'export {k}="{v}"\n')
            fp.write(additional)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--replace-env', action='store_true', help='Generate reverse-angle background (T2I)')
    args = parser.parse_args()
    load_environ(args.replace_env)
    print("config.json has been created.")
