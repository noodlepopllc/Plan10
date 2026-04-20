import os, json

CONFIG_FILE = "config.json"

def load_config():
    cfg = {"VRAM": "14", "QWEN": "Qwen/Qwen3.5-4B", "TRANSFORMERS_OFFLINE": "0", "HF_HUB_OFFLINE": "0", "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface", "DIFFSYNTH_SKIP_DOWNLOAD": "False", "BITSNBYTES":"False"}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: cfg.update(json.load(f))
        except: print(f'{CONFIG_FILE} is missing or broken')
    else:
        with open('config.json', 'w') as c:
            json.dump(cfg, c, indent=4)
    return cfg

def load_environ():
    if "LOADED" not in os.environ:
        # If need something from modelscope, try international first, much faster
        # os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
        cfg = load_config()
        for k, v in cfg.items():
            os.environ[k] = v
        os.environ["LOADED"] = "True"
