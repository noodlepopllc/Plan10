import os, json

CONFIG_FILE = "config.json"
WGP = "False"
LTX = "DISTILLED"
MMH3 = "False"
WIDTH = "768"
HEIGHT = "448"
VRAM = "14"

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
    "WIDTH": "768",
    "HEIGHT": "448",
    "SEED": "42",
    "TRANSFORMERS_CACHE":"$HF_HOME",
    "MMH3": "False",
    "WGP": "False",
    "LTX": "DISTILLED",
    "ANIME": "False",
    "VERBOSE": "False",

}

def load_config(update=False, config=cfg):
    if update:
        with open(CONFIG_FILE, 'w') as c:
            json.dump(config, c, indent=4)
    elif os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: config.update(json.load(f))
        except: print(f'{CONFIG_FILE} is missing or broken')
    else:
        with open(CONFIG_FILE, 'w') as c:
            json.dump(config, c, indent=4)
    return config

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
                if isinstance(v, str):
                    os.environ[k] = v
        os.environ["LOADED"] = "True"
    cfg = load_config()
    if replace_env:
        if os.path.exists('.env'):
            os.remove('.env')
    if not os.path.exists('.env'):
        with Path('.env').open('a') as fp:
            for k, v in cfg.items():
                if isinstance(v, str):
                    fp.write(f'export {k}="{v}"\n')
            #fp.write(additional)

def setconfig(mmh3=False, ltx2=False, hires=False, sdres=False, portrait=False, wangp=False, hivram=False, lovram=False, verbose=False, anime=False, minres=False):
    cfg_original = load_config()
    tmp_cfg = cfg_original.copy()
    high_resolution = ["1280", "720"] 
    standard_resolution = ["832", "480"]
    minimal_resolution = ["768", "448"]
    tmp_cfg['ANIME'] = 'True' if anime else 'False'
    
    if wangp:
        tmp_cfg['WGP'] = 'True'
        tmp_cfg['LTX'] = 'False'
        tmp_cfg['MMH3'] = 'False'
    else:
        tmp_cfg['WGP'] = cfg_original['WGP']
    if mmh3:
        tmp_cfg['MMH3'] = 'True'
        tmp_cfg['LTX'] = 'False'
        high_resolution = ("1344", "768") 
        standard_resolution = ("864", "480")
        minimal_resolution = standard_resolution
    elif ltx2:
        tmp_cfg["LTX"] = "DISTILLED"  # Fixed: missing = operator
        tmp_cfg['MMH3'] = 'False'
        high_resolution = ["1280", "704"] 
    else:
        tmp_cfg['LTX'] = cfg_original['LTX']
        tmp_cfg['MMH3'] = cfg_original['MMH3']
    if hires:
        tmp_cfg["WIDTH"]  = high_resolution[0]
        tmp_cfg["HEIGHT"] = high_resolution[1]
    if sdres:  # Fixed: was duplicate 'if hires'
        tmp_cfg["WIDTH"]  = standard_resolution[0]
        tmp_cfg["HEIGHT"] = standard_resolution[1]
    if minres:
        tmp_cfg["WIDTH"]  = minimal_resolution[0]
        tmp_cfg["HEIGHT"] = minimal_resolution[1]
    if portrait:
        tmp_cfg['WIDTH'], tmp_cfg['HEIGHT'] = tmp_cfg['HEIGHT'], tmp_cfg['WIDTH']
    if hivram:
        tmp_cfg["VRAM"] = 64
    if lovram:
        tmp_cfg["VRAM"] = 14
    if verbose:
        tmp_cfg['VERBOSE'] = 'True'
    else:
        tmp_cfg['VERBOSE'] = 'False'

    # Fixed: Return None if NOTHING changed, return config if something DID change
    if cfg_original == tmp_cfg:
        return None
    
    return tmp_cfg

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--replace-env', action='store_true', help='Generate reverse-angle background (T2I)')
    parser.add_argument('-U', '--update', action='store_true', help='Update config')
    parser.add_argument('--mmh3', action='store_true', help='Minimax H3 is renderer')
    parser.add_argument('--ltx2', action='store_true', help='LTX2.3 distilled is renderer')
    parser.add_argument('--wangp', action='store_true', help='Use wangp MCP server as renderer')
    parser.add_argument('--hires', action='store_true', help='Set to hires mode')
    parser.add_argument('--sdres', action='store_true', help='Set to standard res mode')
    parser.add_argument('--minres', action='store_true', help='Set to lowest resolution')
    parser.add_argument('--portrait', action='store_true', help='Set to portrait mode')
    parser.add_argument('--hivram', action='store_true', help='Set vram mode to high vram')
    parser.add_argument('--lovram', action='store_true', help='Set vram mode to low vram')
    parser.add_argument('--verbose', action='store_true', help='Make wangp output more verbose, useful when models aren\'t downloaded to get status')
    parser.add_argument('--anime', action='store_true', help='Set to anime mode')
    args = parser.parse_args()


    if args.update and update := setconfig(args.mmh3, args.ltx2, args.hires, args.sdres, args.portrait, args.wangp, args.hivram, args.lovram, args.verbose, args.anime, args.minres):
        load_config(True, update)

    load_environ(args.replace_env)

if __name__ == '__main__':
    main()
