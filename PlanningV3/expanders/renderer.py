import os, sys
sys.path.append('./lib')
from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

def build(assets):
    built = set()
    remaining = {a["alias"]: a for a in assets}

    while remaining:
        # find all assets whose deps are satisfied
        ready = [
            alias for alias, a in remaining.items()
            if all(dep in built for dep in a["alias_used"])
        ]

        if not ready:
            print(remaining.items())
            raise RuntimeError("Cyclic or missing dependencies")

        for alias in ready:
            a = remaining.pop(alias)
            print(f">> ALIAS: {a['alias']}\n{a['alias_used']},{a['instruction']}, Width: {WIDTH}, Height: {HEIGHT} Seed: {SEED}\n")
            built.add(alias)


if __name__ == '__main__':
    import sys, json
    basepath = sys.argv[1]
    with open(f'{basepath}/assets{sys.argv[2]}.json') as ass:
        assets = json.load(ass)

    build(assets)
