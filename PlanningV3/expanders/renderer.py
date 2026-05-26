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
            print("\n=== REMAINING ASSETS ===")
            for alias, a in remaining.items():
                print(alias, "depends on", a["alias_used"])

            missing = []
            for alias, a in remaining.items():
                for dep in a["alias_used"]:
                    if dep not in built and dep not in remaining:
                        missing.append((alias, dep))

            print("\n=== MISSING DEPENDENCIES ===")
            for alias, dep in missing:
                print(f"{alias} depends on missing alias {dep}")

            raise RuntimeError("Cyclic or missing dependencies")


        for alias in ready:
            a = remaining.pop(alias)
            print(f">> ALIAS: {a['alias']}\n{a['instruction']}, Width: {WIDTH}, Height: {HEIGHT} Seed: {SEED}\n")
            built.add(alias)


if __name__ == '__main__':
    import sys, json
    basepath = sys.argv[1]
    with open(f'{basepath}/assets{sys.argv[2]}.json') as ass:
        assets = json.load(ass)

    build(assets)
