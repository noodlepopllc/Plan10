import re
from pathlib import Path

def parse_script_txt(path):
    """
    Parse the new script.txt format into structured dicts.
    Returns: list of beats, each beat is a dict with:
      zone, summary, characters[]
    """

    raw = Path(path).read_text()

    # Split blocks by double newlines between beats
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    beats = []

    for block in blocks:
        lines = block.split("\n")

        beat = {
            "zone": None,
            "summary": None,
            "characters": []
        }

        current_char = None

        for line in lines:
            line = line.strip()

            # -------------------------
            # ZONE
            # -------------------------
            m = re.match(r'

\[ZONE:\s*(.+?)\]

', line)
            if m:
                beat["zone"] = m.group(1).strip()
                continue

            # -------------------------
            # SUMMARY
            # -------------------------
            if line.startswith(">>"):
                beat["summary"] = line[2:].strip()
                continue

            # -------------------------
            # CHARACTER HEADER
            # Format: NAME (delivery)
            # -------------------------
            m = re.match(r'([A-Z][A-Z0-9_]*)\s*\(([^)]+)\)', line)
            if m:
                name = m.group(1).strip()
                delivery = m.group(2).strip()

                current_char = {
                    "name": name,
                    "delivery": delivery,
                    "dialog": None,
                    "action": None
                }
                beat["characters"].append(current_char)
                continue

            # -------------------------
            # DIALOG LINE
            # -------------------------
            m = re.match(r'"(.*)"', line)
            if m and current_char:
                current_char["dialog"] = m.group(1).strip()
                continue

            # -------------------------
            # ACTION LINE
            # -------------------------
            if current_char and line and not line.startswith(">>"):
                # Action lines have no quotes
                current_char["action"] = line.strip()
                continue

        beats.append(beat)

    return beats

def main():
    import sys, json
    print(json.dumps(parse_script_txt(sys.argv[1]), indent=4))

if __name__ == '__main__':
    main()
