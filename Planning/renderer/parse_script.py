import re
from pathlib import Path

def parse_script_txt(path):
    """
    Correct parser: groups fragments by ZONE anchors.
    One beat = one ZONE + all following fragments until next ZONE.
    """

    raw = Path(path).read_text()

    lines = [ln.rstrip() for ln in raw.split("\n")]

    beats = []
    beat = None
    current_char = None

    zone_re = re.compile(r'\[ZONE:\s*(.+?)\]')
    char_re = re.compile(r'([A-Z][A-Z0-9_]*)\s*\(([^)]+)\)')
    dialog_re = re.compile(r'"(.*)"')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # -------------------------
        # ZONE = start new beat
        # -------------------------
        m = zone_re.match(line)
        if m:
            # finalize previous beat
            if beat:
                beats.append(beat)

            beat = {
                "zone": m.group(1).strip(),
                "summary": None,
                "characters": []
            }
            current_char = None
            continue

        # -------------------------
        # SUMMARY
        # -------------------------
        if line.startswith(">>"):
            if beat:
                beat["summary"] = line[2:].strip()
            continue

        # -------------------------
        # CHARACTER HEADER
        # -------------------------
        m = char_re.match(line)
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
        # DIALOG
        # -------------------------
        m = dialog_re.match(line)
        if m and current_char:
            current_char["dialog"] = m.group(1).strip()
            continue

        # -------------------------
        # ACTION
        # -------------------------
        if current_char:
            current_char["action"] = line
            continue

    # finalize last beat
    if beat:
        beats.append(beat)

    return beats


def main():
    import sys, json
    print(json.dumps(parse_script_txt(sys.argv[1]), indent=4))

if __name__ == '__main__':
    main()
