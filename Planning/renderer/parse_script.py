import re
import json
from pathlib import Path


def load_registry(registry_path):
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_zone_lookup(registry):
    lookup = {}
    for loc in registry.get("locations", registry.get("locations ", [])):
        loc_name = (loc.get("name") or loc.get("name ", "")).strip()
        for zone in loc.get("zones", registry.get("zones ", [])):
            zname = (zone.get("zone_name") or zone.get("zone_name ", "")).strip()
            if not zname:
                continue
            slug = f"{loc_name}__{zname}".upper().replace(" ", "_")
            lookup[zname] = {"location": loc_name, "zone_key": slug}
    return lookup


def load_character_names(registry):
    names = []
    for bio in registry.get("biographies", registry.get("biographies ", [])):
        raw = bio.get("name", bio.get("name ", ""))
        name = raw.strip()
        if name:
            names.append(name)
    return names


def build_name_normalizer(names):
    sorted_names = sorted(names, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted_names) + r")\b",
        re.IGNORECASE,
    )

    def normalize(text):
        if not text:
            return text
        return pattern.sub(lambda m: m.group(1).upper(), text)

    return normalize


def parse_script_txt(script_path, registry_path):
    registry = load_registry(registry_path)
    zone_lookup = build_zone_lookup(registry)
    char_names = load_character_names(registry)
    normalize = build_name_normalizer(char_names)

    upper_names = set(n.upper() for n in char_names)

    name_alt = "|".join(
        re.escape(n) for n in sorted(upper_names, key=len, reverse=True)
    )
    char_re = re.compile(rf"^({name_alt})\s*(?:\(([^)]+)\))?\s*$")
    zone_re = re.compile(r"\[ZONE:\s*(.+?)\]")
    dialog_re = re.compile(r'^"(.*)"$')

    raw = Path(script_path).read_text(encoding="utf-8")
    lines = [ln.rstrip() for ln in raw.split("\n")]

    beats = []
    beat = None
    current_char = None

    for line in lines:
        s = line.strip()
        if not s:
            continue

        m = zone_re.match(s)
        if m:
            if beat:
                beats.append(beat)
            zone_name = m.group(1).strip()
            zone_info = zone_lookup.get(zone_name, {})
            beat = {
                "zone": zone_name,
                "location": zone_info.get("location"),
                "zone_key": zone_info.get("zone_key"),
                "summary": None,
                "active_characters": [],
                "passive_characters": [],
            }
            current_char = None
            continue

        if s.startswith(">>"):
            if beat:
                beat["summary"] = s[2:].strip()
            continue

        m = char_re.match(s)
        if m and beat:
            current_char = {
                "name": m.group(1).strip(),
                "delivery": m.group(2).strip() if m.group(2) else None,
                "dialog": None,
                "action": None,
            }
            beat["active_characters"].append(current_char)
            continue

        m = dialog_re.match(s)
        if m and current_char:
            current_char["dialog"] = m.group(1).strip()
            continue

        if current_char and not current_char["action"]:
            current_char["action"] = normalize(s)
            continue

    if beat:
        beats.append(beat)

    # ── Passive character detection ──────────────────────────────
    for beat in beats:
        if beat["summary"]:
            beat["summary"] = normalize(beat["summary"])

        active_set = {c["name"] for c in beat["active_characters"]}
        passive_by_name = {}

        # 1. Scan summary (no "mentioned_by" — it's stage direction)
        if beat["summary"]:
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in beat["summary"]:
                    passive_by_name[scan_name] = {
                        "name": scan_name,
                        "source": "summary",
                    }

        # 2. Scan dialog lines
        for char in beat["active_characters"]:
            if not char["dialog"]:
                continue
            dialog_upper = char["dialog"].upper()
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in dialog_upper:
                    existing = passive_by_name.get(scan_name)
                    if existing is None or existing["source"] == "summary":
                        passive_by_name[scan_name] = {
                            "name": scan_name,
                            "source": "dialog",
                            "mentioned_by": char["name"],
                        }

        # 3. Scan action lines
        for char in beat["active_characters"]:
            if not char["action"]:
                continue
            action_upper = char["action"].upper()
            for scan_name in upper_names:
                if scan_name in active_set:
                    continue
                if scan_name in action_upper:
                    existing = passive_by_name.get(scan_name)
                    if existing is None or existing["source"] == "summary":
                        passive_by_name[scan_name] = {
                            "name": scan_name,
                            "source": "action",
                            "mentioned_by": char["name"],
                        }

        beat["passive_characters"] = list(passive_by_name.values())

    return beats


def main():
    import sys
    script = sys.argv[1]
    registry = sys.argv[2] if len(sys.argv) > 2 else "registry.json"
    print(json.dumps(parse_script_txt(script, registry), indent=2))


if __name__ == "__main__":
    main()