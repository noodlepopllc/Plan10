#!/bin/bash
set -euo pipefail

python lib/graphics_gen.py -P "$(python lib/qwen_llm.py -P "generate an anime image prompt for a princess captured and chained in an old dungeon of an abandoned castle by an evil wizard, her clothes are in tatters as she looks sad and defeated, create an entire manga page spread 6 panels, complete with message bubbles of a conversation between the wizard and the princess" | tr -d '\n')" -O manga.png

