#!/usr/bin/env bash
#
# YouTube Shorts Batch Optimizer
# Upscales 704x1280 → 1080x1920, normalizes audio to -14 LUFS.
#
# Usage: ./optimize_shorts.sh [input_dir] [output_dir]
#

# REMOVED: set -e  (causes exit on ((count++)) when count=0)

# ============================================
# CONFIG
# ============================================
INPUT_DIR="${1:-./output_videos}"
OUTPUT_DIR="${2:-$INPUT_DIR/optimized}"
CRF="19"
PRESET="slow"
AUDIO_BITRATE="192k"
TARGET_WIDTH="720"
TARGET_HEIGHT="1280"
FPS="24"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================
# CHECKS
# ============================================
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}✗ Error: ffmpeg not found.${NC}"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Error: Input directory '$INPUT_DIR' not found.${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================
# PROCESS
# ============================================
echo -e "${YELLOW}🎬 Optimizing for YouTube Shorts...${NC}"
echo -e "Input:  $INPUT_DIR"
echo -e "Output: $OUTPUT_DIR"
echo -e "Target: ${TARGET_WIDTH}x${TARGET_HEIGHT} @ ${FPS}fps, -14 LUFS\n"

count=0
failed=0

shopt -s nullglob
files=("$INPUT_DIR"/*.mp4)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo -e "${RED}✗ No .mp4 files found in $INPUT_DIR${NC}"
    exit 0
fi

for input_path in "${files[@]}"; do
    filename=$(basename "$input_path")
    name="${filename%.*}"
    output_path="$OUTPUT_DIR/${name}_optimized.mp4"
    
    if [ -f "$output_path" ]; then
        echo -e "${YELLOW}⊘ Skipping $filename (already exists)${NC}"
        continue
    fi

    echo -e "Processing: ${filename}..."
    
    if ffmpeg -i "$input_path" \
        -vf "scale=${TARGET_WIDTH}:${TARGET_HEIGHT}:flags=lanczos,fps=${FPS},format=yuv420p" \
        -c:v libx264 \
        -crf "$CRF" \
        -preset "$PRESET" \
        -movflags +faststart \
        -c:a aac \
        -b:a "$AUDIO_BITRATE" \
        -ar 48000 \
        -af "loudnorm=I=-14:TP=-1.0:LRA=11" \
        -y \
        "$output_path" > /dev/null 2>&1; then
        
        size=$(du -h "$output_path" | cut -f1)
        echo -e "${GREEN}✓ Saved: ${name}_optimized.mp4 (${size})${NC}"
        count=$((count + 1))  # FIXED: Use $(( )) instead of (( ))
    else
        echo -e "${RED}✗ Failed: $filename${NC}"
        failed=$((failed + 1))  # FIXED: Use $(( )) instead of (( ))
    fi
done

# ============================================
# SUMMARY
# ============================================
echo -e "\n${YELLOW}================================${NC}"
echo -e "${GREEN}✓ Success: $count files optimized${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}✗ Failed: $failed files${NC}"
fi
echo -e "${YELLOW}================================${NC}"
echo -e "Ready to upload: ${OUTPUT_DIR}"
