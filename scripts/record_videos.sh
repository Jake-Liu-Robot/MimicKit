#!/bin/bash
# Record one episode video for each experiment.
# Run from MimicKit/ directory: bash scripts/record_videos.sh
# Videos saved to output/videos/

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

OUT_DIR="output/videos"
mkdir -p "$OUT_DIR"

DEVICE="cuda:0"

# Episodes per experiment: multi-skill and task experiments need more episodes
# to show different motions / latents / commands
get_episodes() {
  case "$1" in
    exp5a*|exp5b*|exp5c*) echo 6 ;;
    exp6*) echo 3 ;;
    *) echo 1 ;;
  esac
}

# Discover all experiments
EXPERIMENTS=()
for d in "$PROJ_DIR"/output/exp*/; do
  name=$(basename "$d")
  if [ -f "$d/model.pt" ] && [ -f "$d/env_config.yaml" ] && [ -f "$d/agent_config.yaml" ] && [ -f "$d/engine_config.yaml" ]; then
    EXPERIMENTS+=("$name")
  fi
done

echo "=============================================="
echo " Recording videos"
echo " Experiments: ${EXPERIMENTS[*]}"
echo " Output: $OUT_DIR"
echo "=============================================="
echo ""

for NAME in "${EXPERIMENTS[@]}"; do
  EXP_DIR="$PROJ_DIR/output/$NAME"
  EPISODES=$(get_episodes "$NAME")
  OUT_FILE="$OUT_DIR/${NAME}.mp4"

  echo "--- Recording: $NAME ($EPISODES episodes) ---"

  python mimickit/record_video.py \
    --env_config "$EXP_DIR/env_config.yaml" \
    --agent_config "$EXP_DIR/agent_config.yaml" \
    --engine_config "$EXP_DIR/engine_config.yaml" \
    --model_file "$EXP_DIR/model.pt" \
    --test_episodes "$EPISODES" \
    --devices "$DEVICE" \
    --out_file "$OUT_FILE"

  echo ""
done

echo "=============================================="
echo " Done. Videos in: $OUT_DIR"
echo "=============================================="
ls -lh "$OUT_DIR"/*.mp4 2>/dev/null
