#!/bin/bash
# Re-evaluate tracking error for all experiments after fixing the AMP/ASE
# tracking error bug (stale _ref_* states in headless TEST mode).
#
# No retraining needed — the fix only affects tracking error computation
# in TEST mode, not the training reward/policy.
#
# Uses the saved config copies from each experiment's output/ directory
# to guarantee identical configs to the original training run.
#
# Usage:
#   bash scripts/run_tracking_error_test.sh              # run all experiments
#   bash scripts/run_tracking_error_test.sh exp3 exp4    # run specific experiments

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

NUM_ENVS=4096
TEST_EPISODES=4096
SEEDS="42 123 456"
DEVICE="cuda:0"

# ── auto-discover experiments from output/ ──────────────────────────────
ALL_EXPERIMENTS=()
for d in "$PROJ_DIR"/output/exp*/; do
  name=$(basename "$d")
  model="$d/model.pt"
  env_cfg="$d/env_config.yaml"
  agent_cfg="$d/agent_config.yaml"
  engine_cfg="$d/engine_config.yaml"
  if [ -f "$model" ] && [ -f "$env_cfg" ] && [ -f "$agent_cfg" ] && [ -f "$engine_cfg" ]; then
    ALL_EXPERIMENTS+=("$name")
  fi
done

# ── filter by command-line args if provided ─────────────────────────────
if [ $# -gt 0 ]; then
  FILTER=("$@")
  EXPERIMENTS=()
  for name in "${ALL_EXPERIMENTS[@]}"; do
    for f in "${FILTER[@]}"; do
      if [[ "$name" == *"$f"* ]]; then
        EXPERIMENTS+=("$name")
        break
      fi
    done
  done
else
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

# ── output setup ────────────────────────────────────────────────────────
OUT_DIR="${PROJ_DIR}/output/tracking_error_reeval"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo " Tracking Error Re-evaluation"
echo " Experiments: ${EXPERIMENTS[*]}"
echo " Test episodes: $TEST_EPISODES per seed"
echo " Seeds: $SEEDS"
echo " Num envs: $NUM_ENVS"
echo " Output: $OUT_DIR"
echo "=============================================="
echo ""

# ── run each experiment ─────────────────────────────────────────────────
for NAME in "${EXPERIMENTS[@]}"; do
  EXP_DIR="$PROJ_DIR/output/$NAME"

  echo "----------------------------------------------"
  echo " Running: $NAME"
  echo "  env_config:    $EXP_DIR/env_config.yaml"
  echo "  agent_config:  $EXP_DIR/agent_config.yaml"
  echo "  engine_config: $EXP_DIR/engine_config.yaml"
  echo "  model_file:    $EXP_DIR/model.pt"
  echo "----------------------------------------------"

  python mimickit/eval_tracking_error.py \
    --env_config "$EXP_DIR/env_config.yaml" \
    --agent_config "$EXP_DIR/agent_config.yaml" \
    --engine_config "$EXP_DIR/engine_config.yaml" \
    --model_file "$EXP_DIR/model.pt" \
    --num_envs "$NUM_ENVS" \
    --test_episodes "$TEST_EPISODES" \
    --rand_seeds $SEEDS \
    --devices "$DEVICE" \
    --out_file "$OUT_DIR/${NAME}.json" \
    2>&1 | tee "$OUT_DIR/${NAME}.log"

  echo ""
done

# ── generate summary table ──────────────────────────────────────────────
echo "=============================================="
echo " Summary"
echo "=============================================="

python -c "
import json, glob, os

out_dir = '$OUT_DIR'
files = sorted(glob.glob(os.path.join(out_dir, '*.json')))

keys = ['root_pos_err', 'root_rot_err', 'body_pos_err', 'body_rot_err',
        'dof_vel_err', 'root_vel_err', 'root_ang_vel_err', 'mean_return', 'mean_ep_len']

header = '{:<35s}'.format('experiment') + ''.join('{:>22s}'.format(k) for k in keys) + '{:>15s}'.format('total_eps')
print(header)
print('-' * len(header))

rows = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    with open(f) as fp:
        data = json.load(fp)
    n_seeds = data.get('num_seeds', 1)
    total_eps = data.get('total_episodes', 'N/A')
    row = '{:<35s}'.format(name)
    for k in keys:
        mean = data.get(k + '_mean', data.get(k, float('nan')))
        std = data.get(k + '_std', None)
        if std is not None and n_seeds > 1:
            row += '{:>12.6f}+-{:<8.6f}'.format(mean, std)
        else:
            row += '{:>22.6f}'.format(mean)
    row += '{:>15s}'.format(str(total_eps))
    print(row)
    csv_row = [name]
    for k in keys:
        csv_row.append(data.get(k + '_mean', data.get(k, '')))
        csv_row.append(data.get(k + '_std', ''))
    csv_row.append(total_eps)
    rows.append(csv_row)

csv_keys = []
for k in keys:
    csv_keys += [k + '_mean', k + '_std']
csv_keys.append('total_episodes')

csv_path = os.path.join(out_dir, 'summary.csv')
with open(csv_path, 'w') as fp:
    fp.write(','.join(['experiment'] + csv_keys) + '\n')
    for r in rows:
        fp.write(','.join(str(x) for x in r) + '\n')
print()
print('CSV saved to: ' + csv_path)
"

echo ""
echo "Done. Results in: $OUT_DIR"
