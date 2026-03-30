#!/bin/bash
# Batch 3: ASE multi-skill (1 GPU, ~4h)
# Run from MimicKit/ directory: bash scripts/run_batch3.sh
# Can run in parallel with Batch 1 or 2 on a free GPU

set -e
cd "$(dirname "$0")/.."

MAX_SAMPLES=500000000
NUM_ENVS=4096
ENGINE=data/engines/isaac_gym_engine.yaml
SEED=42

echo "=== Batch 3: ASE diverse ==="
echo "Max samples: $MAX_SAMPLES"
echo "Start time: $(date)"

# Exp5b: ASE × diverse motions — latent-conditioned multi-skill
# Same dataset as Exp5a (DM) and Exp5c (AMP) for 3-way comparison
# Note: ASE normalizer_samples=500M, may need >500M for full convergence
CUDA_VISIBLE_DEVICES=0 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp5b_ase_diverse.yaml \
    --agent_config data/agents/ase_humanoid_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp5b_ase_diverse \
    2>&1 | tee output/exp5b_ase_diverse.log

echo ""
echo "=== Batch 3 complete: $(date) ==="
