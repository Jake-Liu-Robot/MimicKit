#!/bin/bash
# Batch 2: Multi-skill & task experiments (4 GPUs parallel, ~5h)
# Run from MimicKit/ directory: bash scripts/run_batch2.sh

set -e
cd "$(dirname "$0")/.."

MAX_SAMPLES=500000000
NUM_ENVS=4096
ENGINE=data/engines/isaac_gym_engine.yaml
SEED=42

echo "=== Batch 2: Starting 4 experiments in parallel ==="
echo "Max samples per experiment: $MAX_SAMPLES"
echo "Start time: $(date)"

# Exp5a: DM × diverse motions (walk+spinkick+dance) — DM multi-skill baseline
CUDA_VISIBLE_DEVICES=0 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp5a_dm_diverse.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp5a_dm_diverse \
    > output/exp5a_dm_diverse.log 2>&1 &
PID1=$!
echo "Exp5a (DM diverse) started on GPU 0, PID=$PID1"

# Exp5b: ASE × diverse motions (same dataset) — latent-conditioned multi-skill
CUDA_VISIBLE_DEVICES=1 python mimickit/run.py \
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
    > output/exp5b_ase_diverse.log 2>&1 &
PID2=$!
echo "Exp5b (ASE diverse) started on GPU 1, PID=$PID2"

# Exp6: AMP + steering (locomotion) — gait switching emergence
CUDA_VISIBLE_DEVICES=2 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/amp_steering_humanoid_env.yaml \
    --agent_config data/agents/amp_task_humanoid_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp6_amp_steer \
    > output/exp6_amp_steer.log 2>&1 &
PID3=$!
echo "Exp6 (AMP steering) started on GPU 2, PID=$PID3"

# Exp8: AMP + location (locomotion) — same prior, different task
CUDA_VISIBLE_DEVICES=3 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/amp_location_humanoid_env.yaml \
    --agent_config data/agents/amp_task_humanoid_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp8_amp_location \
    > output/exp8_amp_location.log 2>&1 &
PID4=$!
echo "Exp8 (AMP location) started on GPU 3, PID=$PID4"

echo ""
echo "=== All 4 experiments running. Waiting for completion... ==="
echo "Monitor with: tail -f output/exp*_*.log"
echo "TensorBoard: tensorboard --logdir=output --port=6006 --bind_all"
echo ""

wait $PID1 && echo "Exp5a (DM) finished" || echo "Exp5a (DM) FAILED (exit $?)"
wait $PID2 && echo "Exp5b (ASE) finished" || echo "Exp5b (ASE) FAILED (exit $?)"
wait $PID3 && echo "Exp6 finished" || echo "Exp6 FAILED (exit $?)"
wait $PID4 && echo "Exp8 finished" || echo "Exp8 FAILED (exit $?)"

echo ""
echo "=== Batch 2 complete: $(date) ==="
