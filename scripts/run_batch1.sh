#!/bin/bash
# Batch 1: Core DM vs AMP comparison (4 GPUs parallel, ~4h)
# Run from MimicKit/ directory: bash scripts/run_batch1.sh

set -e
cd "$(dirname "$0")/.."

MAX_SAMPLES=500000000
NUM_ENVS=4096
ENGINE=data/engines/isaac_gym_engine.yaml
SEED=42

echo "=== Batch 1: Starting 4 experiments in parallel ==="
echo "Max samples per experiment: $MAX_SAMPLES"
echo "Start time: $(date)"

# Exp1: DeepMimic × walk — tracking baseline
CUDA_VISIBLE_DEVICES=0 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp1_dm_walk.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp1_dm_walk \
    > output/exp1_dm_walk.log 2>&1 &
PID1=$!
echo "Exp1 (DM walk) started on GPU 0, PID=$PID1"

# Exp2: DeepMimic × spinkick — tracking on dynamic motion
CUDA_VISIBLE_DEVICES=1 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp2_dm_spinkick.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp2_dm_spinkick \
    > output/exp2_dm_spinkick.log 2>&1 &
PID2=$!
echo "Exp2 (DM spinkick) started on GPU 1, PID=$PID2"

# Exp3: AMP × walk — distribution matching baseline
CUDA_VISIBLE_DEVICES=2 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp3_amp_walk.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp3_amp_walk \
    > output/exp3_amp_walk.log 2>&1 &
PID3=$!
echo "Exp3 (AMP walk) started on GPU 2, PID=$PID3"

# Exp4: AMP × spinkick — mode collapse stress test
CUDA_VISIBLE_DEVICES=3 python mimickit/run.py \
    --mode train \
    --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp4_amp_spinkick.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --devices cuda:0 \
    --visualize false \
    --logger tb \
    --rand_seed $SEED \
    --max_samples $MAX_SAMPLES \
    --out_dir output/exp4_amp_spinkick \
    > output/exp4_amp_spinkick.log 2>&1 &
PID4=$!
echo "Exp4 (AMP spinkick) started on GPU 3, PID=$PID4"

echo ""
echo "=== All 4 experiments running. Waiting for completion... ==="
echo "Monitor with: tail -f output/exp*_*.log"
echo "TensorBoard: tensorboard --logdir=output --port=6006 --bind_all"
echo ""

wait $PID1 && echo "Exp1 finished" || echo "Exp1 FAILED (exit $?)"
wait $PID2 && echo "Exp2 finished" || echo "Exp2 FAILED (exit $?)"
wait $PID3 && echo "Exp3 finished" || echo "Exp3 FAILED (exit $?)"
wait $PID4 && echo "Exp4 finished" || echo "Exp4 FAILED (exit $?)"

echo ""
echo "=== Batch 1 complete: $(date) ==="
