#!/bin/bash
# Test all trained models (serial, single GPU)
# Run from MimicKit/ directory: bash scripts/run_tests.sh

set -e
cd "$(dirname "$0")/.."

ENGINE=data/engines/isaac_gym_engine.yaml
NUM_ENVS=4096
TEST_EPS=32

echo "=== Running tests on all trained models ==="
echo "Start time: $(date)"

# Exp1: DM walk
echo "--- Testing Exp1 (DM walk) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp1_dm_walk.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp1_dm_walk/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp1_dm_walk/test_results.txt

# Exp2: DM spinkick
echo "--- Testing Exp2 (DM spinkick) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp2_dm_spinkick.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp2_dm_spinkick/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp2_dm_spinkick/test_results.txt

# Exp3: AMP walk
echo "--- Testing Exp3 (AMP walk) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp3_amp_walk.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp3_amp_walk/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp3_amp_walk/test_results.txt

# Exp4: AMP spinkick
echo "--- Testing Exp4 (AMP spinkick) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp4_amp_spinkick.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp4_amp_spinkick/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp4_amp_spinkick/test_results.txt

# Exp-A: DM spinkick no pose termination
echo "--- Testing Exp-A (DM spinkick no pose term) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/expa_dm_spinkick_no_pose_term.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/expa_dm_spinkick_no_pose_term/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/expa_dm_spinkick_no_pose_term/test_results.txt

# Exp5a: DM diverse
echo "--- Testing Exp5a (DM diverse) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp5a_dm_diverse.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp5a_dm_diverse/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp5a_dm_diverse/test_results.txt

# Exp5c: AMP diverse
echo "--- Testing Exp5c (AMP diverse) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp5c_amp_diverse.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp5c_amp_diverse/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp5c_amp_diverse/test_results.txt

# Exp6: AMP steering
echo "--- Testing Exp6 (AMP steering) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/amp_steering_humanoid_env.yaml \
    --agent_config data/agents/amp_task_humanoid_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp6_amp_steer/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp6_amp_steer/test_results.txt

# Exp5b: ASE diverse
echo "--- Testing Exp5b (ASE diverse) ---"
python mimickit/run.py --mode test --num_envs $NUM_ENVS \
    --engine_config $ENGINE \
    --env_config data/envs/exp5b_ase_diverse.yaml \
    --agent_config data/agents/ase_humanoid_agent.yaml \
    --devices cuda:0 --visualize false \
    --model_file output/exp5b_ase_diverse/model.pt \
    --test_episodes $TEST_EPS \
    2>&1 | tee output/exp5b_ase_diverse/test_results.txt

echo ""
echo "=== All tests complete: $(date) ==="
