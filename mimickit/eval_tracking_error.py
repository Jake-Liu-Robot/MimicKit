"""Re-evaluate tracking error for trained models.

After fixing the AMP/ASE tracking error bug (stale _ref_* in headless TEST
mode), run this script to get corrected numbers.  No retraining needed.

Usage:
    # single experiment (3 seeds × 4096 episodes)
    python mimickit/eval_tracking_error.py \
        --env_config  data/envs/exp3_amp_walk.yaml \
        --agent_config output/exp3_amp_walk/agent_config.yaml \
        --engine_config data/engines/isaac_gym_engine.yaml \
        --model_file output/exp3_amp_walk/model.pt \
        --num_envs 4096 --test_episodes 4096 --rand_seeds 42 123 456

    # batch mode via shell wrapper
    bash scripts/run_tracking_error_test.sh
"""

import json
import numpy as np
import os
import sys
import time

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch


TRACKING_KEYS = [
    "root_pos_err", "root_rot_err", "body_pos_err", "body_rot_err",
    "dof_vel_err", "root_vel_err", "root_ang_vel_err"
]


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, "Failed to load args from: " + arg_file

    return args


def collect_results(env, test_info, seed):
    """Collect tracking errors and test info into a results dict."""
    diag = env.record_diagnostics()

    results = {}
    for k in TRACKING_KEYS:
        if k in diag:
            v = diag[k]
            if torch.is_tensor(v):
                v = v.item()
            results[k] = v

    results["mean_return"] = test_info["mean_return"]
    results["mean_ep_len"] = test_info["mean_ep_len"]
    results["num_eps"] = test_info["num_eps"]
    results["seed"] = seed

    return results


def eval_tracking_error(args):
    num_envs = args.parse_int("num_envs", 4096)
    test_episodes = args.parse_int("test_episodes", 4096)
    model_file = args.parse_string("model_file", "")
    device = args.parse_strings("devices", ["cuda:0"])[0]
    seeds = args.parse_ints("rand_seeds", [42, 123, 456])
    out_file = args.parse_string("out_file", "")

    assert model_file != "", "--model_file is required"

    # initialize once
    mp_util.init(0, 1, device, None)
    util.set_rand_seed(seeds[0])

    # build env and agent once
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    env = env_builder.build_env(env_file, engine_file, num_envs, device,
                                visualize=False, record_video=False)

    # disable pose termination for fair comparison across methods
    env._pose_termination = False

    agent_file = args.parse_string("agent_config")
    agent = agent_builder.build_agent(agent_file, env, device)
    agent.load(model_file)

    Logger.print("Evaluating: {}".format(model_file))
    Logger.print("  num_envs={}, test_episodes={}, seeds={}".format(
        num_envs, test_episodes, seeds))

    # run test for each seed
    all_results = []
    for seed in seeds:
        util.set_rand_seed(seed)

        test_info = agent.test_model(num_episodes=test_episodes)
        r = collect_results(env, test_info, seed)
        all_results.append(r)

        Logger.print("")
        Logger.print("===== Results (seed={}) =====".format(seed))
        Logger.print("Mean Return:         {:.4f}".format(r["mean_return"]))
        Logger.print("Mean Episode Length:  {:.4f}".format(r["mean_ep_len"]))
        Logger.print("Episodes:            {}".format(r["num_eps"]))
        Logger.print("")
        Logger.print("--- Tracking Errors ---")
        for k in TRACKING_KEYS:
            if k in r:
                Logger.print("  {:<20s} {:.6f}".format(k, r[k]))
            else:
                Logger.print("  {:<20s} N/A".format(k))

    # compute mean +/- std across seeds
    all_keys = TRACKING_KEYS + ["mean_return", "mean_ep_len"]
    summary = {"seeds": seeds, "num_seeds": len(seeds), "per_seed": all_results}

    Logger.print("")
    Logger.print("===== Aggregated ({} seeds) =====".format(len(seeds)))
    for k in all_keys:
        vals = [r[k] for r in all_results if k in r]
        if len(vals) > 0:
            mean = np.mean(vals)
            std = np.std(vals)
            summary[k + "_mean"] = float(mean)
            summary[k + "_std"] = float(std)
            Logger.print("  {:<20s} {:.6f} +/- {:.6f}".format(k, mean, std))

    summary["total_episodes"] = sum(r["num_eps"] for r in all_results)

    # save to json if requested
    if out_file != "":
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        Logger.print("\nSaved to: {}".format(out_file))

    return summary


if __name__ == "__main__":
    eval_tracking_error(load_args(sys.argv))
