"""Record one episode of each experiment as video.

Usage:
    python mimickit/record_video.py \
        --env_config output/exp1_dm_walk/env_config.yaml \
        --agent_config output/exp1_dm_walk/agent_config.yaml \
        --engine_config output/exp1_dm_walk/engine_config.yaml \
        --model_file output/exp1_dm_walk/model.pt \
        --out_file output/videos/exp1_dm_walk.mp4

    # batch mode
    bash scripts/record_videos.sh
"""

import numpy as np
import os
import sys

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])
    return args


def record_video(args):
    model_file = args.parse_string("model_file", "")
    device = args.parse_strings("devices", ["cuda:0"])[0]
    out_file = args.parse_string("out_file", "output/video.mp4")
    test_episodes = args.parse_int("test_episodes", 1)

    assert model_file != "", "--model_file is required"

    mp_util.init(0, 1, device, None)
    util.set_rand_seed(42)

    # build env with video recording enabled, 1 env
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    env = env_builder.build_env(env_file, engine_file, num_envs=1, device=device,
                                visualize=False, record_video=True)

    agent_file = args.parse_string("agent_config")
    agent = agent_builder.build_agent(agent_file, env, device)
    agent.load(model_file)

    Logger.print("Recording: {}".format(model_file))
    Logger.print("  episodes={}, out={}".format(test_episodes, out_file))

    # run test (video recording starts automatically in TEST mode)
    test_info = agent.test_model(num_episodes=test_episodes)

    Logger.print("Mean Return: {:.4f}".format(test_info["mean_return"]))
    Logger.print("Mean Episode Length: {:.4f}".format(test_info["mean_ep_len"]))

    # save video
    diag = env.record_diagnostics()
    if "sim_recording" in diag:
        video = diag["sim_recording"]
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        video.save(out_file)
        Logger.print("Video saved: {} ({} frames)".format(out_file, video.get_num_frames()))
    else:
        Logger.print("ERROR: No video recording found in diagnostics")


if __name__ == "__main__":
    record_video(load_args(sys.argv))
