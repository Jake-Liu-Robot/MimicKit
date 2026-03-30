# MimicKit - Codebase Guide

All project info is in `README.md`. This file contains only code conventions for AI assistance.

## Code Conventions

- All configs are YAML. Args can be CLI or `--arg_file`.
- Factory pattern: `env_builder.build_env()`, `agent_builder.build_agent()`, `engine_builder.build_engine()`.
- `env_name` in YAML maps to environment class. `agent_name` maps to agent class.
- Distributed training: `--devices cuda:0 cuda:1` spawns one process per device.
- `output/` is gitignored. `data/motions/` and `data/assets/` are gitignored (large binary files).

## Dependencies

- PyTorch >= 1.9.1, IsaacGym Preview 4, see `requirements.txt`
