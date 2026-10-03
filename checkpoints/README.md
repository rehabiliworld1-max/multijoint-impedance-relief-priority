Place the eight trained policy checkpoints here, one folder per training seed, preserving the
mjlab/RSL-RL run-directory layout used by the notebook:

    checkpoints/logs/rsl_rl/g1_velocity/<run_name>_seed1/model_<N>.pt
    ...
    checkpoints/logs/rsl_rl/g1_velocity/<run_name>_seed8/model_<N>.pt

The notebook's `find_checkpoint(seed)` loads the highest-numbered `model_*.pt` in the most recent
`*_seed<seed>` run directory under `logs/rsl_rl/g1_velocity/`.
