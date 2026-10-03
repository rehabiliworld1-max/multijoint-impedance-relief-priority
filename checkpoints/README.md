# Trained policy checkpoints

This folder contains the eight trained policies (training seeds 1-8) used in all experiments. They are the
same checkpoints used in the companion studies (final PPO iteration, `model_2999.pt`):

    checkpoints/g1_velocity_seed1_model_2999.pt
    ...
    checkpoints/g1_velocity_seed8_model_2999.pt

The evaluation notebook loads checkpoints from the mjlab/RSL-RL run-directory layout
(`find_checkpoint(seed)` takes the highest-numbered `model_*.pt` in the most recent
`logs/rsl_rl/g1_velocity/*_seed<seed>` directory). Recreate that layout from the repository root before
running the notebook:

```bash
for s in 1 2 3 4 5 6 7 8; do
  mkdir -p logs/rsl_rl/g1_velocity/g1_seed${s}
  cp checkpoints/g1_velocity_seed${s}_model_2999.pt logs/rsl_rl/g1_velocity/g1_seed${s}/model_2999.pt
done
```
