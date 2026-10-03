# Multi-joint impedance relief priority in a feedforward bipedal locomotion policy

Code, trained checkpoints and data for:

> Obaru K. *Ankle-Inclusive Relief Is Necessary but Not Sufficient to Restore Gait Stability Under
> Multi-Joint Impedance Perturbations in a Feedforward Bipedal Locomotion Policy.* Frontiers in Robotics and AI (under review).

A trained, unmodified Unitree G1 velocity-tracking policy (mjlab 1.2.0, eight training seeds) is perturbed at
the hip pitch, knee and ankle pitch of one leg with (1) a symmetric actuator-damping increase (contracture-like)
or (2) a direction-selective, velocity-dependent resistance, and every subset of joints is then "relieved"
(restored to nominal) to measure which relief restores stable gait.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/g1_spasticity_revision_experiments.ipynb` | All experiments (Google Colab, NVIDIA T4). Sections: diagnostics, verification, experiments AB (symmetric relief), C (single-joint dose), D1 (direction-selective dose), D2 (direction-selective relief), aggregation, Fig. S1 |
| `analysis/make_tables_figures.py` | Regenerates every table and figure of the manuscript from `results/raw` |
| `analysis/make_supplementary_xlsx.py` | Builds `results/Supplementary_Data_Sheet_1.xlsx` (Tables S1-S7) |
| `results/raw/*.csv` | One row per rollout (8 policies x 4 speeds x 10 rollouts per job) with fall outcome and kinematic/actuator measures; column `kd` is the resistance gain _b_ of the direction-selective model in the manuscript. One truncated line left by an interrupted write in `expC_single_joint_dose.csv` was removed; the affected job had been rerun in full, so no rollout is missing or duplicated |
| `results/tables`, `results/figures` | Generated outputs |
| `checkpoints/` | Trained policies for seeds 1-8 (see `checkpoints/README.md`) |
| `notebooks/original_submission/` | Notebook of the originally submitted version, kept for transparency (see note below) |

## Reproducing

```bash
# tables, figures and key numbers from the raw data (CPU, seconds)
pip install pandas numpy scipy matplotlib openpyxl
python analysis/make_tables_figures.py
python analysis/make_supplementary_xlsx.py
```

Tables were generated with Python 3, pandas, numpy, scipy 1.17.1 and matplotlib; the bootstrap random seed is fixed in the script (20261002), so confidence intervals are reproducible. Exact Wilcoxon P values (multiples of 1/128 with eight policies) require a scipy version that uses the exact distribution for n = 8.

To re-run the simulations, open the notebook in Google Colab with a GPU runtime, point it to the
checkpoints, and run the sections in order. Each job (one policy x one condition, 40 parallel
environments) takes roughly 3-10 s on a T4.

## Training (summary)

The eight policies are the same checkpoints used in the companion studies. They were trained with
`mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1` (mjlab 1.2.0, rsl-rl-lib 5.0.1) with 2,048 parallel
environments for 3,000 PPO iterations (final checkpoint `model_2999.pt`), seeds 1-8 (in mjlab 1.2.0 the
training seed sets both the PPO agent and the training environment, so network initialization, exploration
noise and environment randomization all differ between seeds), initial heading fixed at -90 deg, velocity commands U(-1, 1) m/s forward and lateral and U(-0.5, 0.5) rad/s yaw (the command curriculum, which begins at iteration 5,000, was not reached), on an NVIDIA Tesla T4 (about 2.2 h per policy). The training notebook is
published at https://github.com/rehabiliworld1-max/fall-threshold-unilateral-attenuation
(`notebooks/g1_training_cross_seed.ipynb`).

## Evaluation protocol (summary)

* 250 policy steps (50 Hz) = 5.0 s; fall = base tilt > 70 deg (mjlab `bad_orientation`).
* Constant forward command 0.3/0.5/0.7/0.9 m/s; command resampling, heading control and standing commands disabled; observation noise and random pushes disabled.
* Initial state: joint noise U(-0.05, 0.05) rad, base x/y U(-0.5, 0.5) m, height offset U(0.01, 0.05) m; startup randomization of foot friction, encoder bias, center of mass; evaluation environment seed 100 (set directly in the evaluation notebook).
* Direction-selective torque is written to the GPU-side `sim.data.qfrc_applied` at every physics step (0.005 s) and clipped to the actuator torque limit.
* GPU physics is not bit-for-bit reproducible across executions; probabilities are estimated from 10 rollouts per policy and condition (see Table S7 for a replication check).

## Note on the original submission

In the originally submitted version, the direction-selective torque was written to `sim.mj_data`, a CPU-side
copy that the GPU physics (MuJoCo Warp) never reads, so that resistance was never applied. The corrected implementation writes to the GPU-side `sim.data.qfrc_applied` at every physics step; applying +20 N·m at the right knee of a standing policy (seed 1, evaluation environment seed 100, zero velocity command) changed the mean knee angle over the last 1.0 s of a 2.0 s run from 30.7 deg to 52.7 deg, and the applied force read back from the GPU state was 20.0 N·m (`results/raw/verification_torque_injection.csv`). All results in the
revised manuscript come from the corrected notebook in `notebooks/`.

## License

MIT (see `LICENSE`).
