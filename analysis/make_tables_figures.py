"""Regenerate every table and figure of the revised manuscript from the raw CSVs.

Usage:  python analysis/make_tables_figures.py   (run from the repository root)

Inputs : results/raw/exp*.csv   (written by notebooks/g1_spasticity_revision_experiments.ipynb)
Outputs: results/tables/*.csv, results/figures/*.png|.tiff, results/key_numbers.json
"""
import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW, TAB, FIG = ROOT / "results/raw", ROOT / "results/tables", ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

SPEEDS = [0.3, 0.5, 0.7, 0.9]
SIDES = ["right", "left"]
COND = ["no_treatment", "treat_hip", "treat_knee", "treat_ankle", "treat_hip_knee",
        "treat_hip_ankle", "treat_knee_ankle", "treat_all"]
COND_LABEL = {"no_treatment": "Untreated", "treat_hip": "Hip", "treat_knee": "Knee", "treat_ankle": "Ankle",
              "treat_hip_knee": "Hip + knee", "treat_hip_ankle": "Hip + ankle",
              "treat_knee_ankle": "Knee + ankle", "treat_all": "All three (= unperturbed)"}
C_TARGETS = ["hip_pitch", "knee", "ankle_pitch", "equinovarus"]
C_LABEL = {"hip_pitch": "Hip pitch", "knee": "Knee", "ankle_pitch": "Ankle pitch", "equinovarus": "Ankle pitch + roll"}
C_SCALES = [1, 10, 100, 500, 1000, 2000, 5000, 10000]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
KEY = {}


def rnd(x, nd=1):
    """Round half to even on the decimal value. Fall probabilities are multiples of 1.25 percentage points, so
    halves are frequent; rounding the decimal value (not the binary float) keeps tables and text consistent."""
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_EVEN)


def load(name, key_regex=None):
    d = pd.read_csv(RAW / f"{name}.csv", low_memory=False)
    ok = d["job_key"].astype(str).str.match(r"^(right|left)_")
    d = d[ok].copy()
    d["fell"] = d["fell"].astype(str) == "True"
    d["speed"] = pd.to_numeric(d["speed"]).round(1)
    d["seed"] = d["seed"].astype(int)
    return d


def per_seed(d, keys):
    return d.groupby(keys + ["seed"])["fell"].mean().rename("p").reset_index()


def _unused_holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def seed_ci(g):
    lo = np.array([max(0.0, ci95(v)[0]) for v in g]); hi = np.array([min(1.0, ci95(v)[1]) for v in g])
    m = np.array([np.mean(v) for v in g])
    return m, np.vstack([m - lo, hi - m])


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return (x.mean(), x.mean())
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (x.mean() - h, x.mean() + h)


# ------------------------------------------------------------------ priority experiments
def priority_tables(d, tag):
    ps = per_seed(d, ["side", "condition", "speed"])
    # fall table (mean ± SD across 8 seeds)
    rows = []
    for (side, c, sp), g in ps.groupby(["side", "condition", "speed"]):
        lo, hi = ci95(g.p)
        rows.append({"side": side, "condition": c, "speed": sp, "mean": g.p.mean(), "sd": g.p.std(ddof=1),
                     "ci_lo": lo, "ci_hi": hi, "n_seeds": len(g)})
    ft = pd.DataFrame(rows)
    ft.to_csv(TAB / f"{tag}_fall_probability_long.csv", index=False)
    wide = []
    for side in SIDES:
        for c in COND:
            r = {"side": side, "condition": COND_LABEL[c]}
            for sp in SPEEDS:
                x = ft[(ft.side == side) & (ft.condition == c) & (ft.speed == sp)].iloc[0]
                r[f"{sp} m/s"] = f"{rnd(100*x['mean'])} ± {rnd(100*x['sd'])}"
            wide.append(r)
    pd.DataFrame(wide).to_csv(TAB / f"{tag}_fall_probability_table.csv", index=False)

    # Inference: hierarchical bootstrap (resample seeds, then rollouts within seed), 10,000 replicates.
    # Primary: 95% CI of the difference in fall probability vs untreated; restoration: upper CI bound of the
    # difference vs the all-joints-treated (= unperturbed) reference <= 5 percentage points.
    # Exact paired Wilcoxon p (per-seed probabilities, n = 8) is reported for reference only; with 8 seeds its
    # minimum attainable two-sided p is 0.0078, so it is not multiplicity-corrected.
    rng = np.random.default_rng(20261002)
    B = 10000
    arr = {}
    for (side, cnd, sp), g in d.groupby(["side", "condition", "speed"]):
        g = g.sort_values(["seed", "rollout"])
        arr[(side, cnd, sp)] = g["fell"].to_numpy(float).reshape(g["seed"].nunique(), -1)

    def boot_diff(a, b):
        ns, nr = a.shape
        si = rng.integers(0, ns, size=(B, ns))
        ra = rng.integers(0, nr, size=(B, ns, nr)); rb = rng.integers(0, nr, size=(B, ns, nr))
        ma = np.take_along_axis(a[si], ra, axis=2).mean(axis=(1, 2))
        mb = np.take_along_axis(b[si], rb, axis=2).mean(axis=(1, 2))
        dd = ma - mb
        return np.percentile(dd, 2.5), np.percentile(dd, 97.5)

    ps_piv = ps.pivot_table(index=["side", "speed", "seed"], columns="condition", values="p").reset_index()
    out = []
    for (side, sp), g in ps_piv.groupby(["side", "speed"]):
        for c in COND[1:]:
            a, u, ref = arr[(side, c, sp)], arr[(side, "no_treatment", sp)], arr[(side, "treat_all", sp)]
            dlt = g[c] - g["no_treatment"]
            p = stats.wilcoxon(g[c], g["no_treatment"], zero_method="zsplit").pvalue if (dlt != 0).any() else 1.0
            lo, hi = boot_diff(a, u)
            rlo, rhi = boot_diff(a, ref)
            out.append({"side": side, "speed": sp, "condition": c, "p_fall": a.mean(), "p_untreated": u.mean(),
                        "delta_vs_untreated": a.mean() - u.mean(), "delta_ci_lo": lo, "delta_ci_hi": hi,
                        "effect": "improved" if hi < 0 else ("worsened" if lo > 0 else "n.s."),
                        "seeds_improved": int((dlt < 0).sum()), "seeds_worsened": int((dlt > 0).sum()),
                        "wilcoxon_p_uncorrected": p,
                        "delta_vs_reference": a.mean() - ref.mean(), "delta_ref_ci_hi": rhi,
                        "restored": bool(rhi <= 0.05)})
    tests = pd.DataFrame(out)
    test_name = {"Table2_symmetric": "TableS3_symmetric_relief_tests", "Table3_directional": "TableS4_directional_relief_tests"}[tag]
    tests.to_csv(TAB / f"{test_name}.csv", index=False)
    return ft, tests


def heat(ax, ft, side, title):
    m = ft[ft.side == side].pivot(index="condition", columns="speed", values="mean").reindex(COND) * 100
    im = ax.imshow(m.values, vmin=0, vmax=100, cmap="Reds", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{s:.1f}" for s in SPEEDS])
    ax.set_yticks(range(len(COND))); ax.set_yticklabels([COND_LABEL[c] for c in COND])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{rnd(v, 0)}", ha="center", va="center", fontsize=7, color="white" if v > 55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Commanded speed (m/s)"); ax.set_title(title, fontsize=8.5, loc="left")
    ax.tick_params(length=0)
    return im


def main():
    ab = load("expAB_symmetric_priority"); d2 = load("expD2_directional_priority")
    d1 = load("expD1_directional_dose"); c = load("expC_single_joint_dose")
    jk = c["job_key"].str.extract(r"^(right|left)_(.+)_x(\d+)_s(\d+)$")
    c["target"], c["damping_scale"] = jk[1], jk[2].astype(int)

    ft_ab, t_ab = priority_tables(ab, "Table2_symmetric")
    ft_d2, t_d2 = priority_tables(d2, "Table3_directional")

    # ---------------- Figure 1: heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.8), sharey=True)
    for r, (ft, name) in enumerate([(ft_ab, "Symmetric damping ×10,000 at hip, knee and ankle (contracture-like)"),
                                    (ft_d2, "Direction-selective, velocity-dependent resistance at hip, knee and ankle")]):
        for k, side in enumerate(SIDES):
            im = heat(axes[r, k], ft, side, f"{chr(65 + 2*r + k)}  {side.capitalize()} leg perturbed")
        axes[r, 0].annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(-118, 26),
                            textcoords="offset points", fontsize=9, fontweight="bold", color=INK)
    fig.subplots_adjust(hspace=0.45)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02); cb.set_label("Fall probability (%)"); cb.outline.set_visible(False)
    for ext in ("png", "tiff"):
        fig.savefig(FIG / f"Fig1_fall_probability_heatmaps.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Single-joint dose (symmetric, C) incl. scale 1 from treat_all
    ps_c = per_seed(c, ["side", "target", "damping_scale", "speed"])
    ref = per_seed(ab[ab.condition == "treat_all"], ["side", "speed"])
    for t in C_TARGETS:
        r = ref.copy(); r["target"] = t; r["damping_scale"] = 1
        ps_c = pd.concat([ps_c, r[ps_c.columns]], ignore_index=True)
    dose = ps_c.groupby(["side", "target", "damping_scale", "speed"]).p.agg(["mean", "std", list]).reset_index()
    dose.drop(columns="list").to_csv(TAB / "TableS1_single_joint_dose_symmetric.csv", index=False)

    ps_d1 = per_seed(d1, ["side", "joint_set", "kd", "speed"])
    dose_d1 = ps_d1.groupby(["side", "joint_set", "kd", "speed"]).p.agg(["mean", "std", list]).reset_index()
    dose_d1.drop(columns="list").to_csv(TAB / "TableS2_dose_directional.csv", index=False)

    # ---------------- Figure 2: dose-response at 0.5 m/s
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True)
    styles = {"hip_pitch": (BLUE, "o"), "knee": (ORANGE, "s"), "ankle_pitch": (AQUA, "^"), "equinovarus": (YELLOW, "D")}
    for k, side in enumerate(SIDES):
        ax = axes[0, k]
        for t in C_TARGETS:
            g = dose[(dose.side == side) & (dose.target == t) & (dose.speed == 0.5)].sort_values("damping_scale")
            col, mk = styles[t]
            m_, e_ = seed_ci(g["list"])
            ax.errorbar(g.damping_scale, m_ * 100, yerr=e_ * 100, color=col, marker=mk, ms=4, lw=1.5,
                        capsize=2, elinewidth=0.8, label=C_LABEL[t])
        ax.set_xscale("log"); ax.set_xlabel("Damping scale (×)")
        ax.set_title(f"{'AB'[k]}  Symmetric damping, single joint — {side} leg", fontsize=8.5, loc="left")
        ax = axes[1, k]
        for js, col, mk, lab in [("ankle", AQUA, "^", "Ankle only"), ("three", BLUE, "o", "Hip + knee + ankle")]:
            g = dose_d1[(dose_d1.side == side) & (dose_d1.joint_set == js) & (dose_d1.speed == 0.5)].sort_values("kd")
            m_, e_ = seed_ci(g["list"])
            ax.errorbar(g.kd, m_ * 100, yerr=e_ * 100, color=col, marker=mk, ms=4, lw=1.5,
                        capsize=2, elinewidth=0.8, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("Resistance gain $b$ (N·m·s/rad)")
        ax.set_title(f"{'CD'[k]}  Direction-selective resistance — {side} leg", fontsize=8.5, loc="left")
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(-3, 103)
    axes[0, 0].set_ylabel("Fall probability at 0.5 m/s (%)"); axes[1, 0].set_ylabel("Fall probability at 0.5 m/s (%)")
    axes[0, 0].legend(frameon=False, fontsize=7); axes[1, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "tiff"):
        fig.savefig(FIG / f"Fig2_dose_response.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Table 4 / Figure 3: kinematic & actuator measures (symmetric, 0.5 m/s)
    rows = []
    for side in SIDES:
        s = side
        sub = ab[(ab.side == s) & (ab.speed == 0.5)]
        for cnd in COND:
            g = sub[sub.condition == cnd]
            seedm = g.groupby("seed").agg(
                romH=(f"rom_{s}_hip_pitch", "mean"), romK=(f"rom_{s}_knee", "mean"),
                romA=(f"rom_{s}_ankle_pitch", "mean"), satA=(f"sat_{s}_ankle_pitch", "mean"),
                satH=(f"sat_{s}_hip_pitch", "mean"), satK=(f"sat_{s}_knee", "mean"),
                tauA=(f"tauabs_{s}_ankle_pitch", "mean"), roll=("trunk_roll_std_deg", "mean"),
                pitch=("trunk_pitch_std_deg", "mean"), vx=("base_vx_mean", "mean"),
                ttf=("time_to_fall_s", "mean"), pf=("fell", "mean"))
            r = {"side": side, "condition": cnd}
            for col in seedm.columns:
                r[col + "_mean"] = seedm[col].mean(); r[col + "_sd"] = seedm[col].std(ddof=1)
            rows.append(r)
    kin = pd.DataFrame(rows)
    kin.to_csv(TAB / "Table4_kinematics_symmetric_0p5.csv", index=False)

    # full kinematic tables (supplementary)
    for d, tag in [(ab, "TableS5_kinematics_symmetric_all"), (d2, "TableS6_kinematics_directional_all")]:
        cols = [k for k in d.columns if k.startswith(("rom_", "tauabs_", "taupk_", "sat_", "taudir_", "foot_",
                                                       "trunk_", "base_vx", "vx_", "actstd_", "time_to_fall"))]
        d.groupby(["side", "condition", "speed"])[cols].mean().reset_index().to_csv(TAB / f"{tag}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    yl = [COND_LABEL[cc].replace(" (= unperturbed)", "") for cc in COND]
    kr = kin[kin.side == "right"].set_index("condition").reindex(COND)
    y = np.arange(len(COND))
    for j, (key, col, mk, lab) in enumerate([("romH", BLUE, "o", "Hip"), ("romK", ORANGE, "s", "Knee"),
                                             ("romA", AQUA, "^", "Ankle")]):
        axes[0].errorbar(kr[key + "_mean"], y + (j - 1) * 0.22, xerr=kr[key + "_sd"], fmt=mk, color=col, ms=4,
                         elinewidth=0.8, capsize=1.5, label=lab)
    axes[0].set_xlabel("Range of motion, right leg (deg)"); axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_title("A  Joint excursion", fontsize=8.5, loc="left")
    axes[1].barh(y, kr["satA_mean"] * 100, xerr=kr["satA_sd"] * 100, color=AQUA, height=0.6,
                 error_kw={"elinewidth": 0.8, "capsize": 1.5})
    axes[1].set_xlabel("Time at torque limit (%)")
    axes[1].set_title("B  Ankle actuator saturation", fontsize=8.5, loc="left")
    axes[2].barh(y, kr["roll_mean"], xerr=kr["roll_sd"], color=BLUE, height=0.6,
                 error_kw={"elinewidth": 0.8, "capsize": 1.5})
    axes[2].set_xlabel("Trunk roll SD (deg)"); axes[2].set_title("C  Mediolateral trunk sway", fontsize=8.5, loc="left")
    axes[0].set_yticks(y); axes[0].set_yticklabels(yl); axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    for ext in ("png", "tiff"):
        fig.savefig(FIG / f"Fig3_mechanism_symmetric_right_0p5.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- reproducibility: D1 'three' (untreated) vs D2 no_treatment, identical configuration
    rep = []
    for side, kd in [("right", 40), ("left", 50)]:
        a = d1[(d1.side == side) & (d1.joint_set == "three") & (d1.kd == kd)].sort_values(["seed", "speed", "rollout"])
        b = d2[(d2.side == side) & (d2.condition == "no_treatment")].sort_values(["seed", "speed", "rollout"])
        pa = a.groupby(["seed", "speed"]).fell.mean().values; pb = b.groupby(["seed", "speed"]).fell.mean().values
        rep.append({"side": side, "kd": kd, "rollout_agreement": float((a.fell.values == b.fell.values).mean()),
                    "falls_run1": int(a.fell.sum()), "falls_run2": int(b.fell.sum()), "n_rollouts": len(a),
                    "overall_p_run1": a.fell.mean(), "overall_p_run2": b.fell.mean(),
                    "seed_speed_cell_corr": float(np.corrcoef(pa, pb)[0, 1]),
                    "mean_abs_diff_seed_speed_cell": float(np.abs(pa - pb).mean()),
                    "wilcoxon_p_seed_speed_cells": float(stats.wilcoxon(pa, pb, zero_method="zsplit").pvalue)})
    rep = pd.DataFrame(rep); rep.to_csv(TAB / "TableS7_reproducibility.csv", index=False)

    # ---------------- key numbers for the text
    def fp(ft, side, cnd, sp):
        return float(rnd(100 * float(ft[(ft.side == side) & (ft.condition == cnd) & (ft.speed == sp)]["mean"].iloc[0])))
    for name, ft in [("sym", ft_ab), ("dir", ft_d2)]:
        KEY[name] = {f"{side}_{cnd}_{sp}": fp(ft, side, cnd, sp) for side in SIDES for cnd in COND for sp in SPEEDS}
    KEY["tests_sym"] = t_ab.round(4).to_dict(orient="records")
    KEY["tests_dir"] = t_d2.round(4).to_dict(orient="records")
    KEY["dose_sym_0p5"] = {f"{r.side}_{r.target}_{r.damping_scale}": float(rnd(100 * r["mean"]))
                           for _, r in dose[dose.speed == 0.5].iterrows()}
    KEY["dose_dir"] = {f"{r.side}_{r.joint_set}_{r.kd}_{r.speed}": float(rnd(100 * r["mean"])) for _, r in dose_d1.iterrows()}
    KEY["kin_sym_0p5"] = kin.round(3).to_dict(orient="records")
    KEY["reproducibility"] = rep.round(4).to_dict(orient="records")
    tau_dir = d1[d1.joint_set == "ankle"].groupby(["side", "kd"]).apply(
        lambda g: g[f"taudir_{g.name[0]}_ankle_pitch"].mean(), include_groups=False)
    KEY["dir_ankle_mean_torque"] = {f"{s}_{k}": round(float(v), 2) for (s, k), v in tau_dir.items()}
    KEY["n_rows"] = {"AB": len(ab), "D1": len(d1), "D2": len(d2), "C": len(c)}
    (ROOT / "results/key_numbers.json").write_text(json.dumps(KEY, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
