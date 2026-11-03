"""
experiments_bias.py -- three demonstrations of the ascertainment argument.

  1. The ingrown toenail effect. A condition that is only ever *found* has an
     observed incidence proportional to how often the patient was looked at.
     The association between two such conditions is entirely explained by
     shared contact, and vanishes under stratification.

  2. Differential ascertainment breaks negative controls. Under a shared bias,
     calibrating with a negative control outcome recovers the truth. Under a
     *channel-specific* bias -- an exposure that changes the kind of
     observation, not just its volume -- a negative control drawn from the
     wrong channel under-corrects. The same data, analysed two ways, gives
     answers that differ by a factor of ~2.

  3. Marginal equivalence. Two generators with *identical* marginal prevalences
     differ completely in their joint structure. Any fidelity metric based on
     marginals rates them the same. This is the formal reason an LLM-generated
     CDM can look faithful and still be useless for methods research.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python -m src.experiments_bias
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore.cdm_sim import Scenario, simulate_population, incidental, true_effect

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.4,
})
C = dict(blue="#2C6E9B", orange="#D9822B", green="#4E8B4A",
         red="#B4443C", purple="#6B5B95", grey="#8A8A8A")


def rate(counts, group, post):
    """Events per person-year within a group, given a common follow-up length."""
    return counts[group].sum() / (group.sum() * post)


# --------------------------------------------------------------- experiment 1

def exp_toenail(n=40000):
    """Incidence of a purely 'found' condition tracks contact, not disease."""
    scn = Scenario(n=n, exposure_frac=0.0, seed=3)
    rng = np.random.default_rng(3)
    pop = simulate_population(scn, 3)

    exams = incidental(pop, scn.eps_exam, "exam", rng)[0]
    labs = incidental(pop, scn.eps_lab, "lab", rng)[0]

    # incidence per 1000 person-years by decile of visit count (pre-period)
    q = np.quantile(pop.visits_pre, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(pop.visits_pre, q[1:-1]), 0, 9)
    py = scn.t_exposure

    xs, y_exam, y_lab = [], [], []
    for b in range(10):
        m = bins == b
        if m.sum() < 50:
            continue
        xs.append(pop.visits_pre[m].mean())
        y_exam.append(1000 * exams[m].sum() / (m.sum() * py))
        y_lab.append(1000 * labs[m].sum() / (m.sum() * py))

    # co-occurrence of two found conditions, raw and stratified by visits
    a = exams > 0
    b_ = labs > 0
    raw_or = (a & b_).sum() * (~a & ~b_).sum() / max((a & ~b_).sum() * (~a & b_).sum(), 1)

    # Mantel-Haenszel over visit-count strata
    num, den = 0.0, 0.0
    for s in np.unique(bins):
        m = bins == s
        a1 = (a & b_ & m).sum(); b1 = (a & ~b_ & m).sum()
        c1 = (~a & b_ & m).sum(); d1 = (~a & ~b_ & m).sum()
        n1 = m.sum()
        if n1 == 0:
            continue
        num += a1 * d1 / n1
        den += b1 * c1 / n1
    mh = num / den if den > 0 else np.nan

    return dict(visit_mean=xs, inc_exam=y_exam, inc_lab=y_lab,
                or_raw=float(raw_or), or_stratified=float(mh),
                exams_per_py=float(pop.visits_pre.mean() / py),
                labs_per_py=float(pop.labs_pre.mean() / py))


# --------------------------------------------------------------- experiment 2

def _irr(counts, exposed, post):
    return rate(counts, exposed, post) / rate(counts, ~exposed, post)


def _poisson_fit(y, X, offset, iters=200):
    """Tiny IRLS Poisson regression, log link, fixed offset."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + offset
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        A = X.T @ (X * mu[:, None]) + 1e-8 * np.eye(X.shape[1])
        b_new = np.linalg.lstsq(A, X.T @ (mu * z), rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-10:
            return b_new
        b = b_new
    return b


def _profile(b0b1tau_fn, starts):
    best = None
    for s0 in starts:
        r = optimize.minimize(b0b1tau_fn, x0=s0, method="Nelder-Mead",
                              options=dict(maxiter=40000, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best


def _fit_plugin(w, labs, exposed, post):
    """Structural, but plugging in the patient's OBSERVED channel count.

    The observed count is an unbiased estimate of the rate but a biased
    estimate of the detection probability, because 1-exp(-tau*L/T) is convex
    in L. The bias is largest for patients with few tests, which compresses
    the exposed/unexposed contrast. This is the estimator most people would
    write, and it is not good enough.
    """
    D = exposed.astype(float)
    lab_rate = (labs + 0.5) / post

    def nll(p):
        b0, b1, logtau = p
        pdet = 1.0 - np.exp(-lab_rate * np.exp(logtau))
        mu = np.clip(post * np.exp(b0 + b1 * D) * pdet, 1e-12, None)
        return float(np.sum(mu - w * np.log(mu)))

    r = _profile(nll, [[np.log(0.05), 0.5, np.log(t)] for t in (0.1, 0.3, 1.0)])
    return float(np.exp(r.x[1]))


def _fit_channel_model(w, labs, visits, exposed, post):
    """Model the channel instead of measuring it.

    Stage 1 fits the lab count from visit count and exposure; stage 2 uses the
    fitted channel rate, which removes the convexity artefact. Reachable only
    because the model represents the channel explicitly.
    """
    D = exposed.astype(float)
    X = np.column_stack([np.ones(len(D)), visits, D])
    b = _poisson_fit(labs.astype(float), X, np.log(post))
    mu_lab = np.exp(X @ b + np.log(post))
    lab_rate = mu_lab / post

    def nll(p):
        b0, b1, logtau = p
        pdet = 1.0 - np.exp(-lab_rate * np.exp(logtau))
        mu = np.clip(post * np.exp(b0 + b1 * D) * pdet, 1e-12, None)
        return float(np.sum(mu - w * np.log(mu)))

    r = _profile(nll, [[np.log(0.05), 0.5, np.log(t)] for t in (0.1, 0.3, 1.0)])
    return float(np.exp(r.x[1]))


def exp_negative_controls(reps=12):
    """Shared bias vs channel-specific bias, and what each estimator does."""
    out = {k: [] for k in ["naive", "nco_exam", "nco_lab", "plugin",
                           "channel_model", "oracle"]}
    true_rr, true_irr = [], []

    for rep in range(reps):
        scn = Scenario(seed=100 + rep)
        rng = np.random.default_rng(100 + rep)
        pop = simulate_population(scn, 100 + rep)
        post = scn.years - scn.t_exposure
        e = pop.exposed

        y = incidental(pop, scn.eps_exam, "exam", rng)
        z = incidental(pop, scn.eps_lab, "lab", rng)
        w = true_effect(pop, rng)

        w_post = w["observed_post"]
        irr_w = _irr(w_post, e, post)
        irr_y = _irr(y[1], e, post)
        irr_z = _irr(z[1], e, post)

        out["naive"].append(float(irr_w))
        out["nco_exam"].append(float(irr_w / irr_y))
        out["nco_lab"].append(float(irr_w / irr_z))

        out["plugin"].append(_fit_plugin(w_post, pop.labs_post, e, post))
        out["channel_model"].append(
            _fit_channel_model(w_post, pop.labs_post, pop.visits_post, e, post))

        # oracle: backs out the true p_detect
        oracle = _irr(w_post / w["p_det_post"], e, post)
        out["oracle"].append(float(oracle))

        true_rr.append(_irr(w["true_post"], e, post))
        true_irr.append(scn.rr_true)

    return {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), values=v)
            for k, v in out.items()} | {
        "truth": float(np.mean(true_irr)),
        "truth_from_episodes": float(np.mean(true_rr)),
    }


# --------------------------------------------------------------- experiment 3

def exp_marginal_equivalence(n=8000, n_exam=6, n_lab=6, reps=6):
    """Two generators, identical marginals, different joint structure."""
    scn = Scenario(n=n, exposure_frac=0.0, seed=21)
    rng = np.random.default_rng(21)
    pop = simulate_population(scn, 21)

    eps_exam = np.linspace(0.02, 0.07, n_exam)
    eps_lab = np.linspace(0.05, 0.14, n_lab)

    obs = {}
    for i, e_ in enumerate(eps_exam):
        obs[f"exam_{i}"] = incidental(pop, e_, "exam", rng)[0]
    for i, e_ in enumerate(eps_lab):
        obs[f"lab_{i}"] = incidental(pop, e_, "lab", rng)[0]

    names = list(obs)
    B = np.column_stack([obs[k] for k in names])
    prev = B.mean(axis=0)

    results = {"prevalence": prev.tolist(), "names": names}
    for label in ["marginal", "topology"]:
        if label == "topology":
            M = B
        else:
            # the marginal/LLM-like proxy: independent Bernoulli at the same
            # prevalences, no visit process, one encounter per patient
            M = np.column_stack([rng.binomial(1, p, size=n) for p in prev])

        deg = M.sum(axis=1)
        # pairwise association screen (chi-square, Bonferroni)
        sig = 0
        total = 0
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                total += 1
                tab = np.array([[((M[:, i] > 0) & (M[:, j] > 0)).sum(),
                                 ((M[:, i] > 0) & (M[:, j] == 0)).sum()],
                                [((M[:, i] == 0) & (M[:, j] > 0)).sum(),
                                 ((M[:, i] == 0) & (M[:, j] == 0)).sum()]])
                if tab.min() == 0:
                    tab = tab + 0.5
                try:
                    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
                except ValueError:
                    p = 1.0
                if p < 0.05 / total:
                    sig += 1
                pairs.append((names[i], names[j], float(p)))
        hist = np.bincount(deg, minlength=len(names) + 1)
        results[label] = dict(
            degree_hist=[int(x) for x in hist],
            degree_mean=float(deg.mean()),
            degree_p99=float(np.percentile(deg, 99)),
            degree_max=int(deg.max()),
            significant_pairs=int(sig),
            total_pairs=int(total),
            pairs=pairs,
        )
    return results


# --------------------------------------------------------------- figures

def fig_toenail(t):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    ax.plot(t["visit_mean"], t["inc_exam"], "o-", color=C["orange"], ms=4,
            label="found on examination")
    ax.plot(t["visit_mean"], t["inc_lab"], "s-", color=C["blue"], ms=4,
            label="found on lab test")
    ax.set_xlabel("visits per patient in the period")
    ax.set_ylabel("observed incidence per 1000 PY")
    ax.set_title("Incidence tracks contact, not disease", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.bar(["raw", "stratified\nby visit count"], [t["or_raw"], t["or_stratified"]],
           color=[C["red"], C["green"]], width=0.55)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("odds ratio between two found conditions")
    ax.set_title("The association is entirely shared contact", fontsize=9)
    ax.set_ylim(0, max(t["or_raw"], 1.5) * 1.15)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_toenail.png", bbox_inches="tight")
    plt.close(fig)


def fig_negative_controls(r):
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    order = [("naive", "naive cohort", C["grey"]),
             ("nco_exam", "negative control,\nwrong channel", C["orange"]),
             ("plugin", "plug-in channel\nadjustment", C["purple"]),
             ("channel_model", "channel modelled\n(generative)", C["blue"]),
             ("nco_lab", "negative control,\nright channel", C["green"])]
    ys = np.arange(len(order))[::-1]
    for y_, (k, lab, col) in zip(ys, order):
        m, sd = r[k]["mean"], r[k]["sd"]
        ax.errorbar(m, y_, xerr=1.96 * sd / np.sqrt(len(r[k]["values"])),
                    fmt="o", ms=6, color=col, capsize=3, elinewidth=1.2)
        ax.text(m, y_ + 0.22, f"{m:.2f}", ha="center", fontsize=7.5, color=col)
    ax.axvline(r["truth"], color="k", ls="--", lw=1.0)
    ax.text(r["truth"], -0.75, f"truth {r['truth']:.1f}", fontsize=7.5,
            va="center", ha="center")
    ax.set_ylim(-1.0, len(order) - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([o[1] for o in order], fontsize=7.5)
    ax.set_xlabel("estimated risk ratio for the true effect")
    ax.set_title("Same data, five answers", fontsize=9)
    ax.set_xlim(0, max(r["naive"]["mean"], r["truth"]) * 1.35)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_negative_controls.png", bbox_inches="tight")
    plt.close(fig)


def fig_marginal(m):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))

    ax = axes[0]
    p = np.array(m["prevalence"])
    ax.plot([0, p.max() * 1.05], [0, p.max() * 1.05], "k:", lw=0.8)
    ax.scatter(p, p, color=C["blue"], s=16)
    ax.set_xlabel("prevalence, generator A")
    ax.set_ylabel("prevalence, generator B")
    ax.set_title("Marginals match exactly", fontsize=9)

    ax = axes[1]
    for key, col, lab in [("marginal", C["orange"], "A: independent"),
                          ("topology", C["blue"], "B: visit-driven")]:
        d = m[key]["degree_hist"]
        d = np.array(d) / sum(d)
        ax.plot(np.arange(len(d)), d, "o-", color=col, ms=3, label=lab)
    ax.set_xlabel("distinct conditions per patient")
    ax.set_ylabel("fraction of patients")
    ax.set_title("Joint structure does not", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    vals = [m["marginal"]["significant_pairs"], m["topology"]["significant_pairs"]]
    ax.bar(["A: independent", "B: visit-driven"], vals,
           color=[C["orange"], C["blue"]], width=0.55)
    ax.axhline(m["topology"]["total_pairs"], color="k", ls=":", lw=0.8)
    ax.set_ylabel("significant pairs (Bonferroni)")
    ax.set_title(f"of {m['topology']['total_pairs']} possible pairs", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_marginal.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("1/3 toenail ...")
    t1 = exp_toenail()
    json.dump(t1, open(f"{RES}/toenail.json", "w"), indent=1)
    fig_toenail(t1)

    print("2/3 negative controls ...")
    t2 = exp_negative_controls()
    json.dump(t2, open(f"{RES}/negative_controls.json", "w"), indent=1)
    fig_negative_controls(t2)

    print("3/3 marginal equivalence ...")
    t3 = exp_marginal_equivalence()
    json.dump(t3, open(f"{RES}/marginal_equivalence.json", "w"), indent=1)
    fig_marginal(t3)
    print("done")


if __name__ == "__main__":
    main()
