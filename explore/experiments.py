"""
experiments.py -- probe CiGRAM's stated properties and prototype the multiplex
extension.

Run:  MPLCONFIGDIR=./.mplconfig .venv/bin/python -m src.experiments
Outputs: results/*.json and figs/*.png
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from explore.cigram_core import cigram, cigram_modular, cigram_multiplex

FIG = "figs"
RES = "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.4,
})

C = dict(blue="#2C6E9B", orange="#D9822B", green="#4E8B4A",
         red="#B4443C", purple="#6B5B95", grey="#8A8A8A")


def nmi(a, b):
    """Normalised mutual information between two label vectors (arithmetic mean
    normalisation), computed directly to avoid an sklearn dependency."""
    a = np.asarray(a); b = np.asarray(b)
    n = a.size
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ua.size, ub.size))
    np.add.at(cont, (ia, ib), 1)
    p = cont / n
    pa = p.sum(1); pb = p.sum(0)
    nz = p > 0
    mi = float(np.sum(p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])))
    ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    denom = 0.5 * (ha + hb)
    return mi / denom if denom > 0 else 0.0


def _props(adj):
    G = nx.from_numpy_array(adj)
    d = np.asarray([k for _, k in G.degree()], dtype=float)
    return dict(
        degrees=d,
        assort=float(nx.degree_assortativity_coefficient(G)),
        clustering=float(nx.average_clustering(G)),
        mean_k=float(d.mean()),
        max_k=float(d.max()),
        modularity=float(nx.community.modularity(G, nx.community.louvain_communities(G, seed=0)))
        if G.number_of_edges() else np.nan,
    )


# ---------------------------------------------------------------- power-law fits

def pl_fit(degrees, xmin=None):
    """Clauset-Shalizi-Newman continuous power-law MLE on the tail.

    Returns (alpha, xmin, ks) choosing xmin among candidate quantiles by
    minimising the KS distance, which is the standard heuristic.
    """
    x = np.asarray([v for v in degrees if v > 0], dtype=float)
    x = np.sort(x)
    if x.size < 50:
        return np.nan, np.nan, np.nan
    best = (np.inf, None, None)
    for q in [0.80, 0.85, 0.90, 0.95, 0.97]:
        xm = np.quantile(x, q)
        tail = x[x >= xm]
        if tail.size < 30 or xm <= 0:
            continue
        alpha = 1.0 + tail.size / np.sum(np.log(tail / xm))
        # empirical vs fitted CCDF on the tail
        ks = _ks_pl(tail, alpha, xm)
        if ks < best[0]:
            best = (ks, alpha, xm)
    return best[1], best[2], best[0]


def _ks_pl(tail, alpha, xm):
    tail = np.sort(tail)
    n = tail.size
    emp = 1.0 - np.arange(1, n + 1) / n          # CCDF at each point
    fit = (tail / xm) ** (-(alpha - 1.0))
    return float(np.max(np.abs(emp - fit)))


def lognorm_fit(degrees):
    x = np.asarray([v for v in degrees if v > 0], dtype=float)
    mu, sd = np.log(x).mean(), np.log(x).std(ddof=1)
    xs = np.sort(x)
    n = xs.size
    emp = 1.0 - np.arange(1, n + 1) / n
    from scipy.stats import norm
    fit = 1.0 - norm.cdf((np.log(xs) - mu) / sd)
    return mu, sd, float(np.max(np.abs(emp - fit)))


# ---------------------------------------------------------------- experiment 1

def _assort_job(args):
    n, avg, a, sf, ss, seed = args
    g = cigram(n, avg_deg=avg, sigma_f=sf, sigma_s=ss, a=a, seed=seed)
    p = _props(g.adj)
    return dict(avg=avg, a=a, assort=p["assort"], clustering=p["clustering"])


def exp_assortativity(reps=5):
    jobs = [(1000, avg, a, 0.8, 0.8, s)
            for avg in [4, 6, 10, 15, 20, 30, 50]
            for a in [0.0, 2.0, 4.0, -3.0]
            for s in range(reps)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(_assort_job, jobs))
    res = {}
    for o in out:
        res.setdefault(f"{o['avg']}|{o['a']}", []).append(o)
    agg = {}
    for key, vals in res.items():
        avg, a = key.split("|")
        agg[key] = dict(avg=int(avg), a=float(a),
                        assort=float(np.mean([v["assort"] for v in vals])),
                        assort_sd=float(np.std([v["assort"] for v in vals])),
                        clustering=float(np.mean([v["clustering"] for v in vals])))
    return agg


# ---------------------------------------------------------------- experiment 2

def _tail_job(args):
    n, avg, sf, ss, seed = args
    g = cigram(n, avg_deg=avg, sigma_f=sf, sigma_s=ss, a=0.0, seed=seed)
    d = g.degrees.astype(float)
    alpha, xmin, ks_pl = pl_fit(d)
    mu, sd, ks_ln = lognorm_fit(d)
    return dict(sigma_f=sf, sigma_s=ss, alpha=alpha, xmin=xmin, ks_pl=ks_pl,
                ks_lognorm=ks_ln, mean_k=float(d.mean()), max_k=float(d.max()),
                cv=float(d.std() / d.mean()))


def exp_tail(reps=4):
    jobs = [(2000, 10, sf, ss, s)
            for sf in [0.3, 0.5, 0.8, 1.2]
            for ss in [0.4, 0.6, 0.9, 1.5]
            for s in range(reps)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(_tail_job, jobs))
    agg = {}
    for o in out:
        k = f"{o['sigma_f']}|{o['sigma_s']}"
        agg.setdefault(k, []).append(o)
    return {k: {kk: float(np.nanmean([v[kk] for v in vals])) for kk in vals[0]}
            for k, vals in agg.items()}


# ---------------------------------------------------------------- experiment 3

def _k_job(args):
    n, K, avg, seed = args
    g = cigram_modular(n, K, avg_deg=avg, sigma_f=0.8, sigma_s=0.9, a=0.0,
                       ek=0.05, p_o=0.0, seed=seed)
    d = g.degrees.astype(float)
    alpha, xmin, ks_pl = pl_fit(d)
    return dict(K=K, max_k=float(d.max()), cv=float(d.std() / d.mean()),
                mean_k=float(d.mean()), alpha=alpha, ks_pl=ks_pl,
                modularity=float(nx.community.modularity(
                    nx.from_numpy_array(g.adj),
                    [set(np.nonzero(g.communities == c)[0]) for c in np.unique(g.communities)])))


def exp_K(reps=4):
    jobs = [(2000, K, 10, s) for K in [1, 2, 5, 10, 20, 50, 100] for s in range(reps)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(_k_job, jobs))
    agg = {}
    for o in out:
        agg.setdefault(o["K"], []).append(o)
    return {str(k): {kk: float(np.nanmean([v[kk] for v in vals])) for kk in vals[0]}
            for k, vals in agg.items()}


# ---------------------------------------------------------------- truth check

def exp_truth_vs_observed():
    """How much modularity does Louvain find in a CiGRAM graph that has NO
    planted communities, relative to a degree-preserving null?

    This is the thesis' central assumption made quantitative: a non-modular
    random graph is not distinguishable from a module, so any modularity
    threshold has to be calibrated against the degree sequence.
    """
    out = []
    for K in [1, 3, 5, 10, 20, 50]:
        for seed in range(3):
            g = cigram_modular(1000, K, avg_deg=8, sigma_f=0.8, sigma_s=0.9,
                               ek=0.05, seed=seed)
            G = nx.from_numpy_array(g.adj)
            comms = nx.community.louvain_communities(G, seed=0)
            Q = nx.community.modularity(G, comms)
            qs = []
            for j in range(8):
                H = nx.configuration_model([d for _, d in G.degree()], seed=seed + j)
                H = nx.Graph(H)
                H.remove_edges_from(nx.selfloop_edges(H))
                cs = nx.community.louvain_communities(H, seed=0)
                qs.append(nx.community.modularity(H, cs))
            # how well does Louvain recover the planted partition?
            pred = np.zeros(1000, dtype=int)
            for i, c in enumerate(comms):
                pred[list(c)] = i
            out.append(dict(K=K, seed=seed, Q=float(Q), Q_null=float(np.mean(qs)),
                            Q_null_sd=float(np.std(qs)),
                            nmi_planted=float(nmi(g.communities, pred))))
    return out


# ---------------------------------------------------------------- multiplex demo

def _louvain_labels(adj, seed=0):
    G = nx.from_numpy_array(adj)
    cs = nx.community.louvain_communities(G, seed=seed, resolution=1.0)
    lab = np.zeros(adj.shape[0], dtype=int)
    for i, c in enumerate(cs):
        lab[list(c)] = i
    return lab


def _consensus(adjs, seed=0):
    """Lancichinetti-Fortunato style consensus across layers: Louvain per
    layer, co-association matrix, Louvain on that. A cheap multiplex baseline
    that avoids choosing gamma/omega."""
    labs = [_louvain_labels(a, seed) for a in adjs]
    n = adjs[0].shape[0]
    C = np.zeros((n, n))
    for lab in labs:
        C += (lab[:, None] == lab[None, :])
    C /= len(labs)
    np.fill_diagonal(C, 0.0)
    G = nx.from_numpy_array(C)
    cs = nx.community.louvain_communities(G, seed=seed)
    lab = np.zeros(n, dtype=int)
    for i, c in enumerate(cs):
        lab[list(c)] = i
    return lab


def exp_multiplex(reps=3):
    """CiGRAM-M as an ascertainment experiment."""
    n, K = 600, 6
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    out = {"gamma": gammas, "nmi": {}, "degree_corr": []}

    for gamma in gammas:
        acc = {"dx": [], "rx": [], "meas": [], "union": [], "consensus": []}
        dcorr = []
        for rep in range(reps):
            layers = [
                {"name": "dx", "sigma_s": 1.0, "a": 0.0, "ek": 0.08},
                {"name": "rx", "sigma_s": 0.8, "a": 1.5, "ek": 0.08},
                {"name": "meas", "sigma_s": 1.2, "a": 0.0, "ek": 0.08,
                 "utilisation_bias": gamma},
            ]
            graphs, ctx = cigram_multiplex(n, layers, rho=0.8, avg_deg=8,
                                           K=K, seed=100 + rep)
            truth = graphs[0].communities
            adjs = [g.adj for g in graphs]

            acc["dx"].append(nmi(truth, _louvain_labels(adjs[0])))
            acc["rx"].append(nmi(truth, _louvain_labels(adjs[1])))
            acc["meas"].append(nmi(truth, _louvain_labels(adjs[2])))

            union = np.zeros_like(adjs[0])
            for a in adjs:
                union |= a
            acc["union"].append(nmi(truth, _louvain_labels(union)))
            acc["consensus"].append(nmi(truth, _consensus(adjs)))

            from scipy.stats import spearmanr
            dcorr.append(float(spearmanr(graphs[0].degrees,
                                         graphs[2].degrees).statistic))
        for k in acc:
            out["nmi"].setdefault(k, []).append(float(np.mean(acc[k])))
        out["degree_corr"].append(float(np.mean(dcorr)))

    # rho sweep: does the coupling parameter do what it says on the tin?
    from scipy.stats import spearmanr
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    out["rho"] = rhos
    out["rho_corr"] = []
    for rho in rhos:
        vals = []
        for rep in range(4):
            lay = [{"name": "dx", "sigma_s": 1.0, "ek": 0.08},
                   {"name": "rx", "sigma_s": 0.9, "ek": 0.08},
                   {"name": "meas", "sigma_s": 1.1, "ek": 0.08}]
            gs, _ = cigram_multiplex(500, lay, rho=rho, avg_deg=8, K=5, seed=rep)
            d = [x.degrees for x in gs]
            vals.append([spearmanr(d[0], d[1]).statistic,
                         spearmanr(d[0], d[2]).statistic,
                         spearmanr(d[1], d[2]).statistic])
        out["rho_corr"].append(float(np.mean(vals)))

    layers = [{"name": "dx", "sigma_s": 1.0, "ek": 0.08},
              {"name": "rx", "sigma_s": 0.8, "a": 1.5, "ek": 0.08},
              {"name": "meas", "sigma_s": 1.2, "ek": 0.08,
               "utilisation_bias": 2.5}]
    graphs, _ = cigram_multiplex(n, layers, rho=0.8, avg_deg=8, K=K, seed=5)
    out["snapshot_degrees"] = [g.degrees.tolist() for g in graphs]
    out["snapshot_truth"] = graphs[0].communities.tolist()
    return out


# ---------------------------------------------------------------- figures

def fig_assort(agg):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    styles = {4.0: (C["orange"], "a = +4"), 2.0: (C["green"], "a = +2"),
              0.0: (C["grey"], "a = 0"), -3.0: (C["blue"], "a = -3")}
    for a, (col, lab) in styles.items():
        pts = sorted([v for v in agg.values() if v["a"] == a], key=lambda v: v["avg"])
        xs = [p["avg"] for p in pts]
        ys = [p["assort"] for p in pts]
        es = [p["assort_sd"] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, color=col, label=lab,
                    capsize=2, elinewidth=0.8)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 10, 15, 20, 30, 50])
    ax.set_xticklabels(["4", "6", "10", "15", "20", "30", "50"])
    ax.set_xlabel("mean degree")
    ax.set_ylabel("degree assortativity $r$")
    ax.set_title("Positive assortativity dies with density\n(CiGRAM, $n=1000$)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_assort_density.png", bbox_inches="tight")
    plt.close(fig)


def fig_tail(agg):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # CCDF panel for a few parameter settings
    ax = axes[0]
    picks = [("0.3", "0.4"), ("0.5", "0.6"), ("0.8", "0.9"), ("1.2", "1.5")]
    cols = [C["red"], C["orange"], C["green"], C["blue"]]
    for (sf, ss), col in zip(picks, cols):
        g = cigram(2000, avg_deg=10, sigma_f=float(sf), sigma_s=float(ss), seed=3)
        d = np.sort(g.degrees)[::-1]
        d = d[d > 0]
        ax.plot(d, np.arange(1, d.size + 1) / d.size, color=col, lw=1.2,
                label=fr"$\sigma_f$={sf}, $\sigma_s$={ss}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degree $k$"); ax.set_ylabel(r"$P(K \geq k)$")
    ax.set_title("Heavy tails with a hard cutoff", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    # alpha vs KS comparison
    ax = axes[1]
    als = [v["alpha"] for v in agg.values() if np.isfinite(v["alpha"])]
    ax.hist(als, bins=np.arange(1.8, 6.5, 0.25), color=C["blue"], alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(2.0, color=C["red"], ls="--", lw=1.0, label="clean scale-free ($\\alpha=2$)")
    ax.set_xlabel(r"fitted tail exponent $\hat{\alpha}$")
    ax.set_ylabel("parameter settings")
    ax.set_title("Fitted $\\hat{\\alpha}$ across 16 settings", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_tails.png", bbox_inches="tight")
    plt.close(fig)


def fig_null(truth, agg):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ks = sorted(set(r["K"] for r in truth))

    ax = axes[0]
    q = [np.mean([r["Q"] for r in truth if r["K"] == k]) for k in ks]
    qn = [np.mean([r["Q_null"] for r in truth if r["K"] == k]) for k in ks]
    ax.plot(ks, q, "o-", color=C["red"], ms=4, label="CiGRAM graph")
    ax.plot(ks, qn, "s--", color=C["grey"], ms=4, label="degree-preserving null")
    ax.set_xscale("log")
    ax.set_xlabel("planted communities $K$")
    ax.set_ylabel("Louvain modularity $Q$")
    ax.set_title("$Q\\approx0.3$ needs no communities at all", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1]
    nm = [np.mean([r["nmi_planted"] for r in truth if r["K"] == k]) for k in ks]
    ax.plot(ks, nm, "o-", color=C["blue"], ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("planted communities $K$")
    ax.set_ylabel("NMI vs. planted partition")
    ax.set_title("Recovery improves as blocks shrink and densify", fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_null.png", bbox_inches="tight")
    plt.close(fig)


def fig_multiplex(mp):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    ax = axes[0]
    ax.plot(mp["rho"], mp["rho_corr"], "o-", color=C["blue"], ms=4)
    ax.set_xlabel(r"coupling $\rho$")
    ax.set_ylabel("inter-layer degree correlation")
    ax.set_title(r"$\rho$ is an interpretable knob", fontsize=9)
    ax.set_ylim(-0.05, 1.0)

    ax = axes[1]
    cols = {"dx": C["blue"], "rx": C["green"], "meas": C["orange"],
            "union": C["purple"], "consensus": C["red"]}
    labs = {"dx": "layer 1 (dx)", "rx": "layer 2 (rx)",
            "meas": "layer 3 (biased)", "union": "union",
            "consensus": "consensus"}
    mk = {"dx": "o", "rx": "s", "meas": "^", "union": "v", "consensus": "D"}
    for k in ["dx", "rx", "meas", "union", "consensus"]:
        ax.plot(mp["gamma"], mp["nmi"][k], marker=mk[k], ms=3.5,
                color=cols[k], label=labs[k], lw=1.3)
    ax.set_xlabel("ascertainment bias $\\gamma$ in layer 3")
    ax.set_ylabel("NMI vs. planted structure")
    ax.set_ylim(0, 1.02)
    ax.set_title("Aggregation is hard to beat", fontsize=9)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left")

    ax = axes[2]
    for i, (col, nm) in enumerate(zip([C["blue"], C["green"], C["orange"]],
                                      ["dx", "rx", "meas"])):
        d = np.sort(np.array(mp["snapshot_degrees"][i]))[::-1]
        d = d[d > 0]
        ax.plot(d, np.arange(1, d.size + 1) / d.size, color=col, lw=1.2, label=nm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degree"); ax.set_ylabel(r"$P(K \geq k)$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Same population, different tails", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_multiplex.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("1/4 assortativity vs density ...")
    a1 = exp_assortativity()
    json.dump(a1, open(f"{RES}/assortativity.json", "w"), indent=1)
    fig_assort(a1)

    print("2/4 tail shapes ...")
    a2 = exp_tail()
    json.dump(a2, open(f"{RES}/tails.json", "w"), indent=1)
    fig_tail(a2)

    print("3/4 K dependence ...")
    a3 = exp_K()
    json.dump(a3, open(f"{RES}/K.json", "w"), indent=1)

    print("4/4 multiplex ...")
    a4 = exp_multiplex()
    json.dump(a4, open(f"{RES}/multiplex.json", "w"), indent=1)
    fig_multiplex(a4)

    print("5/5 null calibration ...")
    a5 = exp_truth_vs_observed()
    json.dump(a5, open(f"{RES}/truth.json", "w"), indent=1)
    fig_null(a5, a3)
    print("done")


if __name__ == "__main__":
    main()
