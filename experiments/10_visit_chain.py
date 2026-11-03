"""
10 -- does the visit chain recover the timing structure?

Isolated test of the new layer: take each patient's real record counts and real
visit dates, and re-assign the records to visits using the fitted chain. If the
layer is right, the same-day burstiness and the transition structure come back.
The counts are untouched, so this isolates placement from incidence.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import topocdm as t
from topocdm.io import load_incidence
from topocdm.visits import fit_visit_model, assign_to_visits

FIG, RES = "figs", "results"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

d, groups, cv = load_incidence()
H = int(d.concept_group.max()) + 1
cg_of = dict(zip(d.concept_ids, d.concept_group))
ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"])
vi = pd.read_csv("data/visits.csv", parse_dates=["visit_date"])

model = fit_visit_model(vi, ev, cg_of, H, concept_group=d.concept_group)
print("visit model:", model.summary())

rng = np.random.default_rng(0)
sample = rng.choice(d.n, size=2000, replace=False)
vdates = vi.groupby("person_id")["visit_date"].apply(lambda s: np.sort(s.to_numpy()))

rows = []
for i in sample:
    pid = int(d.person_ids[i])
    if pid not in vdates.index:
        continue
    dates = vdates.loc[pid]
    counts = np.asarray(d.X[i].todense()).ravel()
    pairs = assign_to_visits(counts, len(dates), model, rng)
    for v, j in pairs:
        rows.append((pid, int(d.concept_ids[j]), dates[v]))
syn = pd.DataFrame(rows, columns=["person_id", "concept_id", "event_date"])
print(f"reassigned {len(syn):,} records for {len(sample):,} patients")

def gap_stats(df):
    x = df.sort_values(["person_id", "event_date"])
    g = ((x.event_date - x.groupby("person_id").event_date.shift())
         .dt.total_seconds() / 86400).dropna()
    g = g[g >= 0]
    return dict(same_day=float((g < 1).mean()), median=float(g.median()),
                mean=float(g.mean()), n=int(len(g)))

def trans_lift(df, H):
    df = df.copy(); df["grp"] = df.concept_id.map(cg_of)
    pv = df.groupby(["person_id","event_date","grp"]).size().rename("n").reset_index()
    idx = pv.groupby(["person_id","event_date"])["n"].idxmax()
    g = pv.loc[idx].sort_values(["person_id","event_date"])
    g["nx"] = g.groupby("person_id")["grp"].shift(-1)
    tr = g.dropna(subset=["nx"])
    M = np.zeros((H,H)); np.add.at(M, (tr.grp.astype(int), tr.nx.astype(int)), 1)
    p_from = M.sum(1, keepdims=True); p_to = M.sum(0, keepdims=True); N = M.sum()
    exp = p_from @ p_to / N
    return np.log2(np.maximum(M,0.5)/np.maximum(exp,0.5))

real_g, syn_g = gap_stats(ev), gap_stats(syn)
Lr, Ls = trans_lift(ev, H), trans_lift(syn, H)
iu = np.triu_indices(H, 1)
r = float(np.corrcoef(Lr[iu], Ls[iu])[0,1])
print(f"\n{'':12s} {'real':>10s} {'simulated':>10s}")
for k in ["same_day","median","mean"]:
    print(f"{k:12s} {real_g[k]:10.3f} {syn_g[k]:10.3f}")
print(f"transition lift correlation: {r:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax = axes[0]
labs = ["same day", "1-7 d", "7-30 d", "1-6 mo", "6 mo+"]
edges = [0,1,7,30,180,1e9]
for df, col, lab in [(ev, "#C1440E", "real"), (syn, "#2C6E9B", "visit chain")]:
    x = df.sort_values(["person_id","event_date"])
    g = ((x.event_date - x.groupby("person_id").event_date.shift())
         .dt.total_seconds()/86400).dropna(); g = g[g>=0]
    vals = [((g>=edges[i])&(g<edges[i+1])).mean()*100 for i in range(5)]
    ax.plot(labs, vals, "o-", color=col, label=lab, ms=4)
ax.set_ylabel("% of gaps"); ax.set_title("A. Burstiness is recovered", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
ax.scatter(Lr[iu], Ls[iu], s=12, alpha=0.7, color="#4E8B4A", edgecolor="none")
lim = [min(Lr[iu].min(), Ls[iu].min()), max(Lr[iu].max(), Ls[iu].max())]
ax.plot(lim, lim, "k:", lw=0.9)
ax.set_xlabel("real transition lift"); ax.set_ylabel("simulated")
ax.set_title(f"B. Ordering recovered (r={r:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig19_visit_chain.png", bbox_inches="tight")
print("wrote", f"{FIG}/fig19_visit_chain.png")
