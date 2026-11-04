"""
11 -- Synthea patients as networks, with time.

Any many-patient bipartite plot is a hairball: a patient has ~78 of 773
concepts, so the graph saturates. These three figures each avoid that by
changing the question:

fig20  one patient's own bipartite network, concepts on a circle ordered by when
       they first appeared, edge colour = year. The patient is the hub.
fig21  forty patients as rows, time as x: where records bunch into encounters.
fig22  one patient's *first occurrences* only, so each concept appears once and
       the arrows show the order in which conditions entered the record.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

FIG = "figs"
plt.rcParams.update({"figure.dpi": 170, "font.size": 8})

from topocdm.io import load_incidence
d, groups, cv = load_incidence()
ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"])
ev = ev[ev.concept_id.isin(set(d.concept_ids))]
cv2 = pd.read_csv("data/concepts.csv", usecols=["concept_id", "concept_name"])
names = cv2.set_index("concept_id")["concept_name"].to_dict()
cg_of = dict(zip(d.concept_ids, d.concept_group))

# ---------------------------------------------------------------- fig 20
pids = [1, 3, 247]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4),
                         subplot_kw=dict(aspect="equal"))
for ax, pid in zip(axes, pids):
    one = ev[ev.person_id == pid]
    first = one.groupby("concept_id").event_date.min().sort_values()
    n = len(first)
    ang = -np.pi / 2 + 2 * np.pi * np.arange(n) / n
    xs, ys = np.cos(ang), np.sin(ang)
    t0, t1 = one.event_date.min(), one.event_date.max()
    for k, (c, dt) in enumerate(first.items()):
        col = plt.cm.viridis((dt.year - 2016) / 9)
        ax.plot([0, xs[k]], [0, ys[k]], color=col, lw=0.8, alpha=0.8)
        ax.scatter(xs[k], ys[k], s=14, color=col, zorder=3)
    ax.scatter(0, 0, s=190, color="#1f2933", zorder=4)
    ax.text(0, 0, str(pid), color="white", ha="center", va="center",
            fontsize=8, zorder=5)
    for k in range(0, n, max(1, n // 6)):
        rot = np.degrees(ang[k]) + 90
        if xs[k] < 0:
            rot += 180
        ax.text(xs[k] * 1.13, ys[k] * 1.13,
                names.get(first.index[k], "")[:24], fontsize=5.5,
                ha="center", va="center", rotation=rot,
                rotation_mode="anchor")
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.42, 1.42); ax.axis("off")
    ax.set_title(f"patient {pid}\n{n} concepts, "
                 f"{t0.year}-{t1.year}", fontsize=8.5)
fig.suptitle("A patient's own bipartite network: hub = patient, spokes = concepts "
             "ordered by first appearance, colour = year", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{FIG}/fig20_patient_hub.png", bbox_inches="tight")
plt.close(fig)
print("fig20")

# ---------------------------------------------------------------- fig 21
rng = np.random.default_rng(1)
samp = rng.choice(ev.person_id.unique(), size=40, replace=False)
# the extract is only complete from 2016, so show that window
sub = ev[ev.person_id.isin(samp) & ev.event_date.dt.year.between(2016, 2025)].copy()
sub["order"] = sub.person_id.map({p: i for i, p in enumerate(samp)})
sub = sub.sort_values(["order", "event_date"])
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.scatter(sub.event_date, sub["order"],
           c=[plt.cm.tab20(cg_of[c] % 20) for c in sub.concept_id],
           s=7, alpha=0.55, linewidths=0)
ax.set_xlabel("date (2016-2025, the complete window)"); ax.set_ylabel("patient (40 sampled at random)")
ax.set_yticks([])
ax.set_title("Where records bunch: each dot is one record, colour is concept group\n"
             "the bursts are encounters — 91% of gaps between a patient's records "
             "are under a day", fontsize=10)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig21_temporal_raster.png", bbox_inches="tight")
plt.close(fig)
print("fig21")

# ---------------------------------------------------------------- fig 22
pid = 1
one = ev[ev.person_id == pid]
first = one.groupby("concept_id").event_date.min().sort_values()
n = len(first)
ang = -np.pi / 2 + 2 * np.pi * np.arange(n) / n
pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(first.index, ang)}
fig, ax = plt.subplots(figsize=(8.8, 8.2))
order = list(first.index)
for a, b in zip(order, order[1:]):
    x0, y0 = pos[a]; x1, y1 = pos[b]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, alpha=0.55,
                                 color=plt.cm.viridis((first[b].year - 2016) / 9)))
for c in order:
    ax.scatter(*pos[c], s=26, color=plt.cm.tab20(cg_of[c] % 20),
               edgecolor="white", linewidth=0.8, zorder=3)
step = max(1, n // 22)
for k in range(0, n, step):
    c = order[k]
    x, y = pos[c]
    ax.text(x * 1.12, y * 1.12, names.get(c, "")[:26], fontsize=6,
            ha="center", va="center")
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
ax.axis("off")
ax.set_title(f"Patient {pid}: the order conditions entered the record\n"
             f"{n} first occurrences, {first.index[0] and first.min().year}"
             f"-{first.max().year}; arrows follow time clockwise", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/fig22_first_occurrence.png", bbox_inches="tight")
plt.close(fig)
print("fig22")
