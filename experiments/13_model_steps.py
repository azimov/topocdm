"""
13 -- the model in six steps, drawn in network terms.

A conceptual figure, not a data figure: it walks from the bipartite graph to the
generative rule and the fitting loop.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

FIG = "figs"
plt.rcParams.update({"figure.dpi": 170, "font.size": 10.5})
INK = "#1f2933"
BLUE = "#2C6E9B"
ORANGE = "#C1440E"
GREEN = "#4E8B4A"
GREY = "#9aa5b1"
PURPLE = "#6B5B95"

rng = np.random.default_rng(3)
NP, NC = 8, 6
py = np.linspace(0.05, 0.95, NP)
cy = np.linspace(0.1, 0.9, NC)
patient_g = np.array([0, 1, 1, 0, 2, 2, 0, 1])
concept_g = np.array([0, 0, 1, 1, 2, 2])
GC = [BLUE, ORANGE, GREEN]

# degree pattern: patient i connects to concepts in its own group mostly
E = []
for i in range(NP):
    for j in range(NC):
        same = patient_g[i] == concept_g[j]
        if rng.random() < (0.75 if same else 0.18):
            E.append((i, j))

deg_p = np.array([sum(1 for a, b in E if a == i) for i in range(NP)])
deg_c = np.array([sum(1 for a, b in E if b == j) for j in range(NC)])

fig, axes = plt.subplots(3, 2, figsize=(8.4, 10.6))
axes = axes.ravel()


def draw_graph(
    ax,
    psize=None,
    csize=None,
    pcol=None,
    ccol=None,
    ecol=GREY,
    alpha=0.75,
    lw=1.0,
    labels=False,
    highlight=None,
):
    for a, b in E:
        c = BLUE if highlight == (a, b) else ecol
        w = 3.0 if highlight == (a, b) else lw
        ax.plot([0, 1], [py[a], cy[b]], color=c, lw=w, alpha=alpha, zorder=1)
    ps = psize if psize is not None else np.full(NP, 160)
    cs = csize if csize is not None else np.full(NC, 160)
    pcol = pcol if pcol is not None else np.full(NP, GREY)
    ccol = ccol if ccol is not None else np.full(NC, GREY)
    for i in range(NP):
        ax.add_patch(Circle((0, py[i]), 0.0, color=pcol[i]))
        ax.scatter(0, py[i], s=ps[i], color=pcol[i], zorder=3, edgecolor="white", lw=1.2)
    for j in range(NC):
        ax.scatter(
            1, cy[j], s=cs[j], color=ccol[j], marker="s", zorder=3, edgecolor="white", lw=1.2
        )
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.text(0, -0.03, "patients", ha="center", fontsize=7.5, color=INK)
    ax.text(1, -0.03, "concepts", ha="center", fontsize=7.5, color=INK)
    if labels:
        for i in range(NP):
            ax.text(-0.06, py[i], f"p{i + 1}", ha="right", va="center", fontsize=6.5, color=INK)
        for j in range(NC):
            ax.text(1.06, cy[j], f"c{j + 1}", ha="left", va="center", fontsize=6.5, color=INK)


# 1 -- the object
ax = axes[0]
draw_graph(ax, labels=True)
ax.set_title(
    "1. The object: a bipartite graph\npatients | concepts; one edge = at least one record",
    fontsize=10,
)

# 2 -- patient volume
ax = axes[1]
draw_graph(ax, psize=40 + 40 * deg_p, pcol=np.full(NP, BLUE))
ax.set_title("2. Each patient gets a volume, $u_i$\nnode size = how many edges it has", fontsize=10)

# 3 -- concept frequency
ax = axes[2]
draw_graph(ax, csize=40 + 40 * deg_c, ccol=np.full(NC, ORANGE))
ax.set_title(
    "3. Each concept gets a frequency, $\\lambda_j$\nsquare size = how often it appears",
    fontsize=10,
)

# 4 -- block structure
ax = axes[3]
draw_graph(ax, pcol=[GC[g] for g in patient_g], ccol=[GC[g] for g in concept_g])
B = np.array([[1.6, 0.5, 0.4], [0.5, 1.7, 0.4], [0.4, 0.5, 1.5]])
inset = ax.inset_axes([0.34, 0.46, 0.34, 0.30])
inset.imshow(B, cmap="Blues", vmin=0, vmax=2)
inset.set_xticks([])
inset.set_yticks([])
inset.set_title("$B[g,h]$", fontsize=7, pad=2)
for sp in inset.spines.values():
    sp.set_edgecolor("#C9D1D9")
ax.set_title("4. Patients and concepts are grouped;\n$B$ says which groups attract", fontsize=10)

# 5 -- the rule
ax = axes[4]
hi = (2, 1)
draw_graph(
    ax, pcol=[GC[g] for g in patient_g], ccol=[GC[g] for g in concept_g], highlight=hi, alpha=0.45
)
ax.set_ylim(-0.30, 1.02)
ax.text(
    0.55,
    -0.17,
    "expected records for p3–c2\n"
    "= time at risk $\\times\\ u_{p3}\\ \\times\\ \\lambda_{c2}\\ \\times\\ B[g_{p3},h_{c2}]$",
    ha="center",
    fontsize=7.6,
    color=INK,
    bbox={"boxstyle": "round,pad=0.4", "fc": "#F4F6F8", "ec": "#C9D1D9", "lw": 0.8},
)
ax.set_title("5. The generative rule\none number per patient–concept pair", fontsize=10)

# 6 -- the fitting loop
ax = axes[5]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
b1 = FancyBboxPatch(
    (0.06, 0.60), 0.88, 0.18, boxstyle="round,pad=0.012", fc="#EAF1F7", ec=BLUE, lw=1.0
)
b2 = FancyBboxPatch(
    (0.06, 0.20), 0.88, 0.18, boxstyle="round,pad=0.012", fc="#FBEEE8", ec=ORANGE, lw=1.0
)
ax.add_patch(b1)
ax.add_patch(b2)
ax.text(
    0.5,
    0.69,
    "guess each patient's group\nfrom the current parameters",
    ha="center",
    va="center",
    fontsize=7.6,
)
ax.text(
    0.5,
    0.29,
    "refit $u$, $\\lambda$, $B$ from the guesses\n(counting: observed ÷ expected)",
    ha="center",
    va="center",
    fontsize=7.6,
)
ax.add_patch(
    FancyArrowPatch(
        (0.86, 0.60), (0.86, 0.38), arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.2
    )
)
ax.add_patch(
    FancyArrowPatch(
        (0.14, 0.38), (0.14, 0.60), arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.2
    )
)
ax.text(
    0.5,
    0.07,
    "~15 passes · each pass never lowers the likelihood",
    ha="center",
    fontsize=7.2,
    color=GREY,
)
ax.set_title("6. Fitting: alternate until nothing changes", fontsize=9.5, pad=8)

fig.tight_layout(rect=(0, 0, 1, 1))
fig.savefig(f"{FIG}/fig24_model_steps.png", bbox_inches="tight", facecolor="white")
print("wrote", f"{FIG}/fig24_model_steps.png")
