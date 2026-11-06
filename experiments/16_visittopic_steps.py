"""
16 -- the visit-anchored model in six steps.

Conceptual figure: the object is a stream of visits per patient, a visit is a set
of concepts at one time, visits are drawn from types, and the types follow a
Markov chain. Time is the second axis of the object, not a covariate.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch

FIG = "figs"
plt.rcParams.update({"figure.dpi": 170, "font.size": 10.5})
INK = "#1f2933"
BLUE = "#2C6E9B"
ORANGE = "#C1440E"
GREEN = "#4E8B4A"
GREY = "#9aa5b1"
PURPLE = "#6B5B95"
TC = ["#2C6E9B", "#C1440E", "#4E8B4A", "#6B5B95", "#B8860B"]

rng = np.random.default_rng(11)
fig, axes = plt.subplots(3, 2, figsize=(8.4, 10.6))
axes = axes.ravel()

# ---------------------------------------------------------------- 1
ax = axes[0]
for i in range(3):
    t = np.sort(rng.uniform(0, 1, size=[14, 7, 11][i]))
    ax.plot([0, 1], [2 - i, 2 - i], color="#E4E7EB", lw=2, zorder=1)
    ax.scatter(t, np.full(len(t), 2 - i), s=34, color=BLUE, zorder=3, edgecolor="white", lw=0.8)
    ax.text(-0.04, 2 - i, f"patient {i + 1}", ha="right", va="center", fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.4, 2.4)
ax.axis("off")
ax.annotate(
    "",
    xy=(1.0, -0.25),
    xytext=(0.0, -0.25),
    arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.1},
)
ax.text(0.5, -0.38, "time", ha="center", fontsize=8, color=GREY)
ax.text(0.5, 2.32, "1. The object is a stream of visits", ha="center", fontsize=10, weight="bold")

# ---------------------------------------------------------------- 2
ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(
    0.5,
    0.95,
    "2. A visit is a set of concepts at one time",
    ha="center",
    fontsize=10,
    weight="bold",
)
ax.plot([0.5, 0.5], [0.14, 0.80], color="#E4E7EB", lw=2)
ax.scatter(0.5, 0.16, s=150, color=BLUE, zorder=3, edgecolor="white", lw=1.2)
ax.text(0.5, 0.06, "visit v, time t_v", ha="center", fontsize=8)
for k, y in enumerate(np.linspace(0.34, 0.80, 6)):
    ax.plot([0.5, 0.82], [0.16, y], color=GREY, lw=1.0, alpha=0.8, zorder=1)
    ax.scatter(0.84, y, s=70, marker="s", color=ORANGE, zorder=3, edgecolor="white", lw=1.0)
    ax.text(0.88, y, f"concept {k + 1}", va="center", fontsize=7)
ax.text(0.22, 0.34, "the visit is the\nunit of observation", ha="center", fontsize=7.5, color=GREY)

# ---------------------------------------------------------------- 3
ax = axes[2]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(
    0.5, 0.95, "3. Each visit is drawn from a visit type", ha="center", fontsize=10, weight="bold"
)
th = rng.dirichlet(np.r_[np.ones(3) * 3, np.ones(7) * 0.25], size=4)
ax_ins = ax.inset_axes([0.06, 0.30, 0.52, 0.55])
ax_ins.imshow(th, cmap="Blues", aspect="auto", vmin=0)
ax_ins.set_yticks(range(4))
ax_ins.set_yticklabels([f"type {i + 1}" for i in range(4)], fontsize=6.5)
ax_ins.set_xticks([])
ax_ins.set_xlabel("concepts →", fontsize=6.5, labelpad=1)
ax_ins.tick_params(length=0)
ax_ins.set_title("catalogue $\\theta_k$", fontsize=7, pad=2)
for k in range(4):
    ax.scatter(0.72, 0.72 - 0.12 * k, s=90, color=TC[k], edgecolor="white", lw=1)
    ax.text(0.76, 0.72 - 0.12 * k, f"type {k + 1}", va="center", fontsize=7)

# ---------------------------------------------------------------- 4
ax = axes[3]
ax.axis("off")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.55, 1.35)
ax.text(
    0,
    1.30,
    "4. Types follow a Markov chain — this is the time dynamic",
    ha="center",
    fontsize=10,
    weight="bold",
)
K = 5
ang = -np.pi / 2 + 2 * np.pi * np.arange(K) / K
pos = [(np.cos(a), np.sin(a)) for a in ang]
T = np.array(
    [
        [0.62, 0.15, 0.10, 0.08, 0.05],
        [0.12, 0.58, 0.16, 0.09, 0.05],
        [0.08, 0.14, 0.60, 0.12, 0.06],
        [0.07, 0.09, 0.13, 0.59, 0.12],
        [0.06, 0.05, 0.08, 0.15, 0.66],
    ]
)
for i in range(K):
    for j in range(K):
        if i == j:
            continue
        if T[i, j] < 0.08:
            continue
        ax.add_patch(
            FancyArrowPatch(
                pos[i],
                pos[j],
                connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>",
                mutation_scale=9,
                color=TC[i],
                lw=0.6 + 6 * T[i, j],
                alpha=0.55,
            )
        )
    ax.add_patch(
        Arc(
            pos[i],
            0.42,
            0.42,
            angle=np.degrees(ang[i]),
            theta1=20,
            theta2=160,
            color=TC[i],
            lw=0.8 + 6 * T[i, i],
            alpha=0.8,
        )
    )
    ax.scatter(*pos[i], s=330, color=TC[i], zorder=3, edgecolor="white", lw=1.4)
    ax.text(
        pos[i][0] * 1.28, pos[i][1] * 1.28, f"$z_{i + 1}$", ha="center", va="center", fontsize=8
    )
ax.text(
    0,
    -1.44,
    "$z_v \\mid z_{v-1} \\sim T[z_{v-1},\\cdot]$   ·   mostly stays put",
    ha="center",
    fontsize=7.5,
    color=GREY,
)

# ---------------------------------------------------------------- 5
ax = axes[4]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.5, 0.95, "5. What the patient accumulates", ha="center", fontsize=10, weight="bold")
ax.set_ylim(0.05, 1.0)
ax.text(
    0.5,
    0.62,
    r"$E[X_{ij}] \;=\; \sum_k \; c_{ik}\, \theta_{kj}$" + "\n\n"
    r"$c_{ik}$ = visits of type $k$" + "\n"
    r"$\theta_{kj}$ = concepts a type-$k$ visit draws",
    ha="center",
    va="center",
    fontsize=11,
    color=INK,
    bbox={"boxstyle": "round,pad=0.5", "fc": "#F4F6F8", "ec": "#C9D1D9", "lw": 0.9},
)
ax.text(
    0.5,
    0.22,
    "a sum over visits, so the rank\nis at most the number of types",
    ha="center",
    fontsize=7.5,
    color=GREY,
)

# ---------------------------------------------------------------- 6
ax = axes[5]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("6. What it costs, and how it is fitted", fontsize=10, weight="bold", pad=8)
ax.text(
    0.5,
    0.80,
    r"$\theta:\ K\times C$      $T:\ K\times K$      $\rho:\ K$      $+\ 1$",
    ha="center",
    fontsize=10.5,
    bbox={"boxstyle": "round,pad=0.35", "fc": "#EAF1F7", "ec": BLUE, "lw": 0.9},
)
ax.text(0.5, 0.66, "no term in the number of patients", ha="center", fontsize=7.8, color=GREY)
b1 = FancyBboxPatch(
    (0.08, 0.38), 0.84, 0.15, boxstyle="round,pad=0.012", fc="#FBEEE8", ec=ORANGE, lw=1.0
)
b2 = FancyBboxPatch(
    (0.08, 0.12), 0.84, 0.15, boxstyle="round,pad=0.012", fc="#EAF1F7", ec=BLUE, lw=1.0
)
ax.add_patch(b1)
ax.add_patch(b2)
ax.text(
    0.5, 0.455, "E: each visit's type, given the catalogues", ha="center", va="center", fontsize=7.6
)
ax.text(
    0.5, 0.195, "M: refit the catalogues from the types", ha="center", va="center", fontsize=7.6
)
ax.add_patch(
    FancyArrowPatch(
        (0.88, 0.38), (0.88, 0.28), arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.1
    )
)
ax.add_patch(
    FancyArrowPatch(
        (0.12, 0.28), (0.12, 0.38), arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.1
    )
)
ax.text(0.5, 0.03, "1 sparse matrix product per pass", ha="center", fontsize=7, color=GREY)

fig.tight_layout(h_pad=1.4)
fig.savefig(f"{FIG}/fig26_visittopic_steps.png", bbox_inches="tight", facecolor="white")
print("wrote", f"{FIG}/fig26_visittopic_steps.png")
