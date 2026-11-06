"""
19 -- the per-domain model on the MIMIC-IV demo (OMOP CDM 5.3).

This is the first run on *observed* de-identified EHR data: the 100-patient
MIMIC-IV demo bundled in OHDSI/EunomiaDatasets, converted to OMOP by the
PhysioNet community (Kallfelz et al. 2021, doi:10.13026/p1f5-7x35).

It is not a production CDM. It is 100 ICU patients, its dates are shifted into
2100+, and its observation process is dense in-hospital monitoring, not
outpatient care. It is, however, real data, and its condition_era / drug_era
tables are populated -- so for the first time the era builder can be checked
against a real OMOP build rather than against our own reconstruction.

K is set to 5, not the specification's 20: with 100 patients a rank-20 latent
Gaussian has 210 covariance parameters estimated from 100 points. K is a
small-sample choice here, not a tuned one.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/19_mimic.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm import perdomain as pdm
from topocdm.evaluate import compare, prevalence
from topocdm.model import CDMIncidence

DATA = os.environ.get("MIMIC_DIR", "data/mimic_eunomia")
FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
    }
)
C = {
    "blue": "#2C6E9B",
    "orange": "#C1440E",
    "green": "#4E8B4A",
    "grey": "#9aa5b1",
    "red": "#B4443C",
    "purple": "#6B5B95",
}


def load_mimic(d: str):
    """Read the Eunomia MIMIC-IV demo CSVs into the per-domain model's shape."""

    def rd(t, cols, dates=()):
        return pd.read_csv(
            os.path.join(d, t + ".csv"), usecols=cols, parse_dates=list(dates), low_memory=False
        )

    cc = rd(
        "condition_occurrence",
        ["person_id", "condition_concept_id", "condition_start_date", "condition_end_date"],
        ["condition_start_date", "condition_end_date"],
    )
    de = rd(
        "drug_exposure",
        ["person_id", "drug_concept_id", "drug_exposure_start_date", "drug_exposure_end_date"],
        ["drug_exposure_start_date", "drug_exposure_end_date"],
    )
    vo = rd(
        "visit_occurrence",
        ["person_id", "visit_concept_id", "visit_start_date", "visit_end_date"],
        ["visit_start_date", "visit_end_date"],
    )
    po = rd(
        "procedure_occurrence",
        ["person_id", "procedure_concept_id", "procedure_date"],
        ["procedure_date"],
    )
    me = rd(
        "measurement",
        ["person_id", "measurement_concept_id", "measurement_date"],
        ["measurement_date"],
    )
    ob = rd(
        "observation",
        ["person_id", "observation_concept_id", "observation_date"],
        ["observation_date"],
    )
    dd = rd("death", ["person_id", "death_date"], ["death_date"])
    op = rd(
        "observation_period",
        ["person_id", "observation_period_start_date", "observation_period_end_date"],
        ["observation_period_start_date", "observation_period_end_date"],
    )

    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "person_id": cc.person_id,
                    "domain": "condition",
                    "concept_id": cc.condition_concept_id,
                    "start_date": cc.condition_start_date,
                    "end_date": cc.condition_end_date,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": de.person_id,
                    "domain": "drug",
                    "concept_id": de.drug_concept_id,
                    "start_date": de.drug_exposure_start_date,
                    "end_date": de.drug_exposure_end_date,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": vo.person_id,
                    "domain": "visit",
                    "concept_id": vo.visit_concept_id,
                    "start_date": vo.visit_start_date,
                    "end_date": vo.visit_end_date,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": po.person_id,
                    "domain": "procedure",
                    "concept_id": po.procedure_concept_id,
                    "start_date": po.procedure_date,
                    "end_date": pd.NaT,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": me.person_id,
                    "domain": "measurement",
                    "concept_id": me.measurement_concept_id,
                    "start_date": me.measurement_date,
                    "end_date": pd.NaT,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": ob.person_id,
                    "domain": "observation",
                    "concept_id": ob.observation_concept_id,
                    "start_date": ob.observation_date,
                    "end_date": pd.NaT,
                }
            ),
            pd.DataFrame(
                {
                    "person_id": dd.person_id,
                    "domain": "death",
                    "concept_id": 0,
                    "start_date": dd.death_date,
                    "end_date": pd.NaT,
                }
            ),
        ],
        ignore_index=True,
    )
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    periods = op.rename(
        columns={
            "observation_period_start_date": "start_date",
            "observation_period_end_date": "end_date",
        }
    )
    return events, periods, {"condition_occurrence": cc, "drug_exposure": de}


def _era_stats(df, scol, ecol):
    if df is None or len(df) == 0:
        return {"rows": 0, "median_days": 0.0}
    d = (pd.to_datetime(df[ecol]) - pd.to_datetime(df[scol])).dt.days
    return {"rows": int(len(df)), "median_days": float(d.median())}


def _matrix_for(records, periods, concept_ids):
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    ev = records[records["concept_id"].isin(set(concept_ids))]
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    cols = ev["concept_id"].map(ccol).to_numpy(dtype=np.int64)
    X = sparse.csr_matrix((np.ones(len(ev)), (rows, cols)), shape=(len(persons), len(concept_ids)))
    X.sum_duplicates()
    return X


def main():
    t0 = time.time()
    events, periods, src = load_mimic(DATA)
    print(f"MIMIC-IV demo: {periods.person_id.nunique()} patients, {len(events):,} records")
    print(events.groupby("domain").size().to_string())

    fit = pdm.fit_perdomain(events, periods, K=5)
    noise = 1.0 / np.sqrt(periods.person_id.nunique() * max(fit.k, 1e-9))
    print(
        f"\nfitted in {time.time() - t0:.1f}s: k={fit.k:.2f}, "
        f"propensity_tmean={fit.propensity_tmean:.2f}, "
        f"death={fit.death['frac']:.3f}, n_params={fit.n_parameters:,}, "
        f"K={fit.meta['K']}"
    )
    print(
        f"level-check noise floor: {noise:.1%} (Gamma(k={fit.k:.2f}), "
        f"n={periods.person_id.nunique()})"
    )

    # ---- generate ----
    rng = np.random.default_rng(1)
    syn_periods = pdm.sample_periods(periods, len(periods), rng)
    tables = pdm.generate_cohort(fit, len(syn_periods), syn_periods, seed=2)

    # ---- level check ----
    level = {}
    for table, dom in [
        ("condition_occurrence", "condition"),
        ("drug_exposure", "drug"),
        ("procedure_occurrence", "procedure"),
        ("measurement", "measurement"),
        ("observation", "observation"),
        ("visit_occurrence", "visit"),
    ]:
        gen = len(tables[table])
        obs = fit.n_events[dom]
        level[dom] = {"observed": obs, "generated": gen, "ratio": gen / max(obs, 1)}
    level["death"] = {
        "observed": fit.meta["n_deaths"],
        "generated": len(tables["death"]),
        "ratio": len(tables["death"]) / max(fit.meta["n_deaths"], 1),
    }
    print("\nlevel check (generated / observed):")
    for d, v in level.items():
        print(f"  {d:12s} {v['generated']:>9,} / {v['observed']:>9,}  ratio {v['ratio']:.2f}")

    # ---- summary and spectrum on the mapped clinical concepts ----
    X_src, concept_ids, persons, T_src = pdm.clinical_incidence(events, periods)
    syn_records = pdm.tables_to_records(tables)
    X_syn = _matrix_for(syn_records, syn_periods, concept_ids)
    T_syn = np.maximum(
        (syn_periods["end_date"] - syn_periods["start_date"]).dt.days.to_numpy(float) / 365.25,
        1 / 365.25,
    )
    src_inc = CDMIncidence(
        X=X_src.toarray(), T=T_src, concept_group=np.zeros(len(concept_ids), dtype=int)
    )
    syn_inc = CDMIncidence(
        X=X_syn.toarray(), T=T_syn, concept_group=np.zeros(len(concept_ids), dtype=int)
    )
    stats = compare(src_inc, syn_inc)
    s_src = pdm.spectrum(X_src)
    s_syn = pdm.spectrum(X_syn)
    # When C >> N (2,700 concepts, 100 patients) the zero-padded spectrum is
    # dominated by the shared zero mass and the KS flatters the model. Report
    # the KS on the min(N, C) non-zero singular values as the headline.
    nz = min(X_src.shape)
    ks_nz = pdm.spectral_ks(s_src[:nz], s_syn[:nz])
    ks = pdm.spectral_ks(s_src, s_syn)
    rank_src = pdm.effective_rank(s_src)
    rank_syn = pdm.effective_rank(s_syn)
    print(
        f"\nsummary: prevalence r={stats['prevalence']['r']:.3f}, "
        f"incidence r={stats['incidence']['r']:.3f}, "
        f"degree L1={stats['degree_l1']:.3f}, "
        f"co-occurrence r={stats['cooccurrence_r']:.3f}"
    )
    print(
        f"spectrum: KS(non-zero)={ks_nz:.3f} (zero-padded {ks:.3f}), "
        f"rank>0.1 source {rank_src} / synthetic {rank_syn}"
    )

    # ---- era comparison: real MIMIC build vs our builder vs generated ----
    real_ce = pd.read_csv(
        os.path.join(DATA, "condition_era.csv"),
        parse_dates=["condition_era_start_date", "condition_era_end_date"],
    )
    real_de = pd.read_csv(
        os.path.join(DATA, "drug_era.csv"), parse_dates=["drug_era_start_date", "drug_era_end_date"]
    )
    built_ce = pdm.build_eras(
        src["condition_occurrence"],
        "condition_concept_id",
        "condition_start_date",
        "condition_end_date",
        out_start="condition_era_start_date",
        out_end="condition_era_end_date",
    )
    built_de = pdm.build_eras(
        src["drug_exposure"],
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_exposure_end_date",
        out_start="drug_era_start_date",
        out_end="drug_era_end_date",
    )
    era = {
        "condition": {
            "real": _era_stats(real_ce, "condition_era_start_date", "condition_era_end_date"),
            "built": _era_stats(built_ce, "condition_era_start_date", "condition_era_end_date"),
            "generated": _era_stats(
                tables["condition_era"], "condition_era_start_date", "condition_era_end_date"
            ),
        },
        "drug": {
            "real": _era_stats(real_de, "drug_era_start_date", "drug_era_end_date"),
            "built": _era_stats(built_de, "drug_era_start_date", "drug_era_end_date"),
            "generated": _era_stats(tables["drug_era"], "drug_era_start_date", "drug_era_end_date"),
        },
    }
    dur = {}
    for dom, (tname, _ccol, scol, ecol) in {
        "condition": (
            "condition_occurrence",
            "condition_concept_id",
            "condition_start_date",
            "condition_end_date",
        ),
        "drug": (
            "drug_exposure",
            "drug_concept_id",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
        ),
    }.items():
        src_d = (pd.to_datetime(src[tname][ecol]) - pd.to_datetime(src[tname][scol])).dt.days
        dur[dom] = {
            "source_median": float(src_d.median()),
            "source_zero": float((src_d == 0).mean()),
        }

    print("\nera tables (rows / median days):")
    for dom, v in era.items():
        print(
            f"  {dom:10s} real {v['real']['rows']:>7,}/{v['real']['median_days']:5.0f}"
            f"  built {v['built']['rows']:>7,}/{v['built']['median_days']:5.0f}"
            f"  generated {v['generated']['rows']:>7,}/{v['generated']['median_days']:5.0f}"
        )

    # ---- persist ----
    out = {
        "data": {
            "patients": int(periods.person_id.nunique()),
            "records": int(len(events)),
            "per_domain": {d: int(v) for d, v in events.groupby("domain").size().items()},
        },
        "fitted": {
            "k": float(fit.k),
            "propensity_tmean": float(fit.propensity_tmean),
            "death_frac": float(fit.death["frac"]),
            "n_parameters": int(fit.n_parameters),
            "K": int(fit.meta["K"]),
            "n_atoms": int(fit.n_concepts),
        },
        "level": level,
        "summary": stats,
        "spectrum": {
            "ks": ks,
            "ks_nonzero": ks_nz,
            "rank_source": rank_src,
            "rank_synthetic": rank_syn,
            "top10_source": s_src[:10].tolist(),
            "top10_synthetic": s_syn[:10].tolist(),
        },
        "level_noise_floor": float(noise),
        "duration": dur,
        "era": era,
    }
    json.dump(
        out,
        open(f"{RES}/mimic.json", "w"),
        indent=1,
        default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
    )

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))

    ax = axes[0, 0]
    doms = list(level)
    ratios = [level[d]["ratio"] for d in doms]
    ax.bar(
        doms,
        ratios,
        color=[C["green"] if 0.85 <= r <= 1.15 else C["red"] for r in ratios],
        width=0.6,
    )
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_ylim(0, max(1.35, max(ratios) * 1.05))
    ax.set_ylabel("generated / source")
    ax.set_title("A. Level check", fontsize=9)
    ax.tick_params(axis="x", rotation=35)

    ax = axes[0, 1]
    pv, pvs = prevalence(src_inc), prevalence(syn_inc)
    ax.scatter(pv, pvs, s=8, alpha=0.5, color=C["blue"], edgecolor="none")
    lim = [0, max(pv.max(), pvs.max()) * 1.05]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("source prevalence")
    ax.set_ylabel("synthetic prevalence")
    ax.set_title(
        f"B. Prevalence r={stats['prevalence']['r']:.2f}; co-occ r={stats['cooccurrence_r']:.2f}",
        fontsize=9,
    )

    ax = axes[1, 0]
    k = np.arange(1, min(41, len(s_src) + 1))
    ax.plot(k, s_src[:40], "o-", ms=2.8, color=C["blue"], label="MIMIC")
    ax.plot(k, s_syn[:40], "s-", ms=2.8, color=C["orange"], label="synthetic")
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.set_title(f"C. Spectrum KS={ks_nz:.2f}; rank {rank_src} vs {rank_syn}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    groups = ["condition", "drug"]
    x = np.arange(len(groups))
    w = 0.26
    for off, key, col, lab in [
        (-1, "real", C["grey"], "MIMIC build"),
        (0, "built", C["green"], "our builder"),
        (1, "generated", C["orange"], "generated"),
    ]:
        vals = [era[g][key]["rows"] for g in groups]
        ax.bar(x + off * w, vals, width=w, color=col, label=lab)
        for xi, v in zip(x + off * w, vals):
            ax.text(xi, v * 1.02, f"{v:,}", ha="center", fontsize=6, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("era rows")
    ax.set_title("D. Era tables: real, rebuilt, generated", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig29_mimic.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig29_mimic.png and {RES}/mimic.json")


if __name__ == "__main__":
    main()
