"""
18 -- the negative-control experiment: the kill criterion.

Take drug-outcome pairs with no plausible causal relationship and estimate the
association three ways:

  1. raw co-occurrence odds ratio;
  2. adjusted for patient contact volume (the fitted u_i from the per-domain
     model);
  3. OHDSI-style empirical calibration: fit a null distribution N(mu, sigma^2)
     to the raw log-odds-ratios, accounting for their sampling standard errors,
     and ask what the negative controls look like under it.

The pairs are cross-system by construction: chronic cardiovascular / metabolic /
anticoagulant drugs against adult conditions they cannot plausibly cause. The
analysis is restricted to an adult cohort (age >= 18 at the observation-period
midpoint), because Synthea's acute conditions are heavily paediatric and pairing
them with adult drugs would test age, not ascertainment.

DATA NOTE: this is the Synthea CDM slice, a rule-based patient simulator. It is
NOT a production CDM and NOT observed patients. It is the right place to test the
MECHANISM, because the ground truth (no causal link) is known. The same script
runs against a production database by pointing scripts/extract_dated.sh at it.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/18_negative_control.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm import perdomain as pdm

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22,
                     "grid.linewidth": 0.5})
C = dict(ink="#1f2933", blue="#2C6E9B", orange="#C1440E", green="#4E8B4A",
         grey="#9aa5b1", red="#B4443C", purple="#6B5B95")

# Chronic systemic drugs (cardio / metabolic / anticoagulant / diuretic). None
# has a plausible causal effect on the adult outcomes below.
DRUGS = {
    19078106: "hydrochlorothiazide",
    19080128: "lisinopril",
    19073094: "amlodipine",
    1539463: "simvastatin 10mg",
    1539411: "simvastatin 20mg",
    40163924: "metformin",
    40163554: "warfarin",
    19018935: "digoxin",
    19075601: "clopidogrel",
    40171902: "verapamil",
    957136: "furosemide",
    19126352: "nitroglycerin",
}
# Adult conditions with no plausible causal link to those drugs.
OUTCOMES = {
    81151: "Sprain of ankle",
    4112343: "Acute viral pharyngitis",
    4132891: "Chronic low back pain",
    43530622: "Chronic neck pain",
    4214746: "Severe anxiety (panic)",
}


def _indicator(events, domain, concept_id, persons):
    s = set(events[(events["domain"] == domain)
                   & (events["concept_id"] == concept_id)]["person_id"])
    return np.array([p in s for p in persons])


def _logit_fit(y, X, n_iter=100, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-9)
        z = eta + (y - p) / w
        A = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        b_new = np.linalg.solve(A, X.T @ (w * z))
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    eta = X @ b
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    w = np.maximum(p * (1 - p), 1e-9)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def _raw_or(y, e):
    a = float(np.sum(y & e)); b = float(np.sum(~y & e))
    c = float(np.sum(y & ~e)); d = float(np.sum(~y & ~e))
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float(np.log((a * d) / (b * c))), float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))


def _mh_or(B, strata):
    """Cochran-Mantel-Haenszel OR for every concept pair, stratified."""
    nc = B.shape[1]
    num = np.zeros((nc, nc))
    den = np.zeros((nc, nc))
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 5:
            continue
        Bs = B[m]
        n = Bs.sum(axis=0)
        both = Bs.T @ Bs
        N = float(m.sum())
        a = both
        b = n[None, :] - both
        c = n[:, None] - both
        d = N - n[:, None] - n[None, :] + both
        num += a * d / N
        den += b * c / N
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _raw_or_matrix(B):
    n = B.sum(axis=0)
    both = B.T @ B
    N = float(B.shape[0])
    a = both
    b = n[None, :] - both
    c = n[:, None] - both
    d = N - n[:, None] - n[None, :] + both
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a * d) / (b * c)


def _qstrata(x, k=5):
    q = np.quantile(x, np.linspace(0, 1, k + 1))
    return np.clip(np.digitize(x, q[1:-1]), 0, k - 1)


def _empirical_calibration(theta, se):
    theta = np.asarray(theta, float)
    se = np.asarray(se, float)

    def nll(p):
        mu, log_sigma = p
        s2 = np.exp(2 * log_sigma) + se ** 2
        return 0.5 * float(np.sum(np.log(s2) + (theta - mu) ** 2 / s2))

    r = optimize.minimize(nll, x0=[float(np.mean(theta)), np.log(0.2)],
                          method="Nelder-Mead")
    mu, sigma = float(r.x[0]), float(np.exp(r.x[1]))
    sd = np.sqrt(sigma ** 2 + se ** 2)
    p_raw = 2 * (1 - stats.norm.cdf(np.abs(theta / se)))
    p_cal = 2 * (1 - stats.norm.cdf(np.abs((theta - mu) / sd)))
    return dict(mu=mu, sigma=sigma, p_raw=p_raw, p_cal=p_cal)


def main():
    print("=== NEGATIVE-CONTROL EXPERIMENT (kill criterion) ===")
    print("Data: Synthea CDM slice -- a rule-based simulator, NOT a production CDM.")
    print("True causal effect for these cross-system pairs is null by construction.")
    print("Cohort restricted to adults (age >= 18 at observation midpoint).\n")

    events, periods = pdm.load_extract()
    person = pd.read_csv("data/person.csv", usecols=["person_id", "year_of_birth"])
    birth = person.set_index("person_id")["year_of_birth"]

    fit = pdm.fit_perdomain(events, periods, K=0)   # only need contact volume
    persons = np.asarray(fit.person_ids)
    u = fit.u
    u_of = dict(zip(persons, u))

    pe = periods.set_index("person_id").loc[persons]
    age_mid = ((pe["start_date"].dt.year + pe["end_date"].dt.year) / 2
               - birth.reindex(persons).to_numpy())
    adult = age_mid >= 18
    print(f"adult cohort: {int(adult.sum()):,} of {len(persons):,} patients")

    log_u = np.log(np.maximum(np.array([u_of[p] for p in persons]), 1e-3))
    age = age_mid.astype(float)

    rows = []
    for dcid, dname in DRUGS.items():
        e_all = _indicator(events, "drug", dcid, persons)
        for ocid, oname in OUTCOMES.items():
            y_all = _indicator(events, "condition", ocid, persons)
            y, e, lu, ag = y_all[adult], e_all[adult], log_u[adult], age[adult]
            if y.sum() < 20 or e.sum() < 20:
                continue

            raw, raw_se = _raw_or(y, e)

            # 2. adjusted for contact volume
            ba, sea = _logit_fit(y, np.column_stack([np.ones(len(y)), e, lu]))
            adj, adj_se = float(ba[1]), float(sea[1])

            # diagnostic: adjusted for age, and for age + contact
            bage, seage = _logit_fit(y, np.column_stack([np.ones(len(y)), e, ag]))
            bau, seau = _logit_fit(y, np.column_stack([np.ones(len(y)), e, lu, ag]))

            rows.append(dict(drug=dname, outcome=oname, n_drug=int(e.sum()),
                             n_outcome=int(y.sum()),
                             raw_log_or=raw, raw_se=raw_se,
                             adj_log_or=adj, adj_se=adj_se,
                             age_log_or=float(bage[1]), age_se=float(seage[1]),
                             age_u_log_or=float(bau[1]), age_u_se=float(seau[1])))
    tab = pd.DataFrame(rows)
    print(f"{len(tab)} negative-control pairs\n")

    cal = _empirical_calibration(tab["raw_log_or"].to_numpy(),
                                 tab["raw_se"].to_numpy())
    tab["p_raw"] = cal["p_raw"]
    tab["p_cal"] = cal["p_cal"]

    def summarise(col):
        x = tab[col].to_numpy()
        return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                    sd=float(np.std(x)), mean_or=float(np.exp(np.mean(x))),
                    frac_gt_1_2=float((np.exp(x) > 1.2).mean()),
                    frac_lt_0_8=float((np.exp(x) < 0.8).mean()))

    raw_s, adj_s = summarise("raw_log_or"), summarise("adj_log_or")
    age_s, ageu_s = summarise("age_log_or"), summarise("age_u_log_or")
    moved = tab["raw_log_or"].to_numpy() - tab["adj_log_or"].to_numpy()

    print("association across negative controls (log OR; OR = exp):")
    for name, s in [("raw", raw_s), ("contact-volume adjusted", adj_s),
                    ("age adjusted (diagnostic)", age_s),
                    ("age + contact adjusted", ageu_s)]:
        print(f"  {name:28s} mean {s['mean']:+.3f} (OR {s['mean_or']:.2f})"
              f"  median {s['median']:+.3f}  sd {s['sd']:.3f}")
    print(f"\n  bias removed by u_i adjustment: mean {np.mean(moved):+.3f} "
          f"(OR factor {np.exp(np.mean(moved)):.2f})")
    print(f"  raw p<0.05: {np.mean(tab['p_raw'] < 0.05):.1%} of pairs")
    print(f"  calibrated p<0.05: {np.mean(tab['p_cal'] < 0.05):.1%} of pairs")
    print(f"  calibration: systematic-error mean mu={cal['mu']:+.3f} "
          f"(OR {np.exp(cal['mu']):.2f}), sigma={cal['sigma']:.3f}")

    # per-outcome breakdown: where is the signal?
    print("\nper-outcome mean log OR (raw / contact-adjusted):")
    per_out = []
    for oname, g in tab.groupby("outcome"):
        per_out.append(dict(outcome=oname, n=len(g),
                            raw=float(g.raw_log_or.mean()),
                            adj=float(g.adj_log_or.mean()),
                            age=float(g.age_log_or.mean())))
        print(f"  {oname:24s} n={len(g):2d}  raw {g.raw_log_or.mean():+.3f}"
              f"  adj {g.adj_log_or.mean():+.3f}  age-adj {g.age_log_or.mean():+.3f}")

    # ---- Part B: where does the association the model removes come from? ----
    # The adult drug-outcome pairs are already null, so they cannot show bias
    # removal. The stronger test is condition-condition co-occurrence, which is
    # where the earlier work saw median raw OR 1.71 -> 0.91. Decompose it into
    # contact-volume versus age: if the two stratifications give the same
    # answer, the model's u_i is an age proxy, not a distinct bias channel.
    print("\nwhere the association is: condition-condition pairs")
    cond = events[events["domain"] == "condition"]
    prev = cond.groupby("concept_id")["person_id"].nunique() / len(persons)
    sel = (prev[(prev > 0.02) & (prev < 0.7)]
           .sort_values(ascending=False).head(30).index.tolist())
    B = np.zeros((len(persons), len(sel)), dtype=np.float32)
    for j, c in enumerate(sel):
        s = set(cond[cond.concept_id == c]["person_id"])
        B[:, j] = [p in s for p in persons]
    cv = (events[events["domain"].isin(pdm.CLINICAL_DOMAINS)]
          .groupby("person_id").size().reindex(persons, fill_value=0)
          .to_numpy(float))
    mech = {}
    for label, mask in [("all ages", np.ones(len(persons), bool)),
                        ("adult", adult)]:
        Bm = B[mask]
        r = _raw_or_matrix(Bm)
        mu = _mh_or(Bm, _qstrata(cv[mask]))
        ma = _mh_or(Bm, _qstrata(age[mask]))
        iu = np.triu_indices(len(sel), 1)
        both = (Bm.T @ Bm)[iu]
        ok = (np.isfinite(r[iu]) & np.isfinite(mu[iu]) & np.isfinite(ma[iu])
              & (both >= 25) & (r[iu] > 0) & (r[iu] < 256)
              & (mu[iu] > 0) & (ma[iu] > 0))
        mech[label] = dict(n_pairs=int(ok.sum()),
                           raw=float(np.median(r[iu][ok])),
                           u=float(np.median(mu[iu][ok])),
                           age=float(np.median(ma[iu][ok])))
        print(f"  {label:9s} n={int(mask.sum()):5d} pairs={ok.sum():4d}  "
              f"median OR raw {mech[label]['raw']:.2f}  "
              f"u-stratified {mech[label]['u']:.2f}  "
              f"age-stratified {mech[label]['age']:.2f}")

    # ---- verdict against the stated kill criterion ----
    adj_mean, cal_mu, raw_mean = adj_s["mean"], cal["mu"], raw_s["mean"]
    lands_null = abs(adj_mean) < 0.15
    cal_null = np.mean(tab["p_cal"] < 0.05) < 0.10
    a = mech["all ages"]
    bias_removed = a["raw"] - a["u"]
    age_equiv = abs(a["u"] - a["age"]) < 0.06
    adult_null = abs(mech["adult"]["raw"] - 1.0) < 0.10
    agrees = lands_null and cal_null

    print("\n" + "=" * 72)
    print(f"adult drug-outcome: raw mean log-OR {raw_mean:+.3f}, "
          f"contact-adjusted {adj_mean:+.3f}, calibration mu {cal_mu:+.3f}")
    print(f"condition-condition all ages: raw {a['raw']:.2f} -> u {a['u']:.2f}, "
          f"age {a['age']:.2f}")
    if adult_null and bias_removed > 0.2 and age_equiv:
        print("VERDICT: the mechanism works but it is age. On all-ages data the")
        print("contact-volume adjustment moves the median OR from %.2f to %.2f," % (a["raw"], a["u"]))
        print("so the kill criterion is not triggered -- but age stratification")
        print("gives %.2f, indistinguishable. The bias u_i removes is age," % a["age"])
        print("which any age adjustment also removes. Claim 2 survives only in")
        print("the weak sense (u_i is a valid proxy for the observation process),")
        print("not the strong sense (a bias channel conventional adjustment misses).")
    elif not bias_removed > 0.1:
        print("VERDICT: kill. Contact-volume adjustment does not move the known-null")
        print("association toward the null (%.2f -> %.2f). Claim 2 is dead on this test," % (a["raw"], a["u"]))
        print("and claims 3 and 4 with it.")
    elif agrees:
        print("VERDICT: agreement at the null, but the adult pairs are already null,")
        print("so this demonstrates non-interference, not bias removal. The all-ages")
        print("condition pairs show the adjustment working (%.2f -> %.2f); age-" % (a["raw"], a["u"]))
        print("stratification gives %.2f, so the bias is age. Claim 2 is weakly" % a["age"])
        print("supported, not established as a distinct contribution.")
    else:
        print("VERDICT: disagreement. Adj %.2f, calibration %.2f, bias removed %.2f." % (adj_mean, cal_mu, bias_removed))
        print("Per the stated kill criterion, claim 2 is not supported.")
    print("=" * 72)

    json.dump(dict(pairs=tab.to_dict(orient="records"),
                   summary=dict(raw=raw_s, adjusted=adj_s, age=age_s,
                                age_u=ageu_s, moved=float(np.mean(moved)),
                                cal_mu=cal["mu"], cal_sigma=cal["sigma"],
                                raw_p05=float(np.mean(tab["p_raw"] < 0.05)),
                                cal_p05=float(np.mean(tab["p_cal"] < 0.05)),
                                per_outcome=per_out, mech=mech,
                                lands_null=bool(lands_null),
                                cal_null=bool(cal_null), agrees=bool(agrees),
                                adult_null=bool(adult_null),
                                age_equiv=bool(age_equiv)),
                   note=("Synthea CDM slice (rule-based simulator, not a production CDM). "
                         "Adult cohort; cross-system drug-outcome pairs with null "
                         "true effect.")),
              open(f"{RES}/negative_control.json", "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else str(o))

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))

    ax = axes[0, 0]
    bins = np.linspace(-1.2, 1.2, 25)
    ax.hist(tab["raw_log_or"], bins=bins, alpha=0.55, color=C["red"],
            label="raw", edgecolor="white", linewidth=0.3)
    ax.hist(tab["adj_log_or"], bins=bins, alpha=0.55, color=C["green"],
            label="adjusted for u_i", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.set_xlabel("log odds ratio")
    ax.set_ylabel("negative-control pairs")
    ax.set_title("A. Adult drug-outcome: raw vs contact-adjusted", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.scatter(np.exp(tab["raw_log_or"]), np.exp(tab["adj_log_or"]),
               s=14, color=C["blue"], alpha=0.7, edgecolor="none")
    lim = [0.4, max(2.2, np.exp(tab["raw_log_or"]).max() * 1.05)]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.axhline(1.0, color=C["green"], ls="--", lw=0.8)
    ax.set_xlabel("raw OR")
    ax.set_ylabel("adjusted OR")
    ax.set_title("B. Same pairs, before and after", fontsize=9)

    ax = axes[1, 0]
    ax.plot(np.sort(tab["p_raw"]), np.linspace(0, 1, len(tab)), color=C["red"],
            lw=1.3, label="raw p")
    ax.plot(np.sort(tab["p_cal"]), np.linspace(0, 1, len(tab)), color=C["green"],
            lw=1.3, label="calibrated p")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("p-value")
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"C. Calibration: {np.mean(tab['p_raw'] < .05):.0%} -> "
                 f"{np.mean(tab['p_cal'] < .05):.0%} significant", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    groups = ["all ages", "adult"]
    x = np.arange(len(groups))
    w = 0.26
    for off, key, col, lab in [(0, "raw", C["grey"], "raw"),
                               (1, "u", C["green"], "u-stratified"),
                               (2, "age", C["blue"], "age-stratified")]:
        vals = [mech[g][key] for g in groups]
        ax.bar(x + (off - 1) * w, vals, width=w, color=col, label=lab)
        for xi, v in zip(x + (off - 1) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5,
                    color=col)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("median odds ratio, condition pairs")
    ax.set_title("D. The all-ages bias is age", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig28_negative_control.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig28_negative_control.png and {RES}/negative_control.json")


if __name__ == "__main__":
    main()
