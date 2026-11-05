"""
topocdm.perdomain -- the per-domain temporal model of docs/MODEL_V2.md section 2.

Each domain is a separate process, because each has a different temporal type
and a different concept set:

    condition, drug        interval:  a start process + a duration distribution
    measurement, observation, procedure   point: an instant
    visit                  interval, degenerate (median duration 0)
    death                  absorbing: a hazard that removes the patient

Layer 2 is one patient heterogeneity term shared across all domains, u_i ~
Gamma(k). Layer 3 is a low-rank cross-domain coupling exp(z_i . beta_j), where
z_i is a latent patient state and beta_j are concept loadings. That coupling is
*recording dependence*, not clinical or causal dependence: a lab drawn because a
condition is suspected is a fact about how the database records care. It must
never be read causally, and every place it appears says so.

The fit reuses the tested pieces of the codebase where the object is identical
(marginal IPF for u_i and the concept intensities; negative-binomial marginal
for the Gamma shape; the damped-Newton factor fit for the coupling) and adds the
temporal pieces: per-domain calendar profiles, an estimated duration density, an
estimated death-timing distribution, and interval emission.

What transfers between databases is the *structure* -- the degree correction,
the shared heterogeneity, the coupling, the interval/point/absorbing split. The
domain rates, the duration density and the death timing are estimated from
whatever database the model is fitted to; they are nuisances, not claims, and
must be re-estimated (with the parametric diagnostics re-run) on new data. On a
rule-based simulator such as Synthea the duration shape is a property of the
simulator's rules, not evidence about a health system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from topocdm.factors import fit_factors

#: domains that carry a (start, end) interval
INTERVAL_DOMAINS = ("condition", "drug")
#: domains that are points (no end date in the CDM)
POINT_DOMAINS = ("measurement", "observation", "procedure")
#: visit is its own domain, with three concepts and a degenerate interval
VISIT_DOMAIN = "visit"
#: order used for reporting and generation
ALL_DOMAINS = ("condition", "drug", "measurement", "observation",
               "procedure", "visit")

EPS = 1e-12


# --------------------------------------------------------------------------
# loading and atom construction
# --------------------------------------------------------------------------

def load_extract(events_path="data/events_full.csv",
                 periods_path="data/periods.csv"):
    """Load the dated extract produced by scripts/extract_dated.sh."""
    ev = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
    pe = pd.read_csv(periods_path, parse_dates=["start_date", "end_date"])
    return ev, pe


def build_atoms(events: pd.DataFrame):
    """Index the concept universe.

    Mapped concepts (concept_id != 0) are one atom each. Unmapped records
    (concept_id == 0) are a per-domain atom, because the null category is a
    domain property (44% of procedures) and must be generated, not discarded.
    Returns (atoms_domain, atoms_concept) both length C, plus a dict mapping
    (domain, concept_id) -> column index.
    """
    rows = []
    for dom in ALL_DOMAINS:
        sub = events[events["domain"] == dom]
        for cid in sorted(sub["concept_id"].unique()):
            rows.append((dom, int(cid)))
    domain = np.array([r[0] for r in rows], dtype=object)
    concept = np.array([r[1] for r in rows], dtype=np.int64)
    idx = {(d, c): i for i, (d, c) in enumerate(rows)}
    return domain, concept, idx


def build_counts(events: pd.DataFrame, periods: pd.DataFrame,
                 domain: np.ndarray, concept: np.ndarray, idx: dict):
    """Patient x atom count matrix over mapped + unmapped atoms, plus person-time.

    Returns (X sparse (N, C), T (N,), person_ids (N,)).
    """
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    # death is an absorbing process, not an atom in the count matrix
    ev = events[events["domain"] != "death"]
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    key = list(zip(ev["domain"], ev["concept_id"]))
    cols = np.array([idx[(d, int(c))] for d, c in key], dtype=np.int64)
    vals = np.ones(len(ev), dtype=float)
    X = sparse.csr_matrix((vals, (rows, cols)),
                          shape=(len(persons), len(domain)))
    X.sum_duplicates()

    pe = periods.set_index("person_id").loc[persons]
    T = np.maximum((pe["end_date"] - pe["start_date"]).dt.days.to_numpy(
        dtype=float) / 365.25, 1.0 / 365.25)
    return X, T, persons


#: five clinical domains used for the whole-CDM spectrum (no visit, no unmapped)
CLINICAL_DOMAINS = ("condition", "drug", "measurement", "observation", "procedure")


def clinical_incidence(events: pd.DataFrame, periods: pd.DataFrame,
                       min_events: int = 1):
    """Mapped clinical incidence matrix (the 773-concept object).

    This is the matrix the earlier spectral numbers were computed on: the five
    clinical domains, mapped concepts only, no visit and no unmapped atom. It is
    the apples-to-apples comparison object. Returns (X, concept_ids, persons, T).
    """
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ev = events[events["domain"].isin(CLINICAL_DOMAINS)
                & (events["concept_id"] != 0)]
    concept_ids = np.sort(ev["concept_id"].unique())
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    cols = ev["concept_id"].map(ccol).to_numpy(dtype=np.int64)
    X = sparse.csr_matrix((np.ones(len(ev)), (rows, cols)),
                          shape=(len(persons), len(concept_ids)))
    X.sum_duplicates()
    if min_events > 1:
        keep = np.asarray(X.sum(axis=0)).ravel() >= min_events
        X = X[:, keep]
        concept_ids = concept_ids[keep]
    pe = periods.set_index("person_id").loc[persons]
    T = np.maximum((pe["end_date"] - pe["start_date"]).dt.days.to_numpy(
        dtype=float) / 365.25, 1.0 / 365.25)
    return X, concept_ids, persons, T


def tables_to_records(tables: dict) -> pd.DataFrame:
    """Flatten generated OMOP tables back into (person, domain, concept)."""
    spec = [("condition_occurrence", "condition_concept_id", "condition_start_date",
             "condition"),
            ("drug_exposure", "drug_concept_id", "drug_exposure_start_date", "drug"),
            ("procedure_occurrence", "procedure_concept_id", "procedure_date",
             "procedure"),
            ("measurement", "measurement_concept_id", "measurement_date",
             "measurement"),
            ("observation", "observation_concept_id", "observation_date",
             "observation"),
            ("visit_occurrence", "visit_concept_id", "visit_start_date", "visit")]
    parts = []
    for table, ccol, scol, dom in spec:
        df = tables.get(table)
        if df is None or len(df) == 0:
            continue
        parts.append(pd.DataFrame({"person_id": df["person_id"],
                                   "domain": dom,
                                   "concept_id": df[ccol],
                                   "start_date": df[scol]}))
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["person_id", "domain", "concept_id",
                                       "start_date"]))


def _year_exposure(periods: pd.DataFrame, years: np.ndarray):
    """Person-time observed in each calendar year. (n_persons, n_years) in years."""
    persons = periods["person_id"].to_numpy()
    s = periods["start_date"].to_numpy(dtype="datetime64[D]")
    e = periods["end_date"].to_numpy(dtype="datetime64[D]")
    out = np.zeros((len(persons), len(years)))
    for k, y in enumerate(years):
        ys = np.datetime64(f"{int(y)}-01-01", "D")
        ye = np.datetime64(f"{int(y) + 1}-01-01", "D")
        lo = np.maximum(s, ys)
        hi = np.minimum(e, ye)
        d = np.clip((hi - lo) / np.timedelta64(1, "D"), 0, None)
        out[:, k] = d / 365.25
    return out


# --------------------------------------------------------------------------
# duration and death
# --------------------------------------------------------------------------

def _gmmk(x: np.ndarray, k: int, n_iter: int = 300, seed: int = 0):
    """1D k-component Gaussian mixture on the log scale, by EM."""
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    n = len(x)
    # quantile initialisation, which is stable for the heavy-tailed components
    qs = np.quantile(x, (np.arange(k) + 0.5) / k)
    mu = np.sort(qs)
    s = np.full(k, max(x.std(ddof=1) / max(k, 1), 0.2))
    pi = np.full(k, 1.0 / k)
    for _ in range(n_iter):
        r = np.empty((n, k))
        for j in range(k):
            r[:, j] = pi[j] * norm.pdf(x, mu[j], s[j])
        r /= np.maximum(r.sum(axis=1, keepdims=True), 1e-300)
        nk = r.sum(axis=0)
        pi = nk / n
        for j in range(k):
            if nk[j] < 1e-6:
                continue
            mu[j] = float((r[:, j] * x).sum() / nk[j])
            s[j] = float(np.sqrt((r[:, j] * (x - mu[j]) ** 2).sum() / nk[j]))
        s = np.maximum(s, 1e-3)
    order = np.argsort(mu)
    return pi[order], mu[order], s[order]


def _gmm_bic(x: np.ndarray, kmax: int = 4, seed: int = 0):
    """Choose the mixture order by BIC (likelihood criterion, not a headline)."""
    from scipy.stats import norm
    best = None
    for k in range(2, kmax + 1):
        pi, mu, s = _gmmk(x, k, seed=seed)
        ll = 0.0
        for j in range(k):
            ll += np.sum(np.log(np.maximum(pi[j] * norm.pdf(x, mu[j], s[j]),
                                           1e-300)))
        n_params = 3 * k - 1          # weights sum to 1
        bic = -2 * ll + n_params * np.log(len(x))
        if best is None or bic < best[0]:
            best = (bic, k, pi, mu, s)
    return best[1], best[2], best[3], best[4]


def fit_durations(events: pd.DataFrame, domain: str):
    """Duration model for an interval domain.

    MODEL_V2 §2 assumes `log D ~ Normal`. It is a poor fit here: the observed
    durations are a point mass at 0 (acute/one-off coding), a short-course
    mode, a sharp ~1-year chronic mode, and a heavy tail. A single log-normal
    misplaces the median; a 2-component mixture helps but still puts the
    condition median at ~208 days against an observed 364.

    The working model is therefore a zero-inflated empirical quantile
    function: a point mass at 0 plus 101 quantiles of the positive durations.
    That reproduces the interval structure (and hence era merging) exactly,
    and it is still compact (~102 numbers per domain). The log-normal and
    mixture parameters are retained in the returned dict as the diagnostic
    that the parametric assumptions fail.

    These quantiles are a property of the fitted database, not a claim that
    transfers: on a rule-based simulator they reflect the simulator's rules.
    On a production CDM re-run the diagnostic and re-estimate; do not reuse
    the shape.
    """
    sub = events[events["domain"] == domain]
    d = (sub["end_date"] - sub["start_date"]).dt.days.to_numpy(dtype=float)
    d = np.clip(d, 0.0, None)
    pi0 = float((d == 0).mean())
    dpos = d[d > 0]
    qgrid = np.linspace(0.0, 1.0, 101)
    qvals = np.quantile(dpos, qgrid)
    # parametric diagnostics (not used for generation)
    logd = np.log(dpos)
    logn = (float(logd.mean()), float(logd.std(ddof=1)))
    k2, pi2, mu2, s2 = _gmm_bic(logd, kmax=2)
    return dict(pi0=pi0, qgrid=qgrid, qvals=qvals, k=int(k2),
                pi=pi2, mu=mu2, s=s2,
                lognormal_mu=logn[0], lognormal_s=logn[1],
                observed_median=float(np.median(d)),
                observed_zero=pi0)


def sample_durations(dur: dict, n: int, rng) -> np.ndarray:
    """Draw n durations (days) from the zero-inflated empirical quantile model."""
    out = np.zeros(n, dtype=float)
    pos = rng.random(n) >= dur["pi0"]
    npos = int(pos.sum())
    if npos:
        u = rng.random(npos)
        out[pos] = np.interp(u, dur["qgrid"], dur["qvals"])
    return np.maximum(np.round(out).astype(int), 0)


def fit_weibull(time_days: np.ndarray, event: np.ndarray):
    """Weibull MLE for the death hazard, on (time, censoring indicator).

    time_days is follow-up from observation start to death (event) or to
    observation end (censored). Returns (shape, scale) with scale in days.
    """
    t = np.maximum(time_days, 1e-9)

    def nll(p):
        shape = np.exp(p[0])
        scale = np.exp(p[1])
        z = t / scale
        # log density for events, log survival for censored
        ll = event * (np.log(shape) - np.log(scale) + (shape - 1) * np.log(z)) \
            - z ** shape
        return -float(np.sum(ll))

    r = optimize.minimize(nll, x0=[np.log(1.0), np.log(3.0 * 365.25)],
                          method="Nelder-Mead")
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

@dataclass
class PerDomainFit:
    domains: list = field(default_factory=lambda: list(ALL_DOMAINS))
    atoms_domain: np.ndarray = None       # (C,) domain name
    atoms_concept: np.ndarray = None      # (C,) concept_id (0 = unmapped)
    eta: np.ndarray = None                # (C,) concept intensity (events/PY)
    u: np.ndarray = None                  # (N,) contact volume, mean 1
    k: float = 1.0                        # Gamma shape (heterogeneity)
    propensity_tmean: float = 1.0         # person-time-weighted mean of fitted u
    years: np.ndarray = None              # calendar years
    lam_t: dict = field(default_factory=dict)   # domain -> (n_years,)
    duration: dict = field(default_factory=dict)  # domain -> (mu, s, median)
    death: dict = field(default_factory=dict)     # shape, scale, frac
    z: np.ndarray = None                  # (N, K) recording dependence
    beta: np.ndarray = None               # (C, K) concept loadings
    z_mean: np.ndarray = None             # (K,)
    z_cov: np.ndarray = None              # (K, K)
    person_ids: np.ndarray = None
    n_events: dict = field(default_factory=dict)  # domain -> observed count
    calibration: np.ndarray = None        # (C,) level calibration for generation
    meta: dict = field(default_factory=dict)

    @property
    def n_concepts(self) -> int:
        return len(self.atoms_domain)

    @property
    def n_parameters(self) -> int:
        """Count of released parameters (no per-patient terms).

        eta (C) + per-domain calendar (len(years) each) + duration (6 per
        interval domain: pi0 + pi + 2*mu + 2*s) + death fraction (1) +
        coupling beta (C*K) + latent distribution (K + K(K+1)/2) +
        heterogeneity shape (1). The empirical death-timing array is stored
        separately (not a parameter count) and is reported in meta.
        """
        n = int(len(self.eta)) + 1
        for d in self.lam_t:
            n += int(len(self.lam_t[d]))
        n += 102 * len(INTERVAL_DOMAINS)   # pi0 + 101 duration quantiles
        K = 0 if self.beta is None else self.beta.shape[1]
        n += int(len(self.eta) * K) + int(K + K * (K + 1) / 2)
        return n


def _fit_marginals_sparse(X, T, n_iter: int = 200, tol: float = 1e-10):
    """G=1 IPF on a sparse count matrix, without densifying.

    The M-step only needs the row and column sums, so a large N never has to
    materialise an N x C dense array. Returns (u, eta).
    """
    rowsum = np.asarray(X.sum(axis=1)).ravel().astype(float)
    colsum = np.asarray(X.sum(axis=0)).ravel().astype(float)
    u = np.maximum(rowsum / max(T.sum(), EPS), EPS)
    u /= max(u.mean(), EPS)
    lam = np.maximum(colsum / max(float(np.sum(T * u)), EPS), EPS)
    lam_prev = lam.copy()
    for _ in range(n_iter):
        u = np.maximum(rowsum / np.maximum(T * lam.sum(), EPS), EPS)
        lam = np.maximum(colsum / max(float(np.sum(T * u)), EPS), EPS)
        m = u.mean()
        if m > 0:
            u, lam = u / m, lam * m
        # scale-free convergence on the normalised intensities
        denom = max(float(np.sum(lam)), EPS)
        change = float(np.max(np.abs(lam / denom - lam_prev / denom)))
        lam_prev = lam.copy()
        if change < tol:
            break
    return u, lam


def _fit_gamma_shape(x: np.ndarray, mu: np.ndarray) -> float:
    """Gamma shape from the marginal negative-binomial likelihood of totals."""
    from scipy import optimize
    from scipy.special import gammaln
    x = np.asarray(x, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), EPS)

    def nll(theta):
        k = np.exp(theta[0])
        ll = (gammaln(x + k) - gammaln(k) - gammaln(x + 1)
              + k * (np.log(k) - np.log(k + mu))
              + x * (np.log(mu) - np.log(k + mu)))
        return -float(np.sum(ll))

    r = optimize.minimize_scalar(lambda t: nll(np.array([t])),
                                 bounds=(-6.0, 6.0), method="bounded")
    return float(np.exp(r.x))


def fit_perdomain(events: pd.DataFrame, periods: pd.DataFrame, K: int = 20,
                  n_iter_factors: int = 12, seed: int = 0,
                  max_patients: int | None = None,
                  sample_frac: float | None = None,
                  max_factor_patients: int = 5000) -> PerDomainFit:
    """Fit the per-domain temporal model.

    K is the rank of the recording-dependence coupling. It is set a priori
    (MODEL_V2 §2.4 uses K = 20); it is NOT selected on the spectral distance,
    which is one of the diagnostics reported later.

    Scale. `max_patients` / `sample_frac` fit on a patient sample; the marginal
    fit is sparse (no N x C dense array), and the coupling is fitted on at most
    `max_factor_patients` patients. The released parameters are all O(C) or
    O(K^2), so the same code runs on a subsample of a very large database.
    """
    rng = np.random.default_rng(seed)

    # optional patient subsample, shared by every layer
    if max_patients is not None or sample_frac is not None:
        persons_all = np.sort(periods["person_id"].unique())
        n_take = int(len(persons_all) * (sample_frac or 1.0))
        if max_patients is not None:
            n_take = min(n_take, int(max_patients))
        n_take = max(min(n_take, len(persons_all)), 2)
        keep = np.sort(rng.choice(persons_all, size=n_take, replace=False))
        events = events[events["person_id"].isin(keep)].copy()
        periods = periods[periods["person_id"].isin(keep)].copy()

    atoms_domain, atoms_concept, idx = build_atoms(events)
    X, T, persons = build_counts(events, periods, atoms_domain,
                                 atoms_concept, idx)

    # ---- Layer 1 marginals + Layer 2 heterogeneity (sparse IPF) ----
    u, eta = _fit_marginals_sparse(X, T)
    k = _fit_gamma_shape(np.asarray(X.sum(axis=1)).ravel().astype(float),
                         T * eta.sum())
    propensity_tmean = float(np.sum(T * u) / max(T.sum(), EPS))

    # ---- Layer 1 temporal profiles per domain ----
    years = np.arange(int(events["start_date"].dt.year.min()),
                      int(events["start_date"].dt.year.max()) + 1)
    expo = _year_exposure(periods, years)              # (N, Y)
    pt_y = expo.sum(axis=0)
    lam_t = {}
    n_events = {}
    for d in ALL_DOMAINS:
        m = events["domain"] == d
        n_events[d] = int(m.sum())
        dom = events[m]
        yy = dom["start_date"].dt.year.to_numpy()
        obs = np.zeros(len(years))
        np.add.at(obs, yy - years[0], 1.0)
        rate = obs / np.maximum(pt_y, EPS)
        # person-time-weighted mean 1, so generation averages to the observed
        # calendar rate under the resampled follow-up distribution
        wmean = float(np.sum(pt_y * rate) / max(np.sum(pt_y), EPS))
        lam_t[d] = rate / max(wmean, EPS)

    # ---- Layer 1 durations ----
    duration = {}
    for d in INTERVAL_DOMAINS:
        duration[d] = fit_durations(events, d)

    # ---- Layer 1 death (absorbing) ----
    # A single Weibull hazard is a poor fit: Synthea's deaths are driven by
    # disease modules, not a smooth hazard. For generation we store the observed
    # mortality fraction and the empirical distribution of time-to-death as a
    # fraction of the observation window, which reproduces both the level
    # (10.4%) and the within-window timing without assuming a parametric shape.
    ob_start = periods.set_index("person_id").loc[persons, "start_date"].to_numpy(
        dtype="datetime64[D]")
    ob_end = periods.set_index("person_id").loc[persons, "end_date"].to_numpy(
        dtype="datetime64[D]")
    death = events[events["domain"] == "death"]
    died = death.set_index("person_id")["start_date"].to_dict()
    follow = (ob_end - ob_start) / np.timedelta64(1, "D")
    death_frac_times = []
    for i, p in enumerate(persons):
        if int(p) in died:
            dd = np.datetime64(died[int(p)], "D")
            t = (dd - ob_start[i]) / np.timedelta64(1, "D")
            death_frac_times.append(float(np.clip(t / max(follow[i], 1.0),
                                                   0.0, 1.0)))
    death_fit = dict(frac=float(len(death_frac_times) / len(persons)),
                     frac_times=np.asarray(death_frac_times, dtype=float))

    # ---- Layer 3 recording dependence (low-rank coupling) ----
    z = np.zeros((len(persons), 1))
    beta = np.zeros((len(atoms_domain), 1))
    z_mean = np.zeros(1)
    z_cov = np.eye(1) * 1e-6
    calibration = np.ones(len(atoms_domain))
    if K > 0:
        # fit the coupling on at most max_factor_patients patients; the loadings
        # beta are per concept, and the latent distribution is what generation
        # samples, so a patient subsample is enough.
        n_fit = min(len(persons), int(max_factor_patients))
        sub = np.sort(rng.choice(len(persons), size=n_fit, replace=False))
        mu0 = (T[sub, None] * u[sub, None] * eta[None, :])
        Xd = X[sub].toarray().astype(float)
        ff = fit_factors(mu0, Xd, K=K, n_iter=n_iter_factors, seed=seed)
        z = ff.a
        beta = ff.b
        z_mean = z.mean(axis=0)
        z_cov = np.cov(z, rowvar=False) + 1e-6 * np.eye(K)
        # level calibration: generation samples z from N(z_mean, z_cov), so the
        # expected multiplier per concept is exp(mu_z.b + 0.5 b' Cov b). Dividing
        # eta by it keeps the marginal event level on target while preserving the
        # per-patient recording-dependence structure.
        quad = np.einsum("ck,kl,cl->c", beta, z_cov, beta)
        m_j = np.exp(beta @ z_mean + 0.5 * quad)
        calibration = 1.0 / np.maximum(m_j, EPS)

    return PerDomainFit(
        atoms_domain=atoms_domain, atoms_concept=atoms_concept,
        eta=eta, u=u, k=k, propensity_tmean=propensity_tmean,
        years=years, lam_t=lam_t, duration=duration,
        death=death_fit, z=z, beta=beta, z_mean=z_mean, z_cov=z_cov,
        person_ids=persons, n_events=n_events,
        calibration=calibration,
        meta=dict(n_patients=int(len(persons)),
                  n_concepts=int(len(atoms_domain)), K=int(K),
                  n_events_total=int(X.sum()),
                  n_deaths=int(len(death_frac_times)),
                  factor_patients=int(min(len(persons),
                                          max_factor_patients))),
    )


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def sample_periods(periods: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Resample (start, duration) pairs, keeping follow-up structure only."""
    s = periods["start_date"].to_numpy(dtype="datetime64[D]")
    d = (periods["end_date"] - periods["start_date"]).dt.days.to_numpy()
    pick = rng.integers(0, len(periods), size=n)
    return pd.DataFrame({
        "person_id": np.arange(1, n + 1),
        "start_date": pd.to_datetime(s[pick]),
        "end_date": pd.to_datetime(s[pick] + d[pick].astype("timedelta64[D]")),
    })


def _generate_events(fit: PerDomainFit, periods: pd.DataFrame, rng):
    """Draw dated events (start dates) for every domain, plus death times.

    Returns (records DataFrame, death DataFrame). Cell counts are drawn as
    independent Poissons (equivalent to Poisson totals + multinomial allocation,
    but vectorisable), per domain-year over the patients active that year.
    """
    n = len(periods)
    years = fit.years
    expo = _year_exposure(periods, years)                 # (n, Y)

    # fresh patient heterogeneity and latent coupling state
    k = max(fit.k, 1e-6)
    u = rng.gamma(k, 1.0 / k, size=n)
    u /= u.mean()
    # match the person-time-weighted mean of the fitted u (trap: the unweighted
    # mean is 1 by constraint, but the fitted person-time-weighted mean is below
    # 1 and ignoring it over-predicts the total event rate)
    pt = expo.sum(axis=1)
    if fit.propensity_tmean and fit.propensity_tmean > 0 and pt.sum() > 0:
        u *= fit.propensity_tmean * float(pt.sum()) / float(np.sum(pt * u))
    K = 0 if fit.beta is None else fit.beta.shape[1]
    if K > 1:
        z = rng.multivariate_normal(fit.z_mean, fit.z_cov, size=n)
        coupling = np.exp(np.clip(z @ fit.beta.T, -8.0, 8.0))   # (n, C)
    else:
        coupling = np.ones((n, fit.n_concepts))

    atom_idx = {d: np.where(fit.atoms_domain == d)[0] for d in ALL_DOMAINS}
    pid_parts, dom_parts, cid_parts, date_parts = [], [], [], []
    for d in ALL_DOMAINS:
        cols = atom_idx[d]
        eta_d = fit.eta[cols] * fit.calibration[cols]
        lam_d = fit.lam_t[d]
        for kk, y in enumerate(years):
            act = expo[:, kk] > 0
            if not act.any():
                continue
            mu = (expo[act, kk][:, None] * u[act][:, None] * eta_d[None, :]
                  * lam_d[kk] * coupling[np.ix_(act, cols)])
            X = rng.poisson(mu)
            ii, jj = np.nonzero(X)
            if ii.size == 0:
                continue
            cnt = X[ii, jj].astype(np.int64)
            n_ev = int(cnt.sum())
            pid_full = np.repeat(np.arange(1, n + 1)[act][ii], cnt)
            cid_full = np.repeat(fit.atoms_concept[cols[jj]], cnt)
            ys = np.datetime64(f"{int(y)}-01-01", "D")
            day = rng.integers(0, 365, size=n_ev)
            date_full = ys + day.astype("timedelta64[D]")
            pid_parts.append(pid_full)
            dom_parts.append(np.repeat(d, n_ev))
            cid_parts.append(cid_full)
            date_parts.append(date_full)

    if pid_parts:
        events = pd.DataFrame({
            "person_id": np.concatenate(pid_parts),
            "domain": np.concatenate(dom_parts),
            "concept_id": np.concatenate(cid_parts),
            "start_date": pd.to_datetime(np.concatenate(date_parts)),
            "end_date": pd.NaT,
        })
    else:
        events = pd.DataFrame(columns=[
            "person_id", "domain", "concept_id", "start_date", "end_date"])

    # durations for interval domains, clipped to the observation period: an
    # interval cannot outlive the patient's follow-up (and an unclipped
    # log-normal tail can overflow the datetime range)
    ob_end_map = periods.set_index("person_id")["end_date"]
    for d in INTERVAL_DOMAINS:
        m = events["domain"] == d
        if not m.any():
            continue
        nd = int(m.sum())
        dur = sample_durations(fit.duration[d], nd, rng)
        starts = events.loc[m, "start_date"].to_numpy(dtype="datetime64[D]")
        obe = pd.to_datetime(events.loc[m, "person_id"].map(ob_end_map)) \
            .to_numpy(dtype="datetime64[D]")
        max_dur = np.maximum((obe - starts) / np.timedelta64(1, "D"), 0)
        dur_days = np.minimum(dur.astype(float), max_dur).astype(np.int64)
        ends = starts + dur_days.astype("timedelta64[D]")
        events.loc[m, "end_date"] = ends.astype("datetime64[ns]")

    # death: observed mortality fraction and within-window timing
    ob_start = periods["start_date"].to_numpy(dtype="datetime64[D]")
    ob_end = periods["end_date"].to_numpy(dtype="datetime64[D]")
    follow = (ob_end - ob_start) / np.timedelta64(1, "D")
    died = rng.random(n) < fit.death["frac"]
    if died.any() and len(fit.death["frac_times"]):
        frac = rng.choice(fit.death["frac_times"], size=int(died.sum()))
        t_days = (frac * follow[died]).astype("timedelta64[D]")
    else:
        t_days = np.array([], dtype="timedelta64[D]")
    death_date = ob_start[died] + t_days
    death = pd.DataFrame({
        "person_id": np.arange(1, n + 1)[died],
        "death_date": pd.to_datetime(death_date),
    })

    return events, death


def generate_cohort(fit: PerDomainFit, n: int, periods: pd.DataFrame | None = None,
                    seed: int = 0):
    """Generate a synthetic dated cohort and the era tables.

    Returns a dict of DataFrames keyed by table name:
    condition_occurrence, drug_exposure, procedure_occurrence, measurement,
    observation, visit_occurrence, death, condition_era, drug_era,
    observation_period.
    """
    rng = np.random.default_rng(seed)
    if periods is None:
        raise ValueError("a periods frame (resampled or supplied) is required")
    events, death = _generate_events(fit, periods, rng)
    # keep only events before death (death truncates all other processes)
    if len(death):
        dmap = death.set_index("person_id")["death_date"]
        dd = pd.to_datetime(events["person_id"].map(dmap)).to_numpy(
            dtype="datetime64[ns]")
        start_ns = events["start_date"].to_numpy(dtype="datetime64[ns]")
        keep = np.isnat(dd) | (start_ns <= dd)
        events = events[keep].copy()

    return tables_from_events(events, periods, death)


def tables_from_events(events: pd.DataFrame, periods: pd.DataFrame,
                       death: pd.DataFrame) -> dict:
    """Build the OMOP domain, era and death tables from a dated event frame.

    Shared by the per-domain and visit-anchored generators: both produce the
    same (person_id, domain, concept_id, start_date, end_date) frame, and the
    table/era construction is identical.
    """
    tables = {}
    field = {
        "condition": ("condition_occurrence", "condition_concept_id",
                      "condition_start_date", "condition_end_date"),
        "drug": ("drug_exposure", "drug_concept_id", "drug_exposure_start_date",
                 "drug_exposure_end_date"),
        "procedure": ("procedure_occurrence", "procedure_concept_id",
                      "procedure_date", None),
        "measurement": ("measurement", "measurement_concept_id",
                        "measurement_date", None),
        "observation": ("observation", "observation_concept_id",
                        "observation_date", None),
        "visit": ("visit_occurrence", "visit_concept_id", "visit_start_date",
                  "visit_end_date"),
    }
    for d, (table, ccol, scol, ecol) in field.items():
        sub = events[events["domain"] == d].copy()
        if sub.empty:
            tables[table] = pd.DataFrame()
            continue
        sub = sub.sort_values(["person_id", "start_date"]).reset_index(drop=True)
        out = pd.DataFrame({"person_id": sub["person_id"],
                            ccol: sub["concept_id"], scol: sub["start_date"]})
        if ecol:
            out[ecol] = sub["end_date"]
        tables[table] = out

    # era tables: merge overlapping/adjacent intervals of the same concept
    for d, etable, ecol, oscol, oecol in [
            ("condition", "condition_era", "condition_concept_id",
             "condition_era_start_date", "condition_era_end_date"),
            ("drug", "drug_era", "drug_concept_id",
             "drug_era_start_date", "drug_era_end_date")]:
        src = field[d][0]
        scol = field[d][2]
        ecol_ = field[d][3]
        era = build_eras(tables[src], ecol, scol, ecol_,
                         out_start=oscol, out_end=oecol)
        # OMOP era-table shape: an id column and the source count column
        era.insert(0, f"{d}_era_id", np.arange(1, len(era) + 1, dtype=np.int64))
        era = era.rename(columns={"occurrence_count": f"{d}_occurrence_count"})
        tables[etable] = era

    tables["death"] = death
    tables["observation_period"] = periods.rename(columns={
        "start_date": "observation_period_start_date",
        "end_date": "observation_period_end_date",
    })
    return tables


def build_eras(records: pd.DataFrame, concept_col: str, start_col: str,
               end_col: str, out_start: str = None, out_end: str = None,
               gap_days: int = 30):
    """Collapse overlapping/adjacent intervals of the same concept into eras.

    OMOP merges condition_occurrence rows with the same condition_concept_id
    whose intervals overlap or fall within a persistence window (default 30
    days). `records` may be empty (returns an empty era frame with the right
    columns). `out_start`/`out_end` rename the interval columns for the era
    table (condition_era_start_date etc.).
    """
    out_start = out_start or start_col
    out_end = out_end or end_col
    out_cols = ["person_id", concept_col, out_start, out_end, "occurrence_count"]
    if records is None or len(records) == 0:
        return pd.DataFrame(columns=out_cols)
    df = records.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    df = df.sort_values(["person_id", concept_col, start_col])
    rows = []
    for (pid, cid), g in df.groupby(["person_id", concept_col], sort=False):
        g = g.reset_index(drop=True)
        start = g[start_col].to_numpy()
        end = g[end_col].to_numpy()
        era_start = start[0]
        era_end = end[0]
        cnt = 1
        for i in range(1, len(g)):
            gap = (start[i] - era_end) / np.timedelta64(1, "D")
            if gap <= gap_days:
                era_end = max(era_end, end[i])
                cnt += 1
            else:
                rows.append((pid, cid, era_start, era_end, cnt))
                era_start, era_end, cnt = start[i], end[i], 1
        rows.append((pid, cid, era_start, era_end, cnt))
    out = pd.DataFrame(rows, columns=out_cols)
    return out


# --------------------------------------------------------------------------
# spectral comparison (reported alongside the summary statistics)
# --------------------------------------------------------------------------

def spectrum(M) -> np.ndarray:
    """Singular values of the degree-normalised incidence matrix.

    Same object as experiments/14: N = D_r^-1/2 M D_c^-1/2, singular values are
    the square roots of the eigenvalues of N^T N.
    """
    M = M if sparse.issparse(M) else sparse.csr_matrix(np.asarray(M, float))
    r = np.asarray(M.sum(1)).ravel()
    c = np.asarray(M.sum(0)).ravel()
    N = (sparse.diags(1 / np.sqrt(np.maximum(r, 1e-9))) @ M
         @ sparse.diags(1 / np.sqrt(np.maximum(c, 1e-9))))
    n, m = N.shape
    # use the smaller Gram matrix: the non-zero singular values are the same,
    # and the zero padding restores the full spectrum. This matters when C is
    # large (MIMIC has ~2,700 clinical concepts for 100 patients).
    if n <= m:
        G = (N @ N.T).toarray()
    else:
        G = (N.T @ N).toarray()
    e = np.sort(np.linalg.eigvalsh(G))[::-1]
    s = np.sqrt(np.maximum(e, 0))
    return np.concatenate([s, np.zeros(max(n, m) - len(s))])


def spectral_ks(s1: np.ndarray, s2: np.ndarray, grid: np.ndarray | None = None):
    """Cumulative spectral distance (KS) between two singular-value spectra."""
    grid = np.linspace(0, max(s1.max(), s2.max()), 200) if grid is None else grid
    c1 = np.searchsorted(np.sort(s1), grid, side="right") / len(s1)
    c2 = np.searchsorted(np.sort(s2), grid, side="right") / len(s2)
    return float(np.abs(c1 - c2).max())


def effective_rank(s: np.ndarray, threshold: float = 0.1) -> int:
    """Number of singular values above `threshold` (of the leading value)."""
    return int((s > threshold).sum())
