"""
topocdm.temporal -- calendar time for a synthetic CDM.

A count model with no dates is not a synthetic CDM. It cannot answer "what was
the incidence in 2019", cannot be entered into CohortIncidence, and cannot be
compared to another database on the axis that matters.

The temporal layer here is deliberately thin and is *not* a new fit. It rides
on the fitted cross-sectional model:

    mu_ij(y) = T_i(y) * u_i * lambda_j * B[g_i, h_j] * tau_y

where T_i(y) is person-time observed in calendar year y and tau_y is a per-year
intensity multiplier. For fixed (u, lambda, B) the MLE of tau_y is

    tau_y = observed_events(y) / expected_events(y)

which is exactly indirect standardisation, and is why this can be computed
after the fact without refitting. The same argument gives the visit rate
nu_y = visits(y) / person-time(y).

tau_y is fitted rather than modelled because the real data demands it: this CDM
has a 7x step at 2016 and a 2021 visit spike, both artefacts of how the data
were produced. A smooth secular trend would miss them, and a generator that
missed them would fail exactly the calendar-year comparison it is meant for.

What this layer does *not* capture: within-year seasonality, and any
patient-level trajectory (a person's condition mix changing over time). Both
are real; both are the next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TemporalProfile:
    years: np.ndarray                 # calendar years covered (int)
    tau: np.ndarray                   # event intensity multiplier per year
    visit_rate: np.ndarray            # visits per person-year per year
    events_per_visit: float = 2.0
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(years=[int(y) for y in self.years],
                    tau_mean=float(self.tau.mean()),
                    tau_max=float(self.tau.max()),
                    visit_rate_mean=float(self.visit_rate.mean()),
                    events_per_visit=float(self.events_per_visit))


def year_exposure(periods: pd.DataFrame, years: np.ndarray) -> np.ndarray:
    """Person-time observed in each calendar year.

    periods: person_id, start_date, end_date (as datetime64)
    Returns (n_persons, n_years) in years.
    """
    persons = periods["person_id"].to_numpy()
    rows = {p: i for i, p in enumerate(persons)}
    s = periods["start_date"].to_numpy(dtype="datetime64[D]")
    e = periods["end_date"].to_numpy(dtype="datetime64[D]")
    out = np.zeros((len(persons), len(years)))
    for k, y in enumerate(years):
        ys = np.datetime64(f"{y}-01-01", "D")
        ye = np.datetime64(f"{y+1}-01-01", "D")
        lo = np.maximum(s, ys)
        hi = np.minimum(e, ye)
        days = np.clip((hi - lo) / np.timedelta64(1, "D"), 0, None)
        out[:, k] = days / 365.25
    return out, rows


def fit_temporal(events: pd.DataFrame, visits: pd.DataFrame,
                 periods: pd.DataFrame, params, years: np.ndarray,
                 concept_ids: np.ndarray) -> TemporalProfile:
    """Estimate tau_y and the visit rate by indirect standardisation.

    events: person_id, concept_id (must be columns of the fitted model)
    visits: person_id, visit_date
    periods: person_id, start_date, end_date
    """
    expo, prow = year_exposure(periods, years)
    persons = periods["person_id"].to_numpy()

    # expected events per person per year under the fitted cross-sectional model
    H = int(params.concept_group.max()) + 1
    lam_h = np.bincount(params.concept_group, weights=params.lam, minlength=H)
    if params.B is None:
        Lam = np.full(len(persons), lam_h.sum())
    else:
        # use each person's fitted group; an average over groups would distort
        # tau_y and push the synthetic calendar profile towards the mean
        grp = np.asarray(params.patient_group)[:len(persons)]
        Lam = (params.B @ lam_h)[grp]
    u = np.asarray(params.u, dtype=float)
    rate = u * Lam                                   # events per person-year
    expected = expo * rate[:, None]                  # (n, Y)

    # observed events per person per year
    ey = events["event_date"].dt.year.to_numpy()
    pid = events["person_id"].map(prow).to_numpy()
    idx = np.searchsorted(years, ey)
    ok = np.isfinite(pid.astype(float)) & (idx >= 0) & (idx < len(years))
    obs = np.zeros_like(expected)
    np.add.at(obs, (pid[ok].astype(int), idx[ok]), 1.0)
    inside = np.isin(ey[ok], years)

    exp_y = expected.sum(axis=0)
    obs_y = obs.sum(axis=0)
    tau = np.where(exp_y > 0, obs_y / np.maximum(exp_y, 1e-9), 1.0)
    tau = tau / np.mean(tau[tau > 0])

    # visit rate per person-year, by year
    vy = visits["visit_date"].dt.year.to_numpy()
    vpid = visits["person_id"].map(prow).to_numpy()
    vok = np.isfinite(vpid.astype(float))
    vobs = np.zeros(len(years))
    idx = np.searchsorted(years, vy[vok])
    good = (idx >= 0) & (idx < len(years)) & (years[np.clip(idx, 0, len(years)-1)] == vy[vok])
    np.add.at(vobs, idx[good], 1.0)
    pt_y = expo.sum(axis=0)
    visit_rate = np.where(pt_y > 0, vobs / np.maximum(pt_y, 1e-9), 0.0)

    epv = float(len(events) / max(len(visits), 1))
    return TemporalProfile(years=np.asarray(years), tau=tau,
                           visit_rate=visit_rate, events_per_visit=epv,
                           meta=dict(n_events_in_window=int(inside.sum()),
                                     observed_years=obs_y.tolist()))


def sample_periods(periods: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Draw observation periods by resampling their (start, duration) pairs.

    This keeps the observed distribution of follow-up without carrying any
    patient identity: only the joint distribution of start and length survives.
    """
    d = (periods["end_date"] - periods["start_date"]).dt.days.to_numpy()
    s = periods["start_date"].to_numpy(dtype="datetime64[D]")
    pick = rng.integers(0, len(periods), size=n)
    return pd.DataFrame({
        "person_id": np.arange(1, n + 1),
        "start_date": pd.to_datetime(s[pick]),
        "end_date": pd.to_datetime(s[pick] + d[pick].astype("timedelta64[D]")),
    })


def generate_dated(release, n_patients: int, periods: pd.DataFrame,
                   profile: TemporalProfile, rng=None):
    """Generate visits and dated events for a synthetic cohort.

    Returns (visits, events) DataFrames with OMOP-ready columns:
        visits: person_id, visit_date
        events: person_id, concept_id, event_date
    Events are attached to visits, because in a CDM nothing is recorded outside
    an encounter; a year with events but no visit gets a visit created for it.
    """
    rng = rng or np.random.default_rng(0)
    years = profile.years
    expo, _ = year_exposure(periods, years)
    n = len(periods)

    # propensity and patient group
    k = max(release.propensity_shape, 1e-6)
    u = rng.gamma(k, 1.0 / k, size=n)
    u /= u.mean()
    H = int(release.concept_group.max()) + 1
    lam_h = np.bincount(release.concept_group, weights=release.lam, minlength=H)
    if release.B is None:
        Lam = np.full(n, lam_h.sum()); g = np.zeros(n, dtype=int)
    else:
        g = rng.choice(release.B.shape[0], size=n, p=release.pi)
        Lam = (release.B @ lam_h)[g]

    # visits per year
    mu_visits = expo * profile.visit_rate[None, :]
    n_visits = rng.poisson(mu_visits)

    visit_rows = []
    visit_index = {}          # (person, year) -> list of visit ids
    vid = 0
    for i in range(n):
        for kk, y in enumerate(years):
            v = int(n_visits[i, kk])
            if v == 0:
                continue
            ys = np.datetime64(f"{y}-01-01", "D")
            day = rng.integers(0, 365, size=v)
            dates = pd.to_datetime(ys + day.astype("timedelta64[D]"))
            ids = np.arange(vid, vid + v)
            vid += v
            visit_rows.append(pd.DataFrame({"visit_occurrence_id": ids,
                                            "person_id": i + 1,
                                            "visit_date": dates}))
            visit_index[(i, kk)] = ids
    visits = (pd.concat(visit_rows, ignore_index=True)
              if visit_rows else pd.DataFrame(columns=["visit_occurrence_id", "person_id", "visit_date"]))

    # events per person per year, allocated over concepts and dated at a visit
    mu_events = expo * (u * Lam)[:, None] * profile.tau[None, :]
    counts = rng.poisson(mu_events)

    vdates = {}
    if len(visits):
        vp = visits["person_id"].to_numpy()
        vy = pd.to_datetime(visits["visit_date"]).dt.year.to_numpy()
        vl = visits["visit_date"].to_numpy()
        for ptr, yr, dt in zip(vp, vy, vl):
            vdates.setdefault((int(ptr) - 1, int(yr)), []).append(dt)
    vdates = {k: np.array(v) for k, v in vdates.items()}

    rows = []
    for i in range(n):
        for kk, y in enumerate(years):
            c = int(counts[i, kk])
            if c == 0:
                continue
            q = release.lam if release.B is None else release.lam * release.B[g[i], release.concept_group]
            draw = rng.multinomial(c, q / q.sum())
            nz = np.nonzero(draw)[0]
            if nz.size == 0:
                continue
            pool = vdates.get((i, int(y)))
            if pool is None:
                ys = np.datetime64(f"{y}-01-01", "D")
                pool = ys + rng.integers(0, 365, size=1).astype("timedelta64[D]")
            for j in nz:
                dates = rng.choice(pool, size=int(draw[j]))
                for dt in dates:
                    rows.append((i + 1, int(release.concept_ids[j]), dt))
    events = pd.DataFrame(rows, columns=["person_id", "concept_id", "event_date"])
    return visits, events
