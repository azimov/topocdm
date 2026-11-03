# Calendar time: what the temporal layer does, and what it found

Dates are a hard requirement for a synthetic CDM. This note records the design,
and the diagnostic result that came out of fitting it — which is more
interesting than the layer itself.

## 1. Design

The temporal layer rides on the fitted cross-sectional model rather than being
refitted:

    mu_ij(y) = T_i(y) * u_i * lambda_j * B[g_i, h_j] * tau_y

`T_i(y)` is person-time observed in calendar year y; `tau_y` is a per-year
intensity multiplier. For fixed (u, lambda, B) the MLE of `tau_y` is

    tau_y = observed_events(y) / expected_events(y)

which is indirect standardisation, so it is closed form and needs no
optimisation. The visit rate is estimated the same way:
`nu_y = visits(y) / person_time(y)`.

Generation draws observation periods by resampling (start, duration) pairs,
then samples visits per person-year and attaches events to visits. Events are
never dated outside an encounter.

## 2. What the fit found: the burst is left truncation, not calendar time

This CDM's events per year rise **7× between 2015 and 2016** (48k → 299k) and
then grow smoothly. Fitting `tau_y` on 2016–2025 gives `tau_2016 = 3.06` and
`tau_y ≈ 0.7–0.9` thereafter, which reproduces the observed yearly totals — but
the synthetic data still puts **28% of all events in 2016** against the real
7.7%.

The diagnostic test settles it: move the window to 2017–2025 and the burst
moves with it (`tau_2017 = 3.2`, 35% of synthetic events in 2017). **The burst
follows the start of the analysis window, not the calendar.**

That is the signature of left truncation. Patients enter this database at a
fixed date, and their prior history is compressed into their first observed
year. A per-year intensity multiplier cannot represent that: it is a *cohort
entry* effect, not a *time* effect.

The consequence for generation is concrete: in the real data the first-year
burst belongs to the patients who enter then, and those patients have high
propensities. Generation draws propensity independently of entry date, so the
burst is smeared across everyone. Reproducing it requires modelling the joint
distribution of (observation start, propensity) — i.e. the entry process — not
a richer `tau`.

## 3. What this means for CohortIncidence comparisons

A calendar-year incidence comparison is only meaningful on a cohort definition
that controls for entry. The implementation in `topocdm/cohort_incidence.py`
does the standard thing: require a minimum prior observation window and drop
prevalent cases. Without that, the 2016 row shows an incidence of 3,809 per
1000 person-years purely because nearly every patient's first recorded event
falls in their first observed year.

With a 365-day prior-observation requirement, the same concept gives a
plausible declining series (418 → ~55 per 1000 PY from 2016 to 2025), which is
what a comparison between two databases should be built on.

## 4. Intervals: simulation versus the fit

Repeated simulation gives a 95% interval, but it is the expensive route to a
quantity the fit already contains.

For a cohort with person-time `PT` and model-expected events `mu`, the count is
Poisson (or negative binomial once patient propensity is integrated out), so

    Var(X) = mu                     (Poisson)
    Var(X) = mu + mu^2 / k          (NB, k = fitted propensity shape)

and the predictive interval for the rate is `mu/PT ± 1.96 sqrt(Var)/PT`. No
simulation needed.

The distinction that matters for the cross-database question:

- the **Poisson interval** describes the sampling precision of *one* database;
- the **predictive interval** describes how much a *new* database with the same
  observation process would vary.

Only the second can answer "are these two data sources different, or is this
residual bias?". And it is a property of the fitted model, not of a Monte Carlo
run. `topocdm/cohort_incidence.py` provides both (`poisson_ci`,
`predictive_ci`).

## 5. Next

- Model entry explicitly: a distribution over (observation start, propensity),
  so the first-year burst is reproduced rather than smeared.
- Replace the per-year `tau` with a smooth trend plus an entry term, once entry
  is modelled; otherwise `tau` will keep absorbing cohort artefacts.
- Seasonal structure within a year is not modelled at all.

### 4a. The interval shrinks to the Poisson interval as the cohort grows

The over-dispersion term must be summed over patients, not applied to the total:

    Var(sum) = mu + mu^2 / (n k)

For a cohort of 1,000 patients with 120 expected events and a fitted shape
k = 2.2, the model's predictive interval is 98–142 per 1,000 PY against a
Poisson interval of 99.5–143.5. They are almost identical, because independent
over-dispersion averages out across a thousand people.

Applying the single-patient formula to the total instead gives −40 to 280, which
is wrong by a factor of about sqrt(n).

The practical consequence: **repeated simulation adds little for large cohorts
and matters most for small strata** — a rare outcome in a small database, a
single year, an age band. That is also exactly where a difference between two
data sources is most likely to be an artefact, so the predictive interval earns
its keep precisely where the Poisson interval is least informative.
