# What leaves the building: the disclosure posture of a `Release`

This note exists because the point of the project is to share something useful
about a CDM without sharing the CDM. It states precisely what a `Release`
contains, what it does not, and where the residual risk sits.

## 1. What is released

`make_release(...)` returns:

| field | size | what it is |
|---|---|---|
| `lam` | C | concept intensity, one number per concept |
| `B` | G x H | patient-group x concept-group affinity |
| `pi` | G | group mixing weights |
| `propensity_shape` | 1 | the Gamma shape of care-seeking propensity |
| `concept_ids` | C | which concepts, from a standard vocabulary |
| `concept_group` | C | the grouping used above |
| `density` | 1 | events per person-year |

For a CDM with 200,000 patients and 1,000 concepts that is **2,056 numbers**.
The number of patients does not enter the size of the release at all.

## 2. What is deliberately *not* released

The fit also estimates `u_i`, one propensity per patient — and `u_i` is
genuinely patient-level information. It is used during fitting and is then
**discarded**. `make_release` takes only `lam`, `B`, `pi` and the shape `k`.

`sample_release` draws *new* propensities from the fitted Gamma. A synthetic
cohort is therefore a draw from a population model, not a perturbation, jitter
or resampling of the observed cohort. There is no code path from a `Release`
back to a fitted `u_i`.

## 3. Why this is not "an ML model with weights that could map to patients"

The comparison people will make is to a generative model with millions of
parameters fitted to records, where memorisation is a real failure mode. The
differences are structural, not promises:

- **Parameter count is O(C + GH), independent of N.** There is no capacity to
  memorise a patient because there is no per-patient parameter in the object.
- **The sufficient statistics are aggregates.** The fit depends on the data
  only through `colsum_j` (events per concept), `n_ih` (events per patient per
  *concept group*) and `A_g`. The last two are sums over groups, not over
  individuals. No sufficient statistic in the model is indexed by a person,
  except `u_i`, which is not released.
- **Generation is stochastic and population-level.** Two draws from the same
  release differ as much as two real cohorts would.

That is why fitting on a patient subsample is neither a shortcut nor a privacy
concession: the sample estimates exactly the quantities that are released, and
none of the quantities that are not.

## 4. Residual risk, stated honestly

These are the ways a release could still disclose something, in decreasing
order of plausibility.

1. **Small-cell / rare-concept leakage in the parameters.** `lam_j` for a very
   rare concept is estimated from a handful of events. With a small denominator
   it can approach "n patients had this", which is a small-cell disclosure. Mitigation: suppress
   concepts below a minimum event count before fitting, and publish the
   suppression rule.
2. **Group structure as quasi-identifier.** A concept group with a single
   member, combined with a small population, can point at a condition. Mitigation: require minimum group size.
3. **Composition plus external data.** If an attacker knows the concept list,
   the density and that a given person is in the source database, they learn
   that person's *propensity percentile*, not their diagnoses. This is a weak
   disclosure but it is not zero.
4. **Differencing.** Two releases fitted on overlapping cohorts (e.g. before
   and after adding a group of patients) differ in `lam`, so differences leak
   information about the added patients. Mitigation: control release frequency.

## 5. What to add before this is used in anger

- **Differential privacy on the sufficient statistics.** `colsum_j` and the
  `n_ih` sums are counts; adding calibrated noise to them before the M-step
  gives a formal guarantee with a bounded, reportable privacy budget. This is
  the natural next step and is cheap because the model already consumes the
  data only as sums.
- **A canary test.** Insert a small number of synthetic unique patients,
  refit, generate, and check that no canary's concept set is reproduced more
  often than chance. This is the empirical counterpart to §3 and should be run
  on every release.
- **A minimum-cell rule** for concepts and concept groups, applied before
  fitting, with the rule published alongside the release.

## 6. The sentence to use

> The released artefact is a set of population-level parameters — concept
> intensities, group affinities, mixing weights and a propensity distribution —
> fitted to aggregate sufficient statistics. It contains no per-patient
> parameters. Synthetic patients are drawn from this population model; they are
> not perturbed real patients, and there is no code path from the released
> parameters back to an individual record.

That sentence is defensible because §1 and §2 are enforced in code, not by
convention: `make_release` drops `u`, and `sample_release` never sees it.
