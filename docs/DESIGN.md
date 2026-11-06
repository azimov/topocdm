# Design notes: a likelihood for a fixed-density topology model

This document records why the model is the way it is, and the two decisions
that make the inference defensible. It is the argument I would want a
statistician referee to read.

## 1. The problem with the original formulation

CiGRAM (Gilbert 2015, Ch. 5) draws exactly `m` edges without replacement from
the set of possible edges, with edge weights

```
W_ij = beta_i + beta_j
```

sampled through a reservoir method. The graph is fixed-density by construction.
The cost is that there is no closed-form probability for a realised graph: the
distribution of a without-replacement sample of weighted edges is a Wallenius
noncentral hypergeometric, which does not factorise over edges. So the model
could be simulated but not estimated, and every comparison in the thesis had to
be made on hand-weighted summary statistics.

That is the whole reason CiGRAM never became a statistical model.

## 2. Two observations that remove the obstacle

**(a) Fixed density is a conditioning event, not a constraint on the model.**

Fix the total number of events `M`. Then

```
P(X | Theta) = P(X, M | Theta) = P(M | Theta) * P(X | M, Theta)
```

and `P(X | M, Theta)` is multinomial. The multinomial likelihood over a
patient × concept incidence matrix is closed-form. Nothing is approximated.

The reason this is legitimate rather than a convenience is that the Poisson and
the conditional (multinomial) models have the *same* MLE for the parameters
that enter the cell probabilities. The Poisson total `M` is ancillary for those
parameters. So:

- fit by maximising the Poisson likelihood (or, equivalently, the conditional
  one — they agree);
- generate by conditioning on `M`.

The fixed-density structure survives into generation exactly. A synthetic
database has the same number of records as the target, so "does it have the
right arrangement of events" is a real question rather than an artefact of a
wrong event count.

**(b) A CDM is a multigraph.**

A patient can be coded with the same concept repeatedly. There is no
without-replacement constraint at the cell level, so the Wallenius distribution
never arises. The edge budget is still fixed; the individual events are
exchangeable within cells.

This is not a modelling compromise forced on us by the data. It is what the
data are.

## 3. The model

```
mu_ij = T_i * u_i * lambda_j * B[g_i, h_j]

X_ij | u, lambda, B, g  ~  Poisson(mu_ij)
```

with `T_i` person-time as a *known* offset, so that the model is about rates
rather than counts.

Interpretation:

- `u_i` is care-seeking propensity. It is the same object as CiGRAM's latent
  position: a scalar that determines both how many records a patient has and
  which ones they are likely to be.
- `lambda_j` is the concept's intensity per unit of contact.
- `B[g, h]` is the affinity between patient group `g` and concept group `h`.
  This is the part of the topology that the marginals do not contain: two
  patients with the same propensity and two concepts with the same prevalence
  can still co-occur more than independence predicts.

The epidemiological reading is direct. `u_i` is utilisation. `B` is
multimorbidity. Estimating them separately is the point.

## 4. Identifiability

Two constraints, both enforced by renormalisation after every update with the
compensating factor pushed into `lambda` so the mean matrix is unchanged:

```
mean(u) = 1                             fixes the u / lambda scale
sum_g w_g B[g,h] / sum_g w_g = 1        fixes the lambda / B scale within
                                        concept group h, w_g = sum_{i in g} T_i u_i
```

What is *not* identified, and should be stated in any paper:

- the labelling of patient groups (label switching);
- `lambda` and `B` jointly, within a concept group, except through the
  constraint above;
- `u_i` for a patient with very few events — the MLE is noisy, and this is why
  the propensity distribution for generating new patients is fitted by marginal
  likelihood (§6) rather than by plugging in `u_i`.

## 5. Fitting

Closed-form coordinate ascent, no numerical optimiser:

```
u_i      = rowsum_i / ( T_i * (R @ (B @ Lam))_i )
lambda_j = colsum_j / ( A @ B )[h_j],          A_g = sum_i R_ig T_i u_i
B[g, h]  = sum_i R_ig n_ih / ( A_g * Lam_h )
```

where `R` are the responsibilities and `n_ih` the counts per patient × concept
group. Each line is the exact maximiser given the others.

For the single-group model `R` is all ones and this reduces to IPF: alternating
row-sum and column-sum matching. Verified against a general-purpose optimiser
to 1e-6 in `tests/test_model.py::test_ipf_matches_optimiser`.

Two ordering details matter, and both were bugs first:

1. `A_g` must be computed from the **updated** `u`, not the previous one.
   Using a stale `A_g` breaks the EM monotonicity guarantee, and the
   log-likelihood will visibly oscillate.
2. Parameter normalisation must leave the mean matrix unchanged. It does, but
   only if the compensating factor goes into `lambda` at the same time.

## 6. Generating new patients rather than replaying old ones

Two modes:

- **replay** — keep the fitted `u_i` and groups, redraw the counts. A posterior
  predictive check.
- **fresh** — draw new propensities. The actual synthetic-CDM generator.

For *fresh* mode, plugging the fitted `u_i` into a log-normal is wrong. `u_i` is
estimated from a handful of events per patient, so its spread includes
measurement error, and a distribution fitted to `u_hat` over-states the true
heterogeneity. Synthetic databases generated this way were visibly too
concentrated: Gini 0.79 against a target of 0.53.

The fix is to integrate the propensity out. With `u ~ Gamma(k, 1/k)`,

```
X_i. | g  ~  NegBin(mean = T_i * Lambda_i(g), size = k)
```

and `k` is estimated by maximising that marginal likelihood over patients. This
is the correct marginal, and it reproduces the observed concentration rather
than exaggerating it. This is the single change that made fresh generation work.

## 7. Fitting versus model selection

The classification likelihood — condition on a hard assignment, fit, measure —
is what EM computes internally, and it is *not* the likelihood the model
implies. It over-states fit, because each patient's group assignment is then an
estimated parameter that never appears in a parameter count.

Using it for AIC produced exactly the failure you would predict: the criterion
improved monotonically with the number of groups, picking G = 8 when the truth
was 4. Switching to soft EM on the mixture likelihood,

```
L = sum_i log sum_g pi_g prod_j Poisson(x_ij | mu_ijg)
```

with `N-1 + C + G(H-1) + (G-1)` parameters, fixed it: AIC and BIC then recover
the true group count on the nose.

The general lesson, which is the same lesson as the thesis: a model that can be
simulated but not scored will always look adequate.

## 8. What this does not yet do

- **No real CDM has been fitted.** All results are on simulated data with known
  truth. The immediate next step is fitting an OMOP instance and checking that
  the observation model reproduces its visit and channel structure.
- **No channels.** The programme notes argue that ascertainment is channel-specific
  (examination vs laboratory vs prescribing) and that this is what breaks
  negative-control calibration. The current model has one implicit channel.
  Extending it means `mu_ij = T_i u_i sum_c lambda_jc D_ic` with a per-patient
  channel-intensity vector `D` — a natural next iteration, and the reason to
  keep the person-time offset explicit.
- **No time.** Events are aggregated over the observation period. Incidence is
  modelled as a rate, not as a trajectory.
- **No covariates.** Age, sex and care site are not yet in the model; they would
  enter as offsets or as block structure.

## 9. Scaling: why the fit streams

The model separates across patients once `lambda`, `B` and `pi` are fixed.
`u_i` is local; the global parameters depend on the data only through

```
colsum_j = sum_i X_ij            events per concept
n_ih     = sum_{j in h} X_ij     events per patient per concept group
A_g      = sum_i R_ig T_i u_i    propensity-weighted group mass
```

Note what is *not* in that list: nothing indexed by both a patient and a
concept. The E-step and the `u` update need only `n_ih`, a vector of length H
(the number of concept groups, typically tens), not the patient's full row.

So the fit is one chunk of patients at a time, in memory
`O(chunk * (H + G) + C)`, and a 200,000 x 1,000 CDM with 8M events fits in
seconds (experiments/03_scale.py). The dense matrix would be 1.6 GB before any
arithmetic.

Three consequences worth stating:

- **Subsampling patients is exact for the release.** The released artefact
  (`docs/PRIVACY.md`) contains `lambda`, `B`, `pi` and a propensity shape — no
  `u_i`. A random patient sample estimates exactly those quantities. It is not
  an approximation to the thing being published; it is the thing being
  published, estimated on a sample.
- **A fresh sample each epoch is legitimate.** Because the global sufficient
  statistics are sums over patients, redrawing the sample between epochs keeps
  the estimate unbiased and bounds the working set. 25% of patients reproduces
  the full-data `lambda` to L1 0.014.
- **Warm start, then stream.** The block structure is latent, so a cold start
  has a symmetry problem: with `B` uniform every group scores identically and
  the responsibilities never split. The fit therefore runs the exact EM on a
  bounded sample first (a few thousand patients), and uses that as the starting
  point for the stream. This also means initialisation and refinement share one
  code path.

### 9.1 Let the database do the aggregation

The corollary for every consumer of a CDM, not just the fit:

> **Never stream a large object to the client if an aggregate will do.**

The database engine is the only component that can scan a fact table without
materialising it. Anything a client pulls across the wire is — by definition —
already in memory on the wrong machine. At 10,000 patients the event frame for
one Synthea extract is 5.76M rows; at 100,000 it is ~24M rows and several
gigabytes, and the fidelity report that needed it was doing nothing with it
that SQL could not do in one pass.

The rule, in the form it should be applied:

1. **Ask what the consumer actually uses.** The fidelity report needs, per
   concept, the number of distinct patients and the number of records. That is
   `group by domain, concept_id` — 778 rows against 5.76M, a factor of 7,400.
   `fit_durations` needs a zero fraction and 101 quantiles, not every interval.
   `lam_t` needs events per domain-year, which is one row per domain per year.
2. **Reduce in the engine, then transfer.** The Python side receives O(C) or
   O(C × years) numbers. A temp or permanent table in the results schema is the
   right intermediate when the reduction has more than one step; there is no
   need to return a wide join to the client just to break it up in pandas.
3. **When a per-row pass is genuinely required, bound it first.** The coupling
   fit is the one place that wants rows rather than counts, and it already
   takes at most 5,000 patients. That is a bounded sample, not the database.
   Sample by the same deterministic patient hash the rest of the pipeline uses,
   so the sample is stable and reproducible.
4. **Map-reduce the updates, not the data.** A fit that needs sufficient
   statistics per chunk can reduce them in SQL and combine on the client: the
   marginal layer needs only `rowsum`, `colsum` and `T`, all of which are
   group-bys. Fitting on a bounded sample and correcting the level from
   population aggregates is exact for the quantities the release publishes,
   because no released parameter is per-patient (see §9).
5. **Keep the two paths bit-identical.** A client-side and a SQL-side
   implementation of the same statistic must agree to the last digit, not
   approximately. They will not, by accident: `(X > 0).mean()` over a sparse
   matrix and a ratio of two integers differ by ~1e-16, and a coverage test
   against an interval whose bound *is* that value flips on the difference.
   Count integers and divide once, in both paths.
6. **Every aggregate must describe the same cohort.** Moving work into SQL
   introduces a new failure mode: two queries that each look right but cover
   different patient sets. A capped fit sample (`fit_patients`) plus uncapped
   counts gives you one cohort's records over another cohort's person-time,
   which doubled real incidence and drove coverage to zero while every
   individual number looked plausible. The cohort has to be explicit — pass the
   person ids the extract used, or take every aggregate over the same sample
   table — and the reported `n_patients_real` should equal the count of the
   cohort the fit saw.

The failure this prevents is not slowness, it is a hard stop: a client-side
materialisation that works at 10k patients fails outright at 100k, and it fails
on the machine that has the least memory rather than the one that has the data.
Every place this codebase touches a CDM should be checkable against the rule,
and the check is a single question — *could this be a `group by`?*

## 10. Density is a generator parameter, not a fit output
Fitting estimates the observed density (events per person-year). Generation
takes a density and allocates `M = density * sum(T)` events multinomially, which
is what fixed density means. The same release therefore produces a sparse or a
dense cohort without refitting: requested 2, 8 and 20 events/person-year are
realised exactly.

The connection back to CiGRAM is exact here. The thesis fixed the number of
edges and sampled them without replacement — a model that could not be scored.
Conditioning a Poisson model on the same total keeps the fixed-density
structure, makes the likelihood closed form, and (because a CDM is a
multigraph) removes the without-replacement constraint that caused the problem.
