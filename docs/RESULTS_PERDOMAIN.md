# Re-extraction, the per-domain model, and the negative-control answer

*What was done, what the numbers are, and what it does to the five claims. Written to be read against [`MODEL_V2.md`](MODEL_V2.md) and [`FIDELITY_REPORT.md`](FIDELITY_REPORT.md).*

---

## 0. The short version

1. **The data was re-extracted with end dates.** 5,755,902 records across seven domains. Condition and drug now carry the interval structure the model needs; the empty era tables can be filled.
2. **The per-domain model fits and generates.** Levels are on target (0.94–1.05 of the source on every domain). Prevalence r = 0.95, incidence r = 1.00, co-occurrence r = 0.64, whole-CDM spectral KS = 0.35. That is better than every earlier model in this repo, but the fit is in-sample.
3. **The interval tables are populated.** 179,121 condition eras and 208,028 drug eras from the generated cohort; the source tables are empty. Condition eras are within 17% of what the source occurrences would produce; drug eras are 3.6× too many because the model does not represent repeat-prescription continuity.
4. **The negative-control experiment has an answer, and it is not the one the project wanted.** On adult drug–outcome pairs the raw association is already null (OR 1.01), and contact-volume adjustment leaves it null (OR 0.96). Where a real association exists (all-ages condition pairs, raw OR 1.46), contact-volume adjustment removes it (OR 1.05) — but age adjustment gives the same answer (OR 1.06). **The bias the model removes is age.** Claim 2 survives in the weak sense (the model reproduces the observation process) but not in the strong sense (it uncovers a bias channel that conventional adjustment misses).

The rest of this document is the evidence for those four sentences.

**Data note.** Everything below is the Synthea CDM slice — a rule-based patient simulator, loaded into OMOP 5.4. It is not observed data, not a production CDM, and not observed patients. Every result is a test of whether the model reproduces *another simulator's* output. Nothing here is evidence about a health system. The same pipeline runs on a production CDM by pointing `scripts/extract_dated.sh` at it.

---

## 1. Re-extraction with end dates

`data/events_dated.csv` carried a single `event_date` per record. Condition and drug are interval processes: a condition lasts a median of 364 days and a drug a median of 365 days, and collapsing that to a start date destroys a year of "active" state per interval. Nothing interval-based can be fitted without the end dates.

`scripts/extract_dated.sh` produces `data/events_full.csv`:

| domain | type | rows | end date | unmapped (`concept_id = 0`) |
|---|---|---:|---|---:|
| condition | interval | 295,782 | 100% populated | 2.4% |
| drug | interval | 413,657 | 100% populated | 0.02% |
| visit | interval, degenerate | 474,748 | 100% populated | 0% |
| procedure | point | 673,482 | none | 44.1% |
| measurement | point | 2,142,010 | none | 2.3% |
| observation | point | 1,755,183 | none | 0.01% |
| death | absorbing | 1,040 | none | — |

The row counts reproduce the original audit exactly. Mapped concepts: 218 condition, 202 drug, 156 measurement, 79 observation, 118 procedure, 3 visit. Unmapped records are kept as a per-domain atom, because the null category is a property of the domain (44% of procedures) and has to be generated, not discarded.

### A fixture finding, not a specification claim

`MODEL_V2.md` §2 assumes `log D_j ~ Normal(μ_d, s_d²)`. On Synthea it fails — the observed durations are bimodal:

| | condition | drug |
|---|---:|---:|
| duration = 0 | 22.7% | 12.6% |
| duration > 365 days | 50.0% | 48.6% |
| median, all | 364 d | 365 d |
| median, positive only | 371 d | 371 d |

There is a point mass at zero and a sharp mode near one year, with a heavy tail between. A single log-normal fitted to the positive durations has mean log 5.45 and sd 1.56 (condition) and misplaces the mass; a two-component mixture helps the shape but puts the generated condition median at 208 days against 364 observed.

**But the shape is a property of Synthea, not of CDM data.** Synthea sets many condition end dates and drug supplies by rule (a one-year default is common), so the 0/365 spikes are what the simulator does, not what a health system does. A log-normal fitted here would be a model of Synthea's rules; a quantile table fitted here is a model of the same rules with more numbers. Neither is evidence about the duration distribution a production CDM will have.

What the fixture does establish is narrower and still useful: **the pipeline must estimate the duration distribution rather than assume a family.** The model therefore carries the duration density as an estimated object — here a zero-inflated empirical quantile function (a point mass plus 101 quantiles, sampled by inversion), which reproduces the fixture's quantiles almost exactly — and keeps the log-normal and mixture fits as diagnostics. On a production CDM the first step is to run that diagnostic, not to assume the answer.

The "4 parameters each" in §2.4 is a guess about a distribution nobody has seen. The honest parameterisation is a small shape-free summary, and it is a nuisance component, not a structural claim.

### The death process

The same applies. A single Weibull hazard does not fit Synthea: 10.4% of patients die, at a median of 41 years from observation start, while the fitted Weibull puts the characteristic life at 103 years and under-predicts early mortality. That says Synthea's deaths are module-driven, not that a production CDM's are. The model estimates the death timing nonparametrically — the observed mortality fraction and the empirical within-window distribution — rather than assuming a hazard shape. On real data this should be re-estimated and reported, not reused.

---

## 2. The per-domain model

The fit is the specification of `MODEL_V2.md` §2, with the duration and death layers estimated rather than assumed:

- **Layer 1, per domain.** Interval processes for condition and drug (start process + duration distribution); point processes for measurement, observation and procedure; visit as its own degenerate-interval domain; death absorbing.
- **Layer 2, patient heterogeneity.** One shared `u_i ~ Gamma(k)`, `k = 2.13`, estimated by the marginal negative-binomial likelihood of each patient's total event count. The person-time-weighted mean of the fitted `u` is 0.59; generation matches that, not the unweighted mean of 1, which is what stops the 1.7× over-prediction that cost time earlier.
- **Layer 3, recording dependence.** A rank-20 coupling `exp(z_i · β_j)` on top of the marginal structure, fitted by the damped-Newton factor fit. `z` has a shared Gaussian distribution; generation samples from it, so there is no per-patient parameter in the release. **This is recording dependence, not clinical or causal dependence.** A lab drawn because a condition is suspected is a fact about how the database records care. Every appearance of the coupling in the code and in this document carries that label.

**What transfers, and what does not.** The structural terms — the degree correction, the shared heterogeneity, the cross-domain coupling, the interval/point/absorbing split — are the model. The domain rates, the duration densities and the death timing are estimated per database and are not claims that transfer. A production CDM will have different concept frequencies, different intervals and a different death pattern; the test is whether the same structure fits it, not whether the numbers match Synthea. The fidelity table below is therefore a unit test of the pipeline, not evidence about a health system.

The released object has **17,502 numbers** (`K = 20`, 781 atoms), of which none indexes a patient: the per-patient `u_i` and `z_i` are sampled at generation from their shared distributions, not stored. For comparison, the block model's release was 2,056 numbers; the extra cost buys the intervals, the durations, the death process and the coupling.

### Level check first

Generated totals against the source, before any interpretation:

| domain | generated | source | ratio |
|---|---:|---:|---:|
| condition | 310,385 | 295,782 | 1.05 |
| drug | 389,407 | 413,657 | 0.94 |
| procedure | 643,466 | 673,482 | 0.96 |
| measurement | 2,164,767 | 2,142,010 | 1.01 |
| observation | 1,816,751 | 1,755,183 | 1.04 |
| visit | 473,674 | 474,748 | 1.00 |
| death | 1,036 | 1,040 | 1.00 |

Every domain is within 6% of the source. This is the check that has to pass before the summary statistics below mean anything.

### Summary statistics and spectrum, side by side

| diagnostic | value |
|---|---|
| prevalence, correlation | **0.948** (slope 1.07) |
| incidence, correlation | **0.999** |
| degree distribution, L1 | 0.666 |
| co-occurrence, correlation | **0.641** (over the ~25 concepts with 5–60% prevalence) |
| whole-CDM spectrum, KS | **0.348** |
| effective rank (> 0.1) | source **81**, synthetic **31** |

The spectrum is the degree-normalised incidence matrix's singular values, the same object used in `experiments/12` and `experiments/14`. It is reported here at the same level as the summary statistics, not behind them.

Against the earlier models on the same object: the fixed-effects block model gave prevalence r = 0.86, co-occurrence r = 0.24, spectrum KS = 0.413, rank 8; the visit-anchored model with the heavy-tailed term gave spectrum KS = 0.364, rank 53. The per-domain model improves the summary agreement and the spectrum, and it is the only one that emits intervals. **The comparison is in-sample**, which the earlier held-out numbers were not; that is the main caveat on the table above.

### The empty tables

The generated cohort is passed through an OMOP era builder (same-concept intervals merged when they overlap or fall within a 30-day persistence window). The source tables are empty, so the reference is what the same builder produces from the source occurrence tables:

| table | source occurrence → eras | generated eras | median era length (source / generated) |
|---|---:|---:|---:|
| condition_era | 214,544 | 179,121 | 189 d / 215 d |
| drug_era | 57,710 | 208,028 | 26 d / 370 d |

Condition eras are within 17% of the source-derived count and the lengths are close. **Drug eras are 3.6× the source-derived count**, and that gap is real: Synthea writes chronic prescriptions as repeated refills, so the source has about seven drug records merging into each era (median era 26 days because the course starts earlier and refills overlap), while the generated process draws independent starts and durations and produces about two records per era. The model reproduces the drug duration distribution but not the within-patient refill continuity, which is a renewal-process assumption the specification did not state. The tables are populated and checkable; the drug-era count is not yet right.

That is a concrete deliverable with an honest caveat: a model that emits intervals can fill tables the source leaves empty, and the condition side is close, but "filled" is not "reproduced" for drugs.

![Per-domain fit, levels, and spectrum](figs/fig27_perdomain.png)

---

## 3. The negative-control experiment

This is the kill criterion. The design: drug–outcome pairs with no plausible causal relationship, restricted to an adult cohort (age ≥ 18 at the observation-period midpoint, because Synthea's acute conditions are heavily paediatric and would otherwise test age rather than ascertainment). Twelve chronic cardiovascular/metabolic/anticoagulant drugs against five adult conditions: 60 pairs. The true causal effect is null by construction.

Three estimates per pair:

1. **Raw** — the 2×2 odds ratio.
2. **Adjusted for contact volume** — logistic regression of the outcome on the drug and `log u_i`, where `u_i` is the fitted per-patient contact volume.
3. **OHDSI empirical calibration** — fit a null distribution `N(μ, σ² + se²)` to the raw log-odds-ratios, and recompute p-values against it.

### Result

| estimate | mean log-OR | OR | sd |
|---|---:|---:|---:|
| raw | +0.005 | 1.01 | 0.203 |
| contact-volume adjusted | −0.045 | 0.96 | 0.207 |
| age adjusted (diagnostic) | +0.001 | 1.00 | 0.191 |

Empirical calibration: `μ = +0.021`, `σ = 0.146`. The fraction of pairs significant at 0.05 falls from **18.3% to 1.7%** after calibration.

Read plainly: **the raw drug–outcome associations are already null.** The contact-volume adjustment does not move them toward the null, because there is nothing there to move. It also does not move them away, which is the useful part — it does not manufacture collider bias. But a test in which the null is null before adjustment cannot demonstrate that adjustment removes bias.

### Where the association actually is

The adult pairs are too clean to show the mechanism. The stronger test is condition–condition co-occurrence; the earlier work reported median raw OR 1.71 → 0.91 on 920 all-ages pairs. Re-running the same stratification on a 30-concept subset (430 pairs with both cells well populated) reproduces the direction and separates the two explanations:

| cohort | pairs | raw | contact-volume stratified | age stratified |
|---|---:|---:|---:|---:|
| all ages | 430 | **1.46** | **1.05** | **1.06** |
| adult | 426 | 1.04 | 1.00 | 1.04 |

Two things follow. First, the contact-volume adjustment *does* remove a real association in the all-ages data (1.46 → 1.05), so the kill criterion as literally stated is not triggered. Second, **age stratification does exactly the same thing (1.06)**. The "bias" the model's `u_i` removes is age; `u_i` is a proxy for age because age drives how many records a patient accumulates. In the adult cohort, where age is roughly held fixed, there is no bias for either adjustment to remove.

### Verdict

- The mechanism is real: contact volume predicts record volume, and adjusting for it removes spurious co-occurrence in the all-ages data.
- The contribution is not established: the same adjustment for age does the same job. The model has not been shown to uncover a bias channel that conventional age adjustment misses.
- On adult drug–outcome pairs, raw, adjusted and empirically calibrated estimates all agree on the null. That is a clean negative result for the *magnitude* of ascertainment bias in this slice, and a clean non-interference result for the adjustment.

**Claims 2 and 3 are therefore not dead, but they are not supported.** The decisive experiment that was supposed to separate the model from the field's established method returned the field's established method's answer.

![Negative-control experiment](figs/fig28_negative_control.png)

The epidemiology references for the design are the standard ones: negative controls as a bias-detection tool (Lipsitch, Tchetgen Tchetgen & Cohen 2010, *Epidemiology*, PMID 20335814), empirical calibration (Schuemie et al. 2018, *Phil Trans R Soc A*, PMID 30082302), and the collider mechanism that makes contact-volume adjustment dangerous (Griffith et al. 2020, *Nature Communications*, PMID 33184277). All three identifiers were checked against PubMed.

---

## 4. What this does to the five claims

| claim | before | now |
|---|---|---|
| **1. Topology can model a CDM dataset** | partly done, Synthea only | **stronger**: the per-domain model fits and generates with intervals, prevalence r 0.95, incidence r 1.00, spectrum KS 0.35. Still Synthea only, still in-sample. |
| **2. The structure uncovers systematic bias** | demonstrated on synthetic data only | **not demonstrated**: the bias `u_i` removes is age. Adult drug–outcome pairs are already null. Weak support, not the strong claim. |
| **3. Calibrate incidence and prevalence** | machinery built, nothing demonstrated | **weaker**: the model's "null" is effectively an age-standardised null. The calibration route inherits claim 2's problem. |
| **4. Synthetic data with realistic properties** | passes marginals, fails spectrum | **better**: levels on target, intervals and eras emitted, prevalence r 0.95, spectrum KS 0.35. Rank still 31 against 81; in-sample; drug era counts 3.6× high. |
| **5. Small, shareable, comparable** | the strongest claim | **intact**: 17,502 numbers for K = 20, no per-patient terms (the empirical death-timing array and the duration model are the two places where "small" is softer than the block model's 2,056). |

The honest one-line summary: **the model got better at reproducing the database, and the experiment that was supposed to show it understands the database's biases showed that it understands age.**

---

## 5. Running it on a large database

The pipeline is written to fit on a sample, because the released parameters are all `O(C)` or `O(K²)` and none indexes a patient:

- the marginal fit is sparse IPF — it never materialises an `N × C` dense array;
- the coupling is fitted on at most `max_factor_patients` (default 5,000) patients, since only the concept loadings and the latent distribution are retained;
- generation is vectorised Poisson per domain-year over active patients.

Measured on this machine:

| patients fitted | wall time | `k` | person-time mean `u` |
|---:|---:|---:|---:|
| 2,000 | **2.8 s** | 2.26 | 0.66 |
| 10,000 | **10.3 s** | 2.13 | 0.59 |

The parameters are stable at a fifth of the sample. On a production CDM the sequence is: `SAMPLE_PCT=0.05 scripts/extract_dated.sh` → `fit_perdomain(..., max_patients=...)` → `generate_cohort(...)`. The extract's `SAMPLE_PCT` draws a consistent patient subset in one pass (a materialised CTE, so the same patients appear in every domain table), which keeps the extraction cost proportional to the sample rather than the table.

---

## 6. What I would do next, in order

1. **Run the negative control on a production CDM.** This is the only way to know whether the age-proxy result is a property of Synthea or of the method. The code is database-agnostic; the extraction takes a `PSQL` override.
2. **Make the negative-control cohort age-matched, not just adult.** The current design controls age by restriction; the next one should match on age and sex within the pair, so the residual is ascertainment.
3. **A held-out fidelity check for the per-domain model.** The table in §2 is in-sample. The block model's held-out numbers are the right template.
4. **Rank.** `K = 20` is the specification's a priori choice, not a tuned one. The spectrum still under-ranks (31 against 81). Whether a larger `K` closes it, and at what parameter cost, is a clean question — but it must not be selected on the spectral criterion that it is reported against.
5. **Durations by concept, not by domain.** The quantile table is a domain average; chronic conditions and acute ones have different duration profiles within a domain.
6. **Repeat-prescription continuity.** The drug-era count (208,028 against 57,710 source-derived) is the clearest structural gap. A per-patient, per-concept renewal process — refill with high probability when the previous interval ends — is the smallest change that would close it.
