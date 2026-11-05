# The per-domain model on the MIMIC-IV demo

*First run on observed, de-identified EHR data. 100 ICU patients, OMOP CDM 5.3, from the MIMIC-IV demo in OHDSI/EunomiaDatasets (Kallfelz et al. 2021, doi:10.13026/p1f5-7x35). This is not a production CDM and not a cohort study.*

---

## 0. The short version

1. **The fit runs on real data.** 100 patients, 423,924 records, fitted in 1.1 s. The model does not fall over on a different schema or a different observation process.
2. **The era builder is validated against a real OMOP build.** Applied to the MIMIC condition occurrences it produces **3,771 condition eras, median 6 days — exactly the MIMIC `condition_era` table**. Drug eras are within 11% (8,801 against 7,931).
3. **The generator does not reproduce the interval structure.** It over-produces eras (condition 5,365 against 3,771; drug 12,723 against 7,931) and under-lengths them (median 0–1 day against 6 and 4). Event *counts* are on target; within-patient temporal clustering is not.
4. **The real data's properties are not Synthea's.** `k = 0.50` against Synthea's 2.13, person-time-weighted `u = 0.07` against 0.59, condition durations a median of 0 days against 364. This is the concrete version of the fixture critique: the shapes do not transfer, and on this data the model's fancier layer is doing something quite different.
5. **The spectrum is worse than Synthea's, not better.** Non-zero spectral KS 0.52, rank 61 against the synthetic 20. The zero-padded KS of 0.019 is an artefact of 2,700 concepts for 100 patients and should be ignored.

---

## 1. What the data is

`data/mimic_eunomia/` — the 100-patient MIMIC-IV demo, OMOP CDM 5.3, downloaded from [OHDSI/EunomiaDatasets](https://github.com/OHDSI/EunomiaDatasets) (`datasets/MIMIC/MIMIC_5.3.zip`). Real de-identified records from Beth Israel Deaconess; the dates are shifted into 2110–2202 by the MIMIC de-identification, so the calendar profile is meaningless in absolute terms but the intervals are preserved.

| domain | records | type |
|---|---:|---|
| condition | 16,441 | interval, 76.9% zero-duration |
| drug | 18,229 | interval, median 1 day |
| measurement | 338,550 | point |
| observation | 31,390 | point |
| procedure | 18,447 | point |
| visit | 852 | interval |
| death | 15 | absorbing (15%) |

The observation process is dense in-hospital monitoring: 3,385 measurements per patient. This is not outpatient care, and it is not a claims database.

---

## 2. Fit

`fit_perdomain(events, periods, K=5)`.

| quantity | value |
|---|---:|
| patients | 100 |
| atoms (concepts + unmapped) | 2,723 |
| `k` (Gamma heterogeneity) | **0.50** |
| person-time-weighted mean `u` | **0.072** |
| death fraction | 0.15 |
| released parameters | 17,121 |
| coupling rank `K` | 5 |
| fit time | 1.1 s |

`K = 5`, not the specification's 20: a rank-20 latent Gaussian has 210 covariance parameters, which cannot be estimated from 100 patients. `K` is a small-sample choice here, stated rather than tuned.

`k = 0.50` is the substantive number. The patient heterogeneity is far heavier than Synthea's (`k = 2.13`): a small number of ICU patients accumulate a large share of the records, and most accumulate few. The person-time-weighted mean `u = 0.07` says the same thing from the other side — the fitted contact volume is concentrated, not spread.

---

## 3. Level check

With `k = 0.50` and `n = 100`, the sum of the sampled propensities has a relative standard deviation of `1/sqrt(n k) = 14.2%`. That is the noise floor for this table, not a tolerance to be tightened.

| domain | generated | source | ratio |
|---|---:|---:|---:|
| condition | 17,890 | 16,441 | 1.09 |
| drug | 16,500 | 18,229 | 0.91 |
| procedure | 17,085 | 18,447 | 0.93 |
| measurement | 333,243 | 338,550 | 0.98 |
| observation | 35,597 | 31,390 | 1.13 |
| visit | 698 | 852 | 0.82 |
| death | 16 | 15 | 1.07 |

Every ratio is within about 1.3 noise standard deviations. The visit ratio (0.82) is the largest deviation and is not interpretable at this sample size.

---

## 4. Summary statistics and spectrum

| diagnostic | value |
|---|---|
| prevalence, correlation | 0.933 |
| incidence, correlation | 0.990 |
| degree distribution, L1 | 1.840 |
| co-occurrence, correlation | 0.566 |
| spectral KS, **non-zero part** | **0.520** |
| spectral KS, zero-padded | 0.019 (artefact) |
| effective rank (> 0.1) | source **61**, synthetic **20** |

The gap between the two KS values is a trap worth recording. With 2,700 concepts and 100 patients, 2,600 of the 2,700 singular values are structurally zero in both matrices. The zero-padded KS is measuring the agreement of the zeros, not of the data. The non-zero KS (0.52) and the rank (61 against 20) are the honest numbers.

The rank gap is the same failure as on Synthea, and larger: the model's coupling has rank 5, the heavy-tailed `u` lifts the realised rank to 20, and the real matrix has 61.

---

## 5. The era tables — the new check

MIMIC is the first database in this project whose `condition_era` and `drug_era` tables are populated. That gives a real reference: apply the same builder to the occurrences, and compare.

| table | MIMIC build | our builder on the same occurrences | generated |
|---|---:|---:|---:|
| condition_era | 3,771 rows, median 6 d | **3,771 rows, median 6 d** | 5,365 rows, median 0 d |
| drug_era | 7,931 rows, median 4 d | 8,801 rows, median 3 d | 12,723 rows, median 1 d |

Two separate findings.

**The builder is right.** On conditions it reproduces the MIMIC build exactly — same count, same median length. On drugs it is 11% high (8,801 against 7,931), which is consistent with a different persistence window or concept grouping in the MIMIC ETL (the source's `gap_days` values run 0–6, so it is not a simple gap mismatch). The merge logic itself is validated.

**The generator is not.** It produces 1.42× (condition) and 1.60× (drug) too many eras, and they are too short. The reason is structural: the real data has multiple records per patient and concept close together in time, so the era builder collapses them; the generator draws independent starts and durations, so fewer records overlap and fewer merge. The model reproduces the marginal event counts and the duration density, but not the within-patient temporal clustering. This is the same gap that produced the 3.6× drug-era over-count on Synthea, now visible on real data and on both interval domains.

That is the honest headline of this run: **the pipeline and the era builder work on real data; the generator's interval process is missing a dependence that both databases have.**

---

## 6. What this does and does not say

It does say:
- the per-domain model runs on a real, differently-shaped OMOP CDM without modification;
- the era builder reproduces a real OMOP era build on conditions;
- the real data's heterogeneity, contact-volume and duration properties are very different from Synthea's — the fixture critique is not hypothetical;
- the interval generator has a specific, identifiable structural gap (within-patient clustering), not a vague "fidelity" problem.

It does not say:
- anything about the age-versus-contact question. There are no adult outpatient negative-control outcomes in this demo, the cohort is 100 ICU patients, and the recording process is in-hospital monitoring. The negative-control experiment needs a claims/outpatient CDM.
- anything at production scale. 100 patients gives a 14% level-check noise floor and caps the coupling rank at 5.

Full MIMIC-IV (~300k patients, PhysioNet credentialing, no fee) would fix the scale and the rank; it would not fix the ICU-versus-outpatient mismatch. The next real test of the bias question is still a claims CDM.

![MIMIC run](figs/fig29_mimic.png)
