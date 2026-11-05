# Fidelity report: prevalence and incidence

*Living document. Version 2, 2026-11-05. Code: `experiments/25_fixes_fidelity.py`. Raw results: `results/fixes_fidelity.json`.*

This is the single place where the model's per-concept **prevalence** and **incidence** fidelity is recorded, together with how those quantities are computed. Update it by re-running the experiment, replacing the numbers and figures below, and adding a line to the change log.

**Data status.** The only datasets here are the Synthea slice (a rule-based patient simulator) and the 100-patient MIMIC-IV demo (real de-identified data, but a small ICU sample). Neither is a production CDM. Nothing below is evidence about a health system.

---

## 1. How the quantities are calculated

### 1.1 Prevalence

For concept `j`, prevalence is the fraction of patients who have **at least one** record of `j` during their observation period:

```
P_j = (1 / N) * sum_i 1[ X_ij > 0 ]
```

where `X_ij` is the number of records of concept `j` for patient `i`, and `N` is the number of patients. It is a **patient-level** quantity: one record is enough, and ten records count the same as one.

### 1.2 Incidence

For concept `j`, incidence is the total number of records divided by the total person-time at risk:

```
I_j = 1000 * ( sum_i X_ij ) / ( sum_i T_i )
```

where `T_i` is patient `i`'s observation period in years. It is a **record-level** quantity, reported per 1,000 person-years. One patient with 50 records contributes 50, not 1.

A model can get one right and the other wrong. This one gets the total record counts right and spreads rare concepts across too many patients; that is a prevalence failure with a correct incidence.

### 1.3 Concept set and denominator

Both quantities are computed on the mapped clinical concepts — the five clinical domains (condition, drug, measurement, observation, procedure), excluding unmapped (`concept_id = 0`) records. Results are reported for concepts with real prevalence at least 1%, because the ratio is unstable below that. The number compared is stated in every table: on Synthea it is **446 of 679** concepts (real prevalence ≥ 1%); on MIMIC it is **2,715 of 2,715**.

### 1.4 Prevalence bins

A single prevalence-coverage number can hide which range fails. Coverage is therefore reported in three bins:

| bin | real prevalence |
|---|---|
| rare | 1–3% |
| mid | 3–30% |
| common | 30–100% |

The bins are fixed in advance and are not tuned on the coverage they report.

### 1.5 Synthetic cohorts and the simulation interval

The model is fitted to the real data, then `R` synthetic cohorts are generated. Each cohort has the same number of patients as the real data, with observation periods resampled from the observed `(start, duration)` pairs, so total person-time and the follow-up distribution are preserved.

For each synthetic cohort `r`, the same `P_j` and `I_j` are computed. The **simulation interval** for concept `j` is the 5th to 95th percentile of those `R` values:

```
[ P_j^(5), P_j^(95) ]        [ I_j^(5), I_j^(95) ]
```

`R = 50` on both datasets.

### 1.6 Predictive coverage

**Predictive coverage** is the fraction of compared concepts for which the real value lies inside the simulation interval:

```
coverage = (1 / |J|) * sum_j 1[ real_j in [lo_j, hi_j] ]
```

where `|J|` is the number of compared concepts. A high correlation across concepts says the ordering is right; coverage says the level is right. The two can disagree, and here they do.

### 1.7 Ratio

The ratio is `synthetic median / real`. A value of 1 is perfect; above 1 means over-prediction. The **incidence ratio** is reported alongside coverage because 100% coverage of a wide interval is not calibration; the ratio is the check that the level is right.

---

## 2. What is fitted, and on what

| | Synthea | MIMIC-IV demo |
|---|---|---|
| source | rule-based simulator | real de-identified EHR |
| patients in the report | 2,000 (sample of 10,000) | 100 |
| encounters | derived `(person, date)` key | `visit_occurrence_id` |
| model | visit-anchored, `K = 5` | visit-anchored, `K = 5` |
| synthetic cohorts `R` | 50 | 50 |

The **visit-anchored model** generates a patient as a sequence of encounters: a number of visits, then a set of events inside each visit, then durations for conditions and drugs. It was chosen over the per-domain model because the per-domain generator scatters events across days and produces almost no same-day condition–drug graphs. The derived encounter key is the `(person, date)` proxy; it is the only encounter definition available in Synthea, which has no visit IDs.

The fidelity loop does **not** apply the refill re-dating. Refill preserves record counts and therefore cannot change prevalence or incidence; it is exercised separately, and omitting it keeps the R = 50 run tractable.

---

## 3. Results

![Synthea prevalence and incidence](figs/fig39_synthea_fidelity.png)

![MIMIC prevalence and incidence](figs/fig39_mimic_fidelity.png)

### 3.1 Overall

| dataset | R | concepts compared | prevalence coverage | incidence coverage | incidence ratio (median) |
|---|---:|---:|---:|---:|---:|
| Synthea (2k) | 50 | 446 of 679 | **43%** | **100%** | 0.99 |
| MIMIC-IV demo | 50 | 2,715 of 2,715 | **82%** | **100%** | 1.00 |

### 3.2 By prevalence bin

| dataset | bin | n | prevalence coverage | over / under | prevalence ratio | incidence ratio |
|---|---|---:|---:|---|---:|---:|
| Synthea | rare 1–3% | 180 | **57%** | 78 / 0 | 1.02 | 0.99 |
| Synthea | mid 3–30% | 189 | **47%** | 84 / 17 | 1.13 | 0.99 |
| Synthea | common 30–100% | 77 | **1%** | 48 / 28 | 1.19 | 1.00 |
| MIMIC | rare 1–3% | 1,563 | **93%** | 111 / 0 | 1.00 | 0.98 |
| MIMIC | mid 3–30% | 893 | **81%** | 164 / 8 | 1.12 | 1.01 |
| MIMIC | common 30–100% | 259 | **19%** | 31 / 180 | 0.75 | 1.01 |

**Incidence is calibrated everywhere.** The incidence ratio is between 0.98 and 1.01 in every bin on both datasets, and coverage is 100%. The model reproduces how often each concept is recorded per unit of follow-up.

**Prevalence is not.** On Synthea, the gate fails in all three bins (57%, 47%, 1%). On MIMIC, rare and mid pass (93%, 81%) but common fails badly (19%). The overall numbers (43% and 82%) average over bins that fail for different reasons.

---

## 4. The prevalence gap is two mechanisms

### 4.1 Rare and mid: over-spread

In the rare bin, every miss is an over-prediction on both datasets (78 over / 0 under on Synthea; 111 over / 0 under on MIMIC). The model spreads a rare concept across too many patients: a concept that the real data concentrates in a few patients is given to many, each with few records. This is the "spread too thin" mechanism, and a per-patient concept preference (topic/Dirichlet layer) is the natural candidate.

### 4.2 Common: catalogue / assignment, not level

The common bin is a different failure. On Synthea the model over-predicts prevalence (ratio 1.19); on MIMIC it under-predicts (0.75). Both datasets have incidence ratio ≈ 1.00, so the total records per common concept are right; what is wrong is **which patients get the concept**.

A diagnostic over 5 Synthea / 10 MIMIC cohorts, common bin, all quantities synthetic ÷ real:

| quantity | Synthea | MIMIC |
|---|---:|---:|
| records per concept | 0.989 | 1.018 |
| visits per patient | 0.989 | 1.044 |
| events per visit | 0.999 | 0.984 |
| distinct common concepts per patient | **1.136** | **0.778** |
| records per patient-with-concept | **0.839** | **1.339** |
| prevalence | **1.185** | **0.760** |

The visit count, the events per visit, and the total records are on target. The model gives each patient the wrong number of distinct common concepts — too many on Synthea, too few on MIMIC — and that moves prevalence in opposite directions.

**Diagnosis: the common-bin failure is a catalogue/assignment error, not a visit-count or `u_i` level error.** A topic layer aimed only at rare-concept co-occurrence is not obviously sufficient for it; the common bin must be checked separately after any new layer.

---

## 5. The gate

Prevalence fidelity passes only when **all** of the following hold on both fixtures:

- prevalence coverage **≥ 80% in each bin** (rare, mid, common), not one scalar;
- **incidence ratio** median between 0.9 and 1.1 in each bin;
- `R = 50`;
- the denominator and bin counts stated alongside the result.

Current status:

| dataset | rare | mid | common | gate |
|---|---:|---:|---:|---|
| Synthea | 57% (fail) | 47% (fail) | 1% (fail) | **fail** |
| MIMIC | 93% (pass) | 81% (pass) | 19% (fail) | **fail** |

---

## 6. How to update this report

1. Re-run the experiment:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/25_fixes_fidelity.py
```

2. Replace the overall and per-bin numbers from `results/fixes_fidelity.json`:

```python
import json
r = json.load(open("results/fixes_fidelity.json"))
r["synthea"]["bins"]        # rare_1_3, mid_3_30, common_30_100
r["synthea"]["incidence_ratio_median"]
r["mimic"]["bins"]
```

3. Rebuild the PDF:

```bash
bash scripts/md2pdf.sh docs/FIDELITY_REPORT.md docs/FIDELITY_REPORT.pdf
```

4. Add a row to the change log and bump the version line at the top.

If the model or the sample changes, also update §2 (what is fitted, and on what) and the figure captions.

---

## 7. Change log

| version | date | change |
|---|---|---|
| 1 | 2026-11-05 | First version. Overall coverage: Synthea 40.6%, MIMIC 81.8%; R = 20 / 50. |
| 2 | 2026-11-05 | Split coverage by prevalence bin; added incidence ratio, denominator, and the common-bin mechanism diagnosis. R = 50 on both. Synthea 43% overall (57/47/1 by bin), MIMIC 82% (93/81/19). |

---

## 8. What this report does not show

- No production CDM. Synthea is a simulator; MIMIC is a 100-patient ICU demo. The coverage numbers are properties of these fixtures, not of a health system.
- No held-out split. The model is fitted and evaluated on the same patients. A held-out version would separate "reproduces this sample" from "generalises to unseen patients".
- No uncertainty on the model parameters. The simulation interval reflects sampling of synthetic cohorts given a point-estimate fit; it does not include parameter uncertainty.
- No causal claim. The model is a model of how the database records care, not of what causes what.
