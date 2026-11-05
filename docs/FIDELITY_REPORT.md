# Fidelity report: prevalence and incidence

*Living document. Version 1, 2026-11-05. Code: `experiments/25_fixes_fidelity.py`. Raw results: `results/fixes_fidelity.json`.*

This is the single place where the model's per-concept **prevalence** and **incidence** fidelity is recorded, together with how those quantities are computed. Update it by re-running the experiment, replacing the numbers and figures below, and adding a line to the change log.

**Data status.** The only datasets here are the Synthea slice (a rule-based patient simulator) and the 100-patient MIMIC-IV demo (real de-identified data, but a small ICU sample). Neither is a production CDM. Nothing below is evidence about a health system.

---

## 1. How the quantities are calculated

### 1.1 Prevalence

For concept `j`, prevalence is the fraction of patients who have **at least one** record of `j` during their observation period:

```
P_j = (1 / N) * sum_i 1[ X_ij > 0 ]
```

where `X_ij` is the number of records of concept `j` for patient `i`, and `N` is the number of patients. It is a **patient-level** quantity: one record is enough, and ten records count the same as one. It answers "how many patients have this concept at all", not "how often is it recorded".

### 1.2 Incidence

For concept `j`, incidence is the total number of records divided by the total person-time at risk:

```
I_j = 1000 * ( sum_i X_ij ) / ( sum_i T_i )
```

where `T_i` is patient `i`'s observation period in years. It is a **record-level** quantity, reported per 1,000 person-years. One patient with 50 records contributes 50, not 1. It answers "how often is this concept recorded per unit of follow-up", not "how many patients have it".

Prevalence and incidence are therefore complementary: a model can get one right and the other wrong. The model here gets the total record counts right and spreads rare concepts across too many patients; that is exactly a prevalence failure with a correct incidence.

### 1.3 Concept set

Both quantities are computed on the mapped clinical concepts — the five clinical domains (condition, drug, measurement, observation, procedure), excluding unmapped (`concept_id = 0`) records. For the ratio summaries, concepts with real prevalence below 1% are excluded, to avoid dividing by a near-zero prevalence. The number compared is reported in the results table.

### 1.4 Synthetic cohorts and the simulation interval

The model is fitted to the real data, then `R` synthetic cohorts are generated. Each cohort has the same number of patients as the real data, with observation periods resampled from the observed `(start, duration)` pairs, so total person-time and the follow-up distribution are preserved.

For each synthetic cohort `r`, the same `P_j` and `I_j` are computed. The **simulation interval** for concept `j` is the 5th to 95th percentile of those `R` values:

```
[ P_j^(5), P_j^(95) ]        [ I_j^(5), I_j^(95) ]
```

`R = 20` for Synthea and `R = 50` for MIMIC.

### 1.5 Predictive coverage

**Predictive coverage** is the fraction of compared concepts for which the real value lies inside the simulation interval:

```
coverage = (1 / |J|) * sum_j 1[ real_j in [lo_j, hi_j] ]
```

where `|J|` is the number of compared concepts. This is the quantity that matters: a high correlation across concepts says the ordering is right, while coverage says the level is right. The two can disagree, and here they do.

### 1.6 Ratio

The ratio panels plot `synthetic median / real`. A value of 1 is perfect; values above 1 mean the model over-predicts. The band around the ratio is the simulation interval divided by the real value.

---

## 2. What is fitted, and on what

| | Synthea | MIMIC-IV demo |
|---|---|---|
| source | rule-based simulator | real de-identified EHR |
| patients in the report | 2,000 (sample of 10,000) | 100 |
| encounters | derived `(person, date)` key | `visit_occurrence_id` |
| model | visit-anchored, `K = 5` | visit-anchored, `K = 5` |
| refill | re-clustered drug courses | re-clustered drug courses |
| synthetic cohorts `R` | 20 | 50 |

The **visit-anchored model** generates a patient as a sequence of encounters: a number of visits, then a set of events inside each visit, then durations for conditions and drugs. It was chosen over the per-domain model because the per-domain generator scatters events across days and produces almost no same-day condition–drug graphs. The derived encounter key is the `(person, date)` proxy; it is the only encounter definition available in Synthea, which has no visit IDs. The refill step re-clusters generated drug records into courses; it preserves record counts, so it does not change prevalence or incidence and is included for completeness.

---

## 3. Results

![Synthea prevalence and incidence](figs/fig39_synthea_fidelity.png)

![MIMIC prevalence and incidence](figs/fig39_mimic_fidelity.png)

| dataset | R | concepts compared | prevalence coverage | incidence coverage | prevalence ratio (median / p90 / max) | incidence ratio (median / p90 / max) |
|---|---:|---:|---:|---:|---|---|
| Synthea (2k) | 20 | 446 of 679 | **40.6%** | **100%** | 1.10 / 9.59 / 31.6 | 0.99 / 1.02 / 1.11 |
| MIMIC-IV demo | 50 | 2,715 of 2,715 | **81.8%** | **100%** | 1.00 / 2.25 / 30.0 | 1.00 / 1.09 / 1.44 |

**Incidence is calibrated.** On both datasets the real incidence lies inside the 5–95% simulation interval for every compared concept, and the ratio is tightly centred on 1 (median 0.99 and 1.00). The model reproduces how often each concept is recorded per unit of follow-up.

**Prevalence is not, for a subset of concepts.** The median prevalence ratio is close to 1 (1.10 and 1.00), but the upper tail is large: p90 of 9.6 on Synthea and 2.3 on MIMIC, max around 30×. The model over-predicts how many patients have mid- and rare concepts. That is why coverage is 40.6% on Synthea and 81.8% on MIMIC even though the median ratio looks acceptable: a minority of concepts is badly over-spread, and coverage counts each concept equally.

The mechanism is the same missing per-patient concentration described in the review's gap (a): the generator draws concepts per event from a global catalogue, with no persistent per-patient concept preference. It gets the total number of records right but spreads a rare concept across too many patients. The same gap blocks the refill fix: most generated patient–concept groups contain one record, so there is nothing for the era builder to merge.

---

## 4. How to update this report

1. Re-run the experiment:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/25_fixes_fidelity.py
```

2. Replace the numbers in the results table from `results/fixes_fidelity.json`:

```python
import json
r = json.load(open("results/fixes_fidelity.json"))
r["synthea"]["prevalence_coverage"], r["synthea"]["incidence_coverage"]
r["mimic"]["prevalence_coverage"], r["mimic"]["incidence_coverage"]
```

3. Rebuild the PDF:

```bash
bash scripts/md2pdf.sh docs/FIDELITY_REPORT.md docs/FIDELITY_REPORT.pdf
```

4. Add a row to the change log and bump the version line at the top.

If the model or the sample changes, also update §2 (what is fitted, and on what) and the figure captions.

---

## 5. Change log

| version | date | change |
|---|---|---|
| 1 | 2026-11-05 | First version. Visit-anchored model with derived encounters and refill; Synthea 2k sample, MIMIC demo; incidence coverage 100%, prevalence coverage 40.6% / 81.8%. |

---

## 6. What this report does not show

- No production CDM. Synthea is a simulator; MIMIC is a 100-patient ICU demo. The coverage numbers are properties of these fixtures, not of a health system.
- No held-out split. The model is fitted and evaluated on the same patients. A held-out version would separate "reproduces this sample" from "generalises to unseen patients".
- No uncertainty on the model parameters. The simulation interval reflects sampling of synthetic cohorts given a point-estimate fit; it does not include parameter uncertainty.
- No causal claim. The model is a model of how the database records care, not of what causes what.
