# MIMIC: intervals, visit anchoring, and predictive checks

*Follow-up to `MIMIC_RUN.md`. The per-domain model reproduced event counts and durations but not the interval structure. This adds the encounter as the unit and evaluates the generators against their own output distribution rather than against one realisation.*

---

## 0. The sampling question

The model defines a distribution over datasets given N and the observation periods. The expected count per cell is a deterministic function of the parameters, but a realised dataset is a draw. You cannot "generate the expected counts" and call it data: that produces non-integer cells, removes the sampling variance, and makes every level check look exact by construction. The one case where the total is fixed is the fixed-density conditional model — condition on `M`, and the total is a parameter, but the allocation is still sampled. The current generator does not condition on `M`; it calibrates the expected level and samples.

So the right test is predictive, not a single comparison. Draw R datasets from the model, compute the same statistic on each, and ask where the real database falls. One draw being above or below the data is not evidence against the model; the real value being outside the model's output distribution is.

This experiment uses R = 50.

---

## 1. The interval diagnosis

On the MIMIC demo, for consecutive events within a patient:

| gap | fraction |
|---|---:|
| 0 days | **99.4%** |
| 0 hours | 89.7% |
| < 1 hour | 97.4% |
| < 24 hours | 99.8% |

And `visit_occurrence_id` is **100% populated** in every clinical domain (0% in the Synthea slice). The events are not a continuous-time stream; they are recorded in encounters. The per-domain model draws starts independently in time, so its consecutive-gap distribution is approximately geometric, not a spike at zero. That is the interval failure.

---

## 2. The visit-anchored model

`topocdm/visitanchor.py`. The encounter is the unit:

| component | estimate |
|---|---|
| visits per patient | empirical distribution, bootstrap |
| visit dates | uniform within the resampled observation period |
| events per visit, by domain | empirical distribution, bootstrap |
| concept given domain | empirical catalogue |
| durations | per-domain quantile model (as before) |
| death | empirical fraction and within-window timing |

All events in a visit share the visit date. That is the entire interval fix: it reproduces same-day batching and therefore era merging, without assuming a parametric gap law. The visit domain is the encounter itself, with a visit type drawn from the observed distribution. As everywhere else, the shapes are estimated from the fitted database and are not claims that transfer.

---

## 3. Predictive check

Real MIMIC value against 50 generated cohorts; 5–95% interval; `in` means the real value is inside.

| statistic | MIMIC | per-domain | visit-anchor |
|---|---:|---|---|
| condition events | 16,441 | 15,347 [11,004–21,276] in | 16,899 [11,653–22,732] in |
| drug events | 18,229 | 17,308 [12,101–24,162] in | 17,841 [14,567–22,700] in |
| measurement events | 338,550 | 312,790 [239,533–430,563] in | 330,618 [249,344–434,808] in |
| same-day gap fraction | **0.994** | 0.89 [0.86–0.91] **out** | 0.999 [0.998–0.999] **out** |
| same-concept merge, condition | 0.932 | 0.81 [0.76–0.83] **out** | 0.96 [0.934–0.979] **out** |
| same-concept merge, drug | 0.816 | 0.34 [0.26–0.45] **out** | 0.81 [0.71–0.90] in |
| condition eras | 3,771 | 4,950 [3,603–6,383] in (8th pct) | 3,912 [2,828–4,800] in |
| drug eras | 7,931 | 12,694 [9,562–17,175] **out** | 9,398 [7,474–11,454] in |
| distinct concepts per patient | 271.3 | 203.1 [161.3–250.5] **out** | 270.8 [242.5–305.6] in |
| effective rank | 61 | 20.5 [14.5–29] **out** | 37 [26–45] **out** |

The visit anchor is inside the predictive interval for 7 of 10 statistics; the per-domain model for 4 of 10 (and the condition-era placement is at the 8th percentile, i.e. the real value is at the edge). The three remaining misses are of different kinds:

- **Same-day gap 0.999 against 0.994.** The visit anchor over-clusters slightly: every within-visit gap is exactly zero, while the real data has 0.6% of consecutive pairs on different days. The model's within-visit counts are a little too large relative to its between-visit gaps. This is a small calibration miss, not a structural one.
- **Condition merge 0.96 against 0.932.** The same effect, just outside the lower bound. Drug merge is inside (0.81 against 0.816).
- **Effective rank 37 against 61.** This is the genuine remaining gap. The visit anchor fixes *when* events happen but not *which concepts co-occur inside a visit*: the per-domain counts and concepts are drawn independently. Reproducing the rank requires a within-visit concept dependence — a topic/coupling layer on the visit catalogue — which is the next model, not a tweak to this one.

---

## 4. What changed, and what it means

The per-domain model's failures were mostly interval failures: it could not produce same-day batching, so its eras were too many and too short, and its degree distribution too light. Anchoring events to visits fixes those. The drug-era over-count falls from 1.60× to 1.18× and moves inside the predictive interval; the degree mean moves from 203 to 271 against a real 271.

The rank is a separate object. It is about the joint distribution of concepts within an encounter, not the timing of encounters. That is the next thing to fit, and it should be fitted on the within-visit co-occurrence matrix rather than on the marginal concept frequencies.

![Predictive check](figs/fig30_mimic_predictive.png)
