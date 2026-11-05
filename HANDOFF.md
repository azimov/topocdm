# HANDOFF — read this first

*Condensed context for an agent starting this workspace fresh. Written 2026-11-05. The reviewable specification is [`docs/MODEL_V2.md`](docs/MODEL_V2.md) (and `docs/MODEL_V2.pdf`); read that after this.*

---

## 1. What this project is

The author wrote **CiGRAM** (github.com/azimov/Cigram) for a 2015 PhD: a generative multiplex network model, fitted by spectral matching. He now works in observational epidemiology on the OMOP CDM. The project asks whether the network-topology idea can be brought into the CDM — as a model of how a database *records* care, and as a generator of realistic synthetic cohorts.

**Two objectives, stated by him:** (1) model the network topology of a CDM, (2) generate synthetic data that looks real. Causality is explicitly **out of scope**. Residual bias is claimed as a *consequence*, not a goal.

**Five claims the work is judged against** (graded in `WHERE_THIS_STANDS.md`):

1. Topology can model a dataset in the CDM.
2. The structure can uncover systematic bias.
3. This can calibrate incidence and prevalence estimates.
4. The model can generate synthetic data with realistic properties.
5. The model is small, shareable, privacy-preserving, and comparable across sources and time.

**Claim 5 is the strongest and nearest to done. Claims 2 and 3 have no evidence and are the ones that matter.** The decisive experiment is a negative-control test on a production database (§8).

---

## 2. Ground rules

- **The data is Synthea.** Synthea is a rule-based patient simulator. Never describe the data as observed, production or patient-derived. The author has corrected this once already and it was committed into artefacts; do not repeat it.
- **Do not overclaim.** Every earlier overclaim in this repo traced back to a bookkeeping error, not a modelling error. Check the level (generated totals vs target) before interpreting a result.
- **Do not select a hyperparameter on the criterion you then report.** Several results here were fitted that way and are flagged as such.
- **Verify citations** against arXiv, Crossref or PubMed before writing them. One DOI guess produced a paper on concrete creep.

---

## 3. Environment

```bash
cd /Users/jamie/PycharmProjects/networks_research
MPLCONFIGDIR=./.mplconfig .venv/bin/python ...      # always set MPLCONFIGDIR
```

- Python: `.venv/bin/python` (Homebrew 3.11). The system DSH python **cannot** load the signed `.so` for the C kernel; do not use it for this repo.
- Database: `docker exec ohdsi-lab-postgres psql -U postgres -d postgres`. Schemas `cdm` and `vocab`. No `psql` on the host.
- No LibreOffice binary, but `office-export-pdf` skill provides LibreOffice Kit. PDFs are built pandoc → docx (styled by `/tmp/style2.py`) → document.
- No Rust toolchain. `gcc` and `clang` present.
- Connection card: `.secrets/CONNECTIONS.md` (gitignored).

### Data (`data/`, gitignored)

| file | rows | note |
|---|---|---|
| `events_dated.csv` | 4,925,516 | person_id, concept_id, event_date — **start dates only** |
| `events.csv` | 750,053 | person_id, concept_id, n (aggregated) |
| `visits.csv` | 474,748 | person_id, visit_date |
| `periods.csv` | 10,000 | person_id, start_date, end_date |
| `concepts.csv` | 643 MB | full vocab.concept |
| `person.csv` | 10,000 | demographics |

**Known defect:** `events_dated.csv` carries only start dates. Condition and drug are intervals (median 364 and 365 days) and that structure is lost. Re-extraction with end dates is the first task.

---

## 4. What the CDM actually contains

Audited this session; full detail in `docs/MODEL_V2.md` §1.

| table | rows | type |
|---|---|---|
| measurement | 2,142,010 | point, 2.3% unmapped |
| observation | 1,755,183 | point, 0% unmapped |
| procedure_occurrence | 673,482 | point, **44.1% unmapped** |
| visit_occurrence | 474,748 | 91% outpatient / 6.5% ER / 2.5% inpatient |
| drug_exposure | 413,657 | **interval**, median 365 d |
| condition_occurrence | 295,782 | **interval**, median 364 d |
| death | 1,040 | absorbing (10.4% of patients) |

**Four facts that drive the design:**

1. **`visit_occurrence_id` is 0% populated in all five domain tables.** The schema is visit-centric; the data is not. An earlier model used a `(person, date)` proxy and called it a visit — that was a derived object, not the CDM's visit component.
2. **Domains have different temporal types** (interval vs point vs absorbing), fixed by the schema.
3. **Unmapped codes are a domain property**: procedures are 44% null.
4. **Concepts partition cleanly** — all 773 appear in exactly one source table.

**Empty tables:** `condition_era`, `drug_era`, `dose_era`, `episode`, `episode_event`, `visit_detail`, `fact_relationship`, `note`, `specimen`, `device_exposure`, `cost`, `payer_plan_period`.

**Trap:** `vocab.concept.domain_id` does **not** match the table a record lives in ("Full-time employment" is in `condition_occurrence`). An earlier analysis reported conditions as 1.4% of records; by source table it is **5.9%**. Always group by source table.

---

## 5. Code layout

```
topocdm/
  model.py        CDMIncidence, Params, mean_matrix, loglik_*, aic/bic   [block model]
  fit.py          fit_ipf (soft EM), fit_propensity_nb
  scalable.py     SparseCDM, fit_streaming, array_chunks, fit_propensity_streaming
  distributed.py  Stats, partition_stats, merge, m_step  (map/reduce, reproduces sequential)
  sql.py          plan(), CheckpointExecutor  (portable SQL, content-addressed checkpoints)
  release.py      Release, make_release, sample_release  [the shareable artefact]
  omop.py         generate_omop_tables, write_omop_tables
  temporal.py     year/month exposure, fit_temporal, generate_dated, seasonal_residual
  visits.py       VisitModel, PresenceModel, assign_with_presence
  factors.py      FactorFit, fit_factors  [rank-K interaction, damped Newton]
  visitmodel.py   VisitTopicModel, fit_topics (PLSA/EM), fit_transitions, simulate_visits
  perdomain.py    NEW: per-domain temporal model (MODEL_V2 §2); fit/generate, era builder,
                  spectrum; sample-scalable (max_patients, sparse IPF)
  fast.py + _fast.c  CSR mixture likelihood via ctypes, 21x faster
  cohort_incidence.py, io.py
scripts/extract_dated.sh  NEW: events with end dates -> data/events_full.csv
experiments/01..16  earlier analyses; 17_perdomain.py, 18_negative_control.py NEW
tests/test_model.py (15) + tests/test_perdomain.py (7), 22 passing
figs/, results/
docs/MODEL_V2.md, docs/RESULTS_PERDOMAIN.md (NEW), WHERE_THIS_STANDS.md,
BRIEFING.md, MODEL_AND_READING.md, docs/*.md
```

Repo: **github.com/azimov/topocdm** (private, MIT). PDFs are gitignored.

---

## 6. The three models, and where each stands

**A. Block model (fixed effects).** `E[X_ij] = T_i u_i λ_j B[g_i,h_j]`. EM with closed-form M-steps; IPF when G=1. Reproduces prevalence (r=0.86), calendar incidence (r=0.87–0.95), degree distribution. Fails co-occurrence (r=0.24) and the whole-database spectrum (KS 0.413, rank 8 against a real rank of 81). Fits one propensity **per patient** — untenable at scale.

**B. Factor term.** `exp(a_i·b_j)` added to A. Rank 75, KS 0.366. Fixes the spectrum but adds a per-patient vector.

**C. Visit-anchored topic model.** Patient = sequence of visits; visit drawn from one of K types; types follow a Markov chain. `E[X_ij] = Σ_k c_ik θ_kj`. Parameters `K·C + K² + 2K + 1`, **no term in N**. With K=200 and `u_i ~ Gamma(0.15)`: KS **0.364**, rank 53 — matching B's spectrum with no per-patient parameter. Crossover in fitting cost at ~182,000 patients. But it uses the (person,date) visit proxy and ignores intervals.

**Selected numbers**

| quantity | value |
|---|---|
| real spectral rank (>0.1) | 81 |
| block model | KS 0.413, rank 8 |
| block + factors | KS 0.366, rank 75 |
| visit model K=200 | KS 0.453, rank 57 |
| visit model + Gamma(0.15) | **KS 0.364, rank 53** |
| release size | 2,056 numbers (16 KB) for 10k patients |
| raw→adjusted median OR | 1.71 → 0.91 (46% of pairs lose >half) |
| presence-transition r | 0.91 (vs 0.62 for the modal-group metric) |

---

## 7. Traps that have already cost time

1. **`np.searchsorted` for year mapping** silently assigned all pre-window events to index 0, producing a fake "tau spike". Use exact year matching.
2. **Classification vs mixture likelihood**: using the fitting likelihood for model selection picked G=8 when the truth was 4.
3. **`visit_occurrence_id` is null** — do not join on it.
4. **`vocab.concept.domain_id` ≠ source table**.
5. **`mean(u)=1` is unweighted**; the person-time-weighted mean of the fitted `u` is 0.60, so generating with mean-1 propensities over-predicts events by 1.67×. `Release.propensity_tmean` now handles this.
6. **Poisson NMF KL updates are not the Euclidean ones.** `W *= XHᵀ/(1Hᵀ)` diverges; the KL form needs `X/(WH)`. Use the aspect-model EM instead.
7. **`scipy.cluster.vq.kmeans2` is broken** in this build (both inits fail). Write k-means by hand.
8. **`np.asarray` on a sparse matrix** raises; check `sparse.issparse` first.
9. **`pandas.read_csv(parse_dates=...)`** requires the exact column name (`visits.csv` has `visit_date`, not `visit_start_date`).
10. **The per-patient visit count** must be drawn per patient; drawing one value for the cohort collapses the rank.

---

## 8. State after the per-domain session, and what to do next

The four items that were open are done; the results are in `docs/RESULTS_PERDOMAIN.md` (PDF built).

1. **Re-extraction with end dates is done.** `scripts/extract_dated.sh` produces `data/events_full.csv` (5,755,902 records, seven domains, 100% end-date coverage on condition/drug/visit). Unmapped records are kept as per-domain atoms.
2. **The per-domain model is fitted and generates.** `topocdm/perdomain.py`; experiment `17_perdomain.py`. Levels 0.94–1.05 on every domain; prevalence r 0.95, incidence r 1.00, co-occurrence r 0.64; spectrum KS 0.35, rank 31 against 81. In-sample. Two fixture findings (not claims about a production CDM): the duration density is estimated (a zero-inflated empirical quantile function; log-normal and mixture fail on Synthea), and death timing is estimated nonparametrically (Weibull fails). Re-run both diagnostics on a production CDM; do not reuse the shapes.
3. **The negative-control experiment has an answer.** `18_negative_control.py`. Adult drug–outcome pairs: raw OR 1.01, contact-adjusted 0.96, calibration μ 0.02 — all null. All-ages condition pairs: raw 1.46 → contact 1.05, but age 1.06. **The bias `u_i` removes is age.** Claim 2 survives weakly, not strongly. Read `docs/RESULTS_PERDOMAIN.md` §3 before touching the epidemiology.
4. **The empty tables are populated, with a caveat.** condition_era 179,121 rows (source-derived 214,544); drug_era 208,028 (source-derived 57,710 — 3.6× too many, because repeat-prescription continuity is not modelled).

**Kill criterion:** not triggered as literally stated (contact adjustment does move the all-ages association, 1.46 → 1.05), but the contribution is not established because age adjustment does the same (1.06). The decisive next test is a production CDM and age-matched controls. See `docs/RESULTS_PERDOMAIN.md` §6.

**Scalability:** `fit_perdomain(..., max_patients=...)` fits a sample; the marginal fit is sparse IPF and the coupling uses at most `max_factor_patients`. 2,000 patients in 2.8 s, 10,000 in 10.3 s, parameters stable.

---

## 9. Working style the author expects

He has a 2015 network-science PhD and does not want statistics lectures. Explain in network terms — nodes, edges, degree, null models, spectra. He pushes back hard and correctly; he caught the observed-data framing and the visit-weighting problem. He wants honest grading of claims, not encouragement. He asked for PDFs he can read and hand to a reviewer, so keep documents self-contained and readable.
