# A per-domain temporal model of the OMOP CDM

**Specification for review.** This document states what the CDM actually contains in this workspace, derives a model from that structure, and records what has already been tried and failed. It is written to be criticised by a reviewer with no access to the code.

---

## 1. What the data actually is

A Synthea-generated OMOP CDM 5.4, 10,000 patients, 288,605 person-years. Synthea is a rule-based patient simulator, not observed data, so every result below is a test of whether a model can reproduce *another simulator's* output. Nothing here is evidence about a health system.

### 1.1 The schema audit

The full CDM 5.4 schema is present (39 tables). Populated:

| table | rows | notes |
|---|---|---|
| measurement | 2,142,010 | point events |
| observation | 1,755,183 | point events |
| procedure_occurrence | 673,482 | point events, **44.1% unmapped** |
| visit_occurrence | 474,748 | 91% outpatient, 6.5% ER, 2.5% inpatient |
| drug_exposure | 413,657 | **intervals**, median 365 days |
| condition_occurrence | 295,782 | **intervals**, median 364 days |
| death | 1,040 | absorbing |
| observation_period | 10,000 | one per patient |

**Empty:** `condition_era`, `drug_era`, `dose_era`, `episode`, `episode_event`, `visit_detail`, `fact_relationship`, `note`, `specimen`, `device_exposure`, `cost`, `payer_plan_period`.

### 1.2 Four findings that determine the model

**(a) The schema is visit-centric; the data is not.** Every domain table carries a `visit_occurrence_id` foreign key. In this load it is **0% populated in all five tables**. The visit table exists as an independent point process with its own concepts, and no record can be attributed to a visit except by matching dates.

This invalidates the visit-anchored framing used in earlier work: the "visit" there was a derived `(person, date)` bucket, not the CDM's visit component. That reconstruction is defensible, but it is a *derived* object and should not be presented as the CDM's structure.

**(b) Domains have different temporal types, fixed by the schema.**

| domain | type | evidence |
|---|---|---|
| condition | interval | 100% have end date, median 364 days |
| drug | interval | 100% have end date, median 365 days |
| visit | interval, degenerate | median duration 0 days |
| procedure | point | median duration 0 days |
| measurement | point | no end date column |
| observation | point | no end date column |
| death | absorbing | one row per person |

Collapsing intervals to start dates destroys a year of "active" state per condition and per drug. Earlier work in this repo did exactly that, and every number produced before this document inherits the error.

**(c) Unmapped codes are a domain property, not noise.** `procedure_occurrence` is **44.1%** `concept_id = 0`; condition 2.4%, measurement 2.3%, drug and observation 0%. A generative model must reproduce the null category as a first-class outcome, not discard it.

**(d) Concepts partition cleanly across tables.** All 773 distinct concepts appear in exactly one source table. Per-domain models therefore have disjoint parameter sets.

### 1.3 Why the derived tables being empty matters twice

It removes any possibility of building on `condition_era`, `drug_era` or `episode` — intervals must be modelled from start and end dates directly. But it also creates a concrete deliverable: a generative model that emits intervals can **populate** those tables, and OMOP 5.4's `episode` / `episode_event` structure is the natural schema for the care-episode concept this model would produce.

---

## 2. The model

### 2.1 Layer 1 — per-domain temporal processes

Each domain is a separate process, because each has a different temporal type and a different concept set.

**Interval domains** (condition, drug). A record is a pair `(start, end)`. Generative story: a *start* process decides which concept begins when; a *duration* distribution decides when it ends.

```
starts:   Poisson process with intensity  λ_d(t) · u_i · θ_d[j]  · exp(z_i · β_j)
duration: log D_j ~ Normal(μ_d, s_d²)   per domain (4 parameters each)
```

**Point domains** (measurement, observation, procedure). A record is an instant. Panel and instrument structure is captured by the concept catalogue: a visit draws a *set*, so the catalogue should be sparse and the set size is itself a random variable.

```
events:   Poisson process with intensity  λ_d(t) · u_i · θ_d[j] · exp(z_i · β_j)
```

**Absorbing** (death). A hazard `h_i(t)` that removes the patient from all other processes. 10.4% of patients die in this sample.

**Visit** is treated as its own point/interval domain with three concepts, not as the organising principle.

### 2.2 Layer 2 — patient heterogeneity

One parameter, shared across all domains:

```
u_i ~ Gamma(k)
```

This is not cosmetic. In earlier work, removing it moved the spectral distance from 0.364 to 0.453; it is the single most important term for reproducing the joint structure.

### 2.3 Layer 3 — cross-domain coupling

A low-rank latent patient state `z_i ∈ R^K` modulating every domain through loadings `β_j`:

```
intensity multiplier = exp(z_i · β_j)
```

This is the layer where the interesting topology lives — drug with condition, measurement with condition. It is **recording dependence, not clinical or causal dependence**. A lab drawn because a condition is suspected is a recording fact. The model must not be read causally, and the document that describes it must say so in the same sentence every time.

`z_i` is latent and per-patient, but its distribution (`K` means and a covariance) is small and shared, so it does not add N parameters.

### 2.4 Parameter count

| component | parameters |
|---|---|
| concept catalogues `θ_d` | C (773) |
| domain rates and temporal profiles | O(D × 12) ≈ 84 |
| duration distributions | O(D) with 4 each ≈ 28 |
| coupling `β` | K × C_reduced, or K × D if kept coarse |
| patient heterogeneity | 1 |
| latent state distribution | K + K(K+1)/2 |

**No term in the number of patients.** For K = 20 and coarse coupling this is roughly 20,000 numbers.

### 2.5 What the model emits

A complete set of CDM tables at patient level: five domain tables with correct date semantics, `visit_occurrence`, `death`, `observation_period`, and — because intervals are modelled — `condition_era` and `drug_era`, which are empty in the source. That is a stronger deliverable than a count matrix.

---

## 3. What has already been tried

Recorded so a reviewer can see the search rather than only its endpoint.

**Block model (fixed effects).** `E[X_ij] = T_i u_i λ_j B[g_i, h_j]`, fitted by EM. Reproduces prevalence (r = 0.86), calendar incidence (r = 0.87–0.95), degree distribution. Fails on co-occurrence (r = 0.24) and on the whole-database spectrum (KS 0.413 against a real rank of 81, recovering rank 8).

**Factor term.** Added `exp(a_i · b_j)` on top. Fixes the rank (75) and the spectrum (KS 0.366), but adds a per-patient vector and was fitted in-sample at the time.

**Visit-anchored topic model.** A patient is a sequence of visits; each visit is drawn from one of K types; types follow a Markov chain. `E[X_ij] = Σ_k c_ik θ_kj`. Parameters `K·C + K² + 2K`. Matches the spectrum (KS 0.364 at K = 200 with the Gamma heterogeneity term) with no per-patient parameter. But it inherits the (person, date) visit reconstruction, ignores intervals entirely, and models a single event stream across domains.

**What was learned that matters for the new model:** rank is a property of the parameterisation, not of the partition; a single heavy-tailed patient term does most of the work that per-patient fixed effects did; and the spectral diagnostic is the only one that disagreed with the summary statistics.

---

## 4. Open questions for the reviewer

1. **Is the per-domain decomposition defensible, or does it fragment the object?** Five independent marginals plus a coupling layer may reproduce marginals well and interactions badly. The alternative — one joint model over all concepts — is what failed before.
2. **Is `exp(z_i · β_j)` the right coupling, or does it need concept-level structure?** Coarse coupling (K × D) will not express "drug X with condition Y". Concept-level coupling is C² and needs sparsity or low rank.
3. **How should intervals be estimated when the source is itself panel-observed?** A visit sees a snapshot; the true start may precede it. This is interval censoring and it is not addressed above.
4. **Does death need a competing-risks framing** rather than an independent hazard, given that it truncates 10.4% of records?
5. **Is the unmapped mass (44% of procedures) a category to model, or a separate missingness process?**
6. **What is the identification strategy** for the coupling layer, given that `z` and `β` are only identified up to rotation, and that the coupling can absorb any association in the data?

---

## 5. Reading

Verified against Crossref or arXiv in this session unless marked.

### Intervals, points and panel observation

**Meira-Machado, de Uña-Álvarez, Cadarso-Suárez & Andersen (2009)**, *Multi-state models for the analysis of time-to-event data*, Statistical Methods in Medical Research 18:195–222. [doi:10.1177/0962280208092301](https://doi.org/10.1177/0962280208092301)
The framework for a patient moving between states over time, observed only at discrete times. This is the closest classical literature to the per-domain model.

**Jackson (2011)**, *Multi-state models for panel data: the msm package*, Journal of Statistical Software 38(8). [doi:10.18637/jss.v038.i08](https://doi.org/10.18637/jss.v038.i08)
The worked implementation, including how to handle observation only at clinic visits. Directly applicable given that `visit_occurrence_id` is empty and visits must be reconstructed.

**Fine & Gray (1999)**, *A proportional hazards model for the subdistribution of a competing risk*, JASA 94:496–509. [doi:10.1080/01621459.1999.10474144](https://doi.org/10.1080/01621459.1999.10474144)
For death as an absorbing state that competes with everything else.

### Time in networks

**Butts (2008)**, *A relational event framework for social action*, Sociological Methodology. [doi:10.1111/j.1467-9531.2008.00203.x](https://doi.org/10.1111/j.1467-9531.2008.00203.x)

**Holme & Saramäki (2012)**, *Temporal networks*, Physics Reports 519:97–125. [doi:10.1016/j.physrep.2012.03.001](https://doi.org/10.1016/j.physrep.2012.03.001)

**Hawkes (1971)**, *Spectra of some self-exciting and mutually exciting point processes*, Biometrika 58:83–90. [doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83)
The family that replaces a first-order Markov chain when the process has memory: the interaction kernel is an explicit network over concepts.

**Masuda & Lambiotte (2016)**, *A Guide to Temporal Networks*, World Scientific. [doi:10.1142/q0033](https://doi.org/10.1142/q0033)

### Catalogues and low-rank count structure

**Blei, Ng & Jordan (2003)**, *Latent Dirichlet allocation*, JMLR 3:993–1022.
The concept catalogue is a topic. The Bayesian version supplies the sparsity the point domains need.

**Gopalan, Hofman & Blei (2014)**, *Scalable recommendation with Poisson factorization*, UAI. [arXiv:1311.1704](https://arxiv.org/abs/1311.1704)

**Townes, Hicks, Aryee & Irizarry (2019)**, *Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model*, Genome Biology. [doi:10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6)

### Identifiability and validation

**Peel, Larremore & Clauset (2017)**, *The ground truth about metadata and community detection in networks*, Science Advances. [arXiv:1608.05878](https://arxiv.org/abs/1608.05878)

**Karrer & Newman (2011)**, *Stochastic blockmodels and community structure in networks*, Phys Rev E 83:016107. [arXiv:1008.3926](https://arxiv.org/abs/1008.3926)

**von Luxburg (2007)**, *A tutorial on spectral clustering*, Statistics and Computing. [arXiv:0711.0189](https://arxiv.org/abs/0711.0189)
The spectral comparison is the diagnostic that caught what every summary statistic missed. Its guarantees are worth knowing before relying on it.

### Epidemiology

**Griffith et al. (2020)**, *Collider bias undermines our understanding of COVID-19 disease risk and severity*, Nature Communications. [PMID 33184277](https://pubmed.ncbi.nlm.nih.gov/33184277/)

**Schuemie et al. (2018)**, *Improving reproducibility by using high-throughput observational studies with empirical calibration*, Phil Trans R Soc A. [PMID 30082302](https://pubmed.ncbi.nlm.nih.gov/30082302/)

**Lipsitch, Tchetgen Tchetgen & Cohen (2010)**, *Negative controls: a tool for detecting confounding and bias in observational studies*, Epidemiology. [PMID 20335814](https://pubmed.ncbi.nlm.nih.gov/20335814/)

---

## 6. The three to read first

1. **Meira-Machado et al. (2009)** — because the core problem is now multi-state observation at discrete times, and this is the established framework.
2. **Hawkes (1971)** — because a first-order chain is the simplest possible memory and this is the family that replaces it, with the interaction kernel *being* the network.
3. **Schuemie et al. (2018)** — because none of the epidemiology claims are worth anything until the negative-control experiment is run.
