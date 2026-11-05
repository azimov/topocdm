# Where this actually stands

*Written for someone with a network-science PhD and an epidemiology day job, who does not want another lecture on latent variable models.*

---

## 1. Your five claims, graded

I've been building and reporting things without ever holding these up against each other. Here they are, each with what it means operationally, where it actually stands, and what would settle it.

### Claim 1 — topology can model a dataset in the CDM

**Operationally:** fit the model to a CDM and show it reproduces that CDM's structure on data it wasn't fitted to.

**Status: partly done.** On the Synthea slice, held-out patients: prevalence r = 0.86, incidence by calendar year r = 0.87–0.95, degree distribution close. But co-occurrence only r = 0.24, and the whole-database spectrum is wrong in shape (though the rank can now be fixed).

**Not done:** no real CDM. Everything is against Synthea, which is itself a simulator. So the strongest true statement is *"the pipeline works end to end on a CDM-shaped database, and the model fits one generator's output moderately well."*

**What would settle it:** the same three numbers on a real database, plus a check of what fails.

### Claim 2 — the structure uncovers systematic bias

**Operationally:** the model has an explicit parameter for each patient's contact volume (`u_i`). If you adjust associations for it, spurious ones should disappear and real ones should survive.

**Status: demonstrated on synthetic data only.** On the Synthea slice, the median co-occurrence odds ratio across 920 concept pairs is 1.71 raw and 0.91 after stratifying by propensity; 46% of pairs lose more than half their association. That is the mechanism working.

**The worry:** conditioning on total degree can itself induce associations (collider bias). The estimator may be manufacturing part of the effect it reports. This is the biggest open methodological question and it is question 5 in the review list.

**What would settle it:** a negative-control test on data where you know the answer — pairs that must be null.

### Claim 3 — calibrate incidence and prevalence estimates

**Operationally:** use the fitted model to compute what a rate *would* be under the observation process alone, and compare the observed rate to that.

**Status: machinery built, nothing demonstrated.** There is an analytic predictive interval and a calibration module. No evidence that adjusting a real estimate by this method improves it.

**This is the weakest claim and the one you'd most want to be true.** It is also the one where "novel methodology" is plausible.

**What would settle it:** take a rate with an external benchmark (a chart-reviewed incidence, or a replicated estimate from a second database). Show the calibrated estimate is closer than the raw one.

### Claim 4 — synthetic data with realistic properties

**Operationally:** generate a synthetic cohort and show it passes a pre-specified list of checks.

**Status: partly done, and the checks are the problem.** Currently passes prevalence and calendar incidence, fails the spectral shape and co-occurrence.

**What would settle it:** a fixed panel — prevalence, incidence by year, degree distribution, co-occurrence, timing — declared in advance, with pass/fail. Right now the goalposts move with each experiment, which is the failure mode to avoid.

### Claim 5 — small, shareable, privacy-preserving, comparable

**Operationally:** the fitted artefact contains no per-patient parameters; it is small; two artefacts can be compared.

**Status: the most solid of the five.** 2,056 numbers for a 10,000-patient database (16 KB). Parameter count is `2C + G·H + G + 1` and does not grow with patient count. There is genuinely no code path from a release back to a patient. The GDPR argument is structural rather than statistical, which is unusual and defensible.

**Not done:** the comparison application. Two time cuts, or two databases, and a decomposition of why rates differ. That's the piece that would make this feel like a contribution rather than a toolkit.

---

## 2. The model, in the language you already use

### The data is a bipartite graph

Patients on one side, concepts on the other. An edge means "this patient has at least one record of this concept". Weight = number of records. Two node types, no edges within a type.

This is the object. Everything else is a projection of it, including the concept–concept co-occurrence network that disease-network papers build.

### The model is a random graph model with a likelihood

Your CiGRAM assigned each node a position on a circle and connected nodes with probability depending on geometry. It could be *sampled* but not *scored*: given a graph, there was no number saying how likely it was.

This model does the same job but in a form you can score. For each patient–concept pair, the expected number of records is a product of four terms:

```
expected = time at risk × patient's contact volume × concept's frequency × group affinity
```

- **contact volume** — one number per patient. How many records they generate.
- **concept frequency** — one number per concept. How often it appears.
- **group affinity** — patients are sorted into ~15 groups, concepts into ~18 groups, and this says which patient groups get which concept groups.

That's the whole model. Three parameter sets, and the bipartite graph falls out of them.

**The one idea worth taking away:** because it is written as a product of per-pair terms, the probability of the whole database factorises into one term per pair. That is what makes it scorable. Your thesis model could not do this because edges were sampled *without replacement*, which couples every edge to every other.

### How the fitting works

You don't know which group each patient belongs to. That's the only hard part.

- If you knew the groups, fitting the rest is counting: patient's records ÷ expected records.
- If you knew the rest, guessing the group is a comparison: which group makes this patient's record most likely?

So you alternate. Guess groups from current parameters, refit parameters from guessed groups, repeat. It converges in about fifteen passes and never gets worse, because each step is the exact best answer given the other.

That's it. The word for it is EM, but the word isn't the idea.

### Three things that can go wrong, and do

**Identifiability.** Different parameter values can produce the same distribution over graphs. If so, the parameters aren't facts about the world, they're one arbitrary labelling among many. We pin down two such freedoms with constraints. There may be others we haven't found — that's a live question for a referee.

**Over-dispersion.** Poisson says variance equals mean. Real counts vary more. If you fit a distribution to point estimates you add estimation noise back in as if it were real heterogeneity. This bit me twice in this project.

**Reproduction is not explanation.** A free multiplier per time period will reproduce a pandemic, a coding change, and an ETL bug identically. If a parameter is to mean anything outside the sample, it has to be constrained.

---

## 3. What you actually need to learn

Six things. Each is small, and each maps to a specific claim you need to defend.

### (a) Likelihood as a scoring rule

**Why:** it's the difference between your thesis model and this one, and it underpins claims 1–4.

**The idea:** a model that assigns a probability to a dataset can be compared, tested, and rejected. A model that can only be sampled cannot. Nothing more complicated than that.

**Test yourself:** can you explain why a model that samples edges without replacement has no closed-form likelihood, and why a CDM (where a patient can be coded repeatedly) does?

### (b) Latent variables and alternating fitting

**Why:** the group structure is latent; this is the core algorithm.

**Test yourself:** write the `G=1` case in thirty lines of NumPy — no groups, just patient volume and concept frequency, alternating updates. If you can write it, you understand the model. This is the single highest-value exercise on the list, and it's an afternoon.

### (c) Identifiability

**Why:** it bounds claims 1 and 2. Peel et al. (2017) proved no algorithm uniquely solves community detection, so "the model found five groups" is always a statement about a model, not about the world.

**Test yourself:** given a fitted model, can you say which of its parameters are determined by the data and which are conventions?

### (d) Model comparison

**Why:** choosing the number of groups, the number of factors, whether the extra complexity earns its keep.

**Test yourself:** can you explain why the likelihood you fit with is not always the likelihood you should compare with? (In this project, using the fitting likelihood for model selection picked 8 groups when the truth was 4.)

### (e) Null models

**Why:** you already know this from modularity — the null model *is* the hypothesis. This model's contribution is that it gives you a null for the observation process: generate the database with no associations and see what remains.

**Test yourself:** state, for claim 2, exactly which null you are testing against, and what a significant departure means.

### (f) Ascertainment and collider bias

**Why:** claim 2 rests on it. Griffith et al. (2020) is the clean statement: conditioning on healthcare contact induces associations among its causes.

**Test yourself:** the ingrown toenail. A patient presents for something else, the toenail is coded. Which part of that does a propensity parameter capture, and which part does it miss?

---

## 4. The one experiment I'd run before investing further

Claims 2 and 3 are the interesting ones and neither has evidence. Both can be tested with one design, on real data, cheaply:

**Take a negative control pair** — a drug and an outcome with no plausible causal relationship, in a real CDM. Estimate the association three ways:

1. raw co-occurrence;
2. adjusted for patient contact volume (the `u_i` route);
3. the standard OHDSI route: empirical calibration against many negative controls.

If (2) and (3) agree, the model's bias parameter is measuring the same thing as the field's established method, using 2,000 numbers instead of the whole database. That's a contribution.

If they disagree, one of them is wrong, and finding out which is the paper.

**Kill criterion:** if adjusting for `u_i` doesn't move a known-null association toward the null, claim 2 is dead and claims 3 and 4 go with it. Everything else — the fitting, the scaling, the privacy story — is engineering around that one result.

---

## 5. On the confusion

The last several exchanges produced a lot of machinery — factor models, visit chains, spectral diagnostics — and each was justified by a real failure, but they accumulated faster than they could be assessed. That's my doing.

The state of things is simpler than the sequence suggests: the model fits, it captures the marginal structure and not yet the interaction structure, it produces a genuinely small shareable object, and its most interesting claim (bias) has been demonstrated on simulated data and nothing else.

The next thing that matters is the negative-control experiment in §4. Everything else is optional until that's answered.
