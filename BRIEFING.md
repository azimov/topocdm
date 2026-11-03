# topocdm — technical briefing for external review

*A self-contained summary of the model, what was fitted, what was validated, and what is known to be wrong. Written so a reviewer with no access to the codebase can criticise it.*

---

## 1. The problem

Given an OMOP CDM, fit a **generative model of its topology** that (a) can be scored — not merely simulated — and (b) produces synthetic patient data that reproduces the real database's structure. The intended use is twofold: synthetic data for methods development without patient-level disclosure, and a *null* against which observed rates can be judged ("what would this database look like if nothing caused anything?").

Data are reduced to an incidence matrix `X` of shape `N × C`: counts of concept `j` for patient `i`, with person-time `T_i`. Synthea 10k slice: `N=10,000`, `C=773`, 4.93M events, 288,605 person-years.

## 2. The model

```
E[X_ij] = T_i · u_i · λ_j · B[g_i, h_j]

u_i    patient care-seeking propensity (free, one per patient)
λ_j    concept intensity
B[g,h] affinity between patient group g and concept group h
g_i    latent patient group (G groups), h_j observed concept group (H groups)
```

Counts are Poisson. **Fixed density is imposed by conditioning on the total `M = Σ X`**, giving a multinomial. Two models differing only in scale have the same conditional MLE, so the Poisson MLE *is* the fixed-density MLE; generation conditions on `M`. The CDM is a multigraph (a patient can be coded repeatedly), so there is no without-replacement constraint — this is what makes the conditional likelihood closed-form, and it is the specific obstacle that made the author's earlier model (CiGRAM, PhD thesis 2015) unsco­reable.

**Fitted parameters:** `N−1 + C + G(H−1) + (G−1)`. Dominated by `u`.
**Released parameters:** `2C + GH + G + 1` — no per-patient terms. 1,832 numbers for this CDM (16 KB); independent of `N`.

## 3. Fitting

Closed-form coordinate ascent. For fixed `(u, λ, B)` the responsibilities are computed per patient; the M-step updates are ratios of observed to expected. With `G=1` this is exactly iterative proportional fitting; verified against a general-purpose optimiser to 1e-6.

Identifiability constraints, enforced by renormalisation with the compensating factor pushed into `λ` so the mean matrix is unchanged:

```
mean(u) = 1
Σ_g w_g B[g,h] / Σ_g w_g = 1      for each concept group h,  w_g = Σ_{i∈g} T_i u_i
```

Model selection uses the **mixture** likelihood `Σ_i log Σ_g π_g Π_j Poisson(x_ij | μ_ijg)` with `N−1 + C + G(H−1) + (G−1)` parameters. The classification likelihood (hard assignment) is what EM computes internally and must *not* be used for AIC: it silently treats every patient's group label as a free parameter that never appears in the count, and AIC then marches to `G=8` when the truth is 4.

## 4. Engineering that constrains the statistics

- **Streaming fit.** Global parameters depend on the data only through `colsum_j`, `n_ih` (events per patient per concept group), `A_g`, `π_num`. Nothing indexed by both patient and concept. Fits in `O(chunk·(H+G) + C)` memory: 200k patients × 1,000 concepts × 8M events in ~8 s at 0.65 GB.
- **Map/reduce form.** The reduced form reproduces the sequential fit bit-for-bit (`λ` and `B` differ by 0.0e+00). Shuffle is one record per partition.
- **C kernel** for the CSR mixture likelihood: 21× faster than NumPy, agreeing to 1e-15. The subtlety: a zero cell still contributes `−μ`, but CSR does not store zeros, so the total `μ` is subtracted in closed form as `T_i u_i (BΛ)_g`. Getting this wrong silently inflates the likelihood.
- **Propensity distribution** for generating *new* patients is fitted by marginal (negative binomial) likelihood, not by plugging in the noisy `u_i`. Plug-in over-disperses: Gini 0.79 against a target of 0.53.

## 5. What was validated

**Synthetic with known truth.** Mixture-AIC recovers the true `G`; latent groups recovered at ~81% accuracy; mean structure correlates with truth at r = 0.96.

**Held-out patients, real CDM** (fit on 75%, predict 25%):

| model | held-out log-lik/event | degree L1 | co-occurrence r |
|---|---|---|---|
| topology | **−0.708** | 0.89 | 0.24 |
| marginals only | −0.787 | 1.07 | 0.18 |

**Calendar-year incidence, 9 clinical conditions, 2016–2025:** log-log r = 0.874, median ratio 0.97.

**Timing.** 91% of gaps between consecutive events are under a day; a constant-rate model predicts 4%. A visit layer (per-group two-state presence chains, `2H` parameters) reproduces presence transitions at r = 0.91 off-diagonal (vs 0.62 for a simpler generator).

**The bias argument.** Across 920 concept pairs, the median raw co-occurrence odds ratio is 1.71; the same odds ratios computed within patient-propensity strata have median 0.91. 46% of pairs lose more than half their association once propensity is fixed. This is the claim that the topology model captures something the projection does not.

## 6. Known weaknesses

1. **Under-dispersion.** Synthetic per-patient event counts have CV 1.1 against 1.34 observed. Concept rates are compressed: low-rate concepts over-predicted, high-rate under-predicted (log-log slope < 1).
2. **Co-occurrence only partly reproduced** (r = 0.24). Degree distribution and marginals are good; pairwise structure is not.
3. **No calendar shocks.** COVID-19 sits flat at ~13/1000 PY while observed spikes to 91 then returns to zero. The seasonal term `τ(h, month)` can reproduce it in-sample but does not explain it, and will equally absorb artefacts.
4. **No channel structure.** Ascertainment is modelled as one implicit channel; exam vs laboratory vs prescription is not distinguished.
5. **No parameter uncertainty.** Point estimates only; predictive intervals are computed analytically from the fit but the covariance of `(λ, B, τ)` is not propagated.
6. **Seasonal term not yet wired into generation.**
7. **Nothing has been fitted to a real CDM.** All results are on a Synthea slice.

## 7. Questions for the reviewer

1. Is the conditioning-on-total argument sound as a justification for calling the Poisson MLE the fixed-density MLE, given that generation conditions on `M`?
2. Is mixture-likelihood AIC defensible when the fit itself uses EM with a hard-assignment warm start and a block structure that is only locally optimal?
3. Is `B` identified under free per-patient `u_i`, beyond the stated constraints? The claim is that row *shapes* identify `B` once `λ` is fixed by the column constraint. Is that sufficient?
4. Is the topology-vs-marginals comparison fair, given that the marginal model is not given an over-dispersion parameter of its own?
5. Does the propensity-stratified odds ratio actually identify "bias", or does conditioning on total degree induce collider bias of its own? If the latter, what is the right estimator?
6. Is the residual-bias claim defensible: that a model reproducing the topology must reproduce the observation process, and that its residual is therefore candidate signal?

---

# Reading list (10)

Verified against arXiv/PubMed. Links resolve.

**The modern statistical framework your model belongs to**

1. **Karrer & Newman (2011)**, *Stochastic blockmodels and community structure in networks*, Phys Rev E 83:016107. [arXiv:1008.3926](https://arxiv.org/abs/1008.3926)
   Degree-corrected SBM. This is the workhorse your model is adjacent to, and the reason "how many edges" and "who to" are separable parameters. Read for the parameterisation, not the communities.

2. **Peel, Larremore & Clauset (2017)**, *The ground truth about metadata and community detection in networks*, Sci Adv. [arXiv:1608.05878](https://arxiv.org/abs/1608.05878)
   Proves no algorithm uniquely solves community detection; metadata are not ground truth. This bounds what any cluster result can claim — including yours.

3. **Pamfil, Howison & Lambiotte (2018)**, *Relating modularity maximization and stochastic block models in multilayer networks*. [arXiv:1804.01964](https://arxiv.org/abs/1804.01964)
   Modularity is MAP inference under a particular SBM. Your thesis's central critique of modularity, formalised — and it tells you what your module definition was actually assuming.

4. **Bazzi, Jeub, Arenas, Howison & Porter (2020)**, *A framework for the construction of generative models for mesoscale structure in multilayer networks*, Phys Rev Research. [arXiv:1608.06196](https://arxiv.org/abs/1608.06196)
   The direct descendant of what CiGRAM was trying to be: generative benchmark models with explicit, user-specified structure. Compare their design decisions to yours.

**Multiplex — the two results that killed the naive version**

5. **Taylor, Shai, Stanley et al. (2016)**, *Enhanced detectability of community structure in multilayer networks through layer aggregation*, PRL 116:228301. [arXiv:1511.05271](https://arxiv.org/abs/1511.05271)
   Summing layers is a very strong baseline; the detectability limit vanishes as `O(L^{-1/2})`. Any multiplex method must beat "add the matrices", and often doesn't. Your own experiment reproduced this.

6. **De Domenico, Nicosia, Arenas & Latora (2015)**, *Structural reducibility of multilayer networks*, Nat Commun 6:6864. [doi:10.1038/ncomms7864](https://doi.org/10.1038/ncomms7864)
   Most empirical multilayer networks collapse to far fewer layers with negligible loss. Read before adding any layer.

**The belief that changed while you were away**

7. **Broido & Clauset (2019)**, *Scale-free networks are rare*, Nat Commun 10:1017. [doi:10.1038/s41467-019-08746-5](https://doi.org/10.1038/s41467-019-08746-5)
   Across ~1,000 networks, strongly scale-free structure is rare; log-normal fits most. The benchmark CiGRAM was measured against assumed power laws everywhere.

**The epidemiological half**

8. **Monchka, Leung, Cordeiro et al. (2022)**, *The effect of disease co-occurrence measurement on multimorbidity networks*, BMC Med Res Methodol 22:165. [PMID 35676621](https://pubmed.ncbi.nlm.nih.gov/35676621/)
   Seven networks from the same data using different co-occurrence measures; different clusters and hubs. The edge is a choice, not a measurement.

9. **Griffith, Morris, Tudball et al. (2020)**, *Collider bias undermines our understanding of COVID-19 disease risk and severity*, Nat Commun. [PMID 33184277](https://pubmed.ncbi.nlm.nih.gov/33184277/)
   Conditioning on healthcare contact induces associations among its causes. This is the mechanism behind every "disease network" artefact, stated generally.

10. **Schuemie, Ryan, Hripcsak et al. (2018)**, *Improving reproducibility by using high-throughput observational studies with empirical calibration*, Phil Trans R Soc A. [PMID 30082302](https://pubmed.ncbi.nlm.nih.gov/30082302/)
    Negative controls and empirical calibration — the standard your bias argument has to beat, and the reason "we model the observation process" must be distinguished from "we calibrated against controls".

*Optional extras if the above is too network-heavy: Kleineberg, Boguñá & Serrano (2016) on geometric multiplex* [arXiv:1601.04071](https://arxiv.org/abs/1601.04071); *De Bacco et al. (2017) on layer interdependence* [arXiv:1701.01369](https://arxiv.org/abs/1701.01369); *Mucha et al. (2010) on multislice modularity* [arXiv:0911.1824](https://arxiv.org/abs/0911.1824).

---

# Core concepts you need

**1. A likelihood is what makes a model falsifiable.** Your thesis could simulate but not score; every comparison was a hand-weighted distance. The field's answer is: write down `P(data | θ)`, then you get parameter uncertainty, model comparison, and the ability to *reject*. Everything else in this list follows from that.

**2. Conditioning on a sufficient statistic is a modelling move, not a hack.** "Fixed density" = condition on the total. Poisson → multinomial. Your edge-sampling intuition survives; the intractability doesn't.

**3. Latent variables need marginalising, not assigning.** EM alternates. But the likelihood you *report* must integrate the latent variable out (mixture), not condition on the assignment (classification). Using the wrong one silently underestimates complexity and over-selects models. This was a real bug here.

**4. Identifiability bounds the claim.** "The algorithm found 5 communities" is not a fact about the world (Peel). The right statement is "under model M with K=5, the MAP partition is…". Say the model, say the sensitivity.

**5. Degree correction separates two questions.** How many edges a node has, and who it connects to. Conflating them is what makes disease networks look like utilisation. This is the conceptual core of your `u_i`.

**6. Every null model is a choice.** Configuration model, degree-preserving rewiring, propensity stratification — each answers a different question. Modularity's null is a specific SBM (Pamfil). Pick the null that encodes your alternative hypothesis.

**7. Over-dispersion is not noise.** Real counts vary more than Poisson. Fitting a distribution to point estimates adds the estimation error back in as if it were real heterogeneity. Fit the marginal instead.

**8. Level vs contrast.** A negative-control calibration transfers a *bias distribution between contrasts*. An incidence rate is a *level*, whose bias is outcome-specific and not transferable. This is why "negative controls don't solve incidence" is a structural claim, not a complaint.

**9. Reproduction is not explanation.** A free per-period multiplier will reproduce a pandemic, a coding change, and an ETL bug identically. If you want to claim you modelled something, the parameter must mean something outside the sample.

**10. Scale changes the algorithm, not the model.** The sufficient statistics are sums over patients; nothing is indexed by both patient and concept. That is why petabyte-scale fitting is a streaming problem rather than an impossibility — and why the release has no per-patient parameters at all.
