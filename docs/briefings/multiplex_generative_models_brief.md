# Multiplex/multilayer generative models and community structure, 2013–2026

Links resolved (arXiv/Crossref/OpenAlex); flags at end.

## 1. Generative-model families

**Multiplex / multilayer SBMs.** [Barbillon et al. 2017](https://arxiv.org/abs/1501.06444) extend the SBM to multiplex data: shared block memberships, one block-probability matrix per layer, variational EM, ICL selection of K. [Valles-Català et al. 2016](https://arxiv.org/abs/1411.1098) invert it — from one aggregate graph, infer the latent layers, treating aggregation as a model-selection problem. [Stanley et al. 2016](https://arxiv.org/abs/1507.01826) add "strata": groups of layers sharing one SBM — the cleanest layer-coupled formulation, since coupling is a layer partition, not a parameter. [De Bacco et al. 2017](https://arxiv.org/abs/1701.01369) is the most flexible: overlapping communities shared across layers, each layer mixing them via its own affinity matrix (assortative, disassortative or directed), degree parameters, and an information-theoretic layer-interdependence measure enabling layer bundling and link prediction. All four are likelihood-based. None has latent geometry.

**Tensor / spectral multilayer SBMs.** Regularised tensor decomposition (TWIST; ALMA, [Fan et al. 2021](https://arxiv.org/abs/2102.10226)) jointly recovers layer partition and communities with consistency guarantees; [Noroozi & Pensky 2023](https://arxiv.org/abs/2206.07602) relax the shared-block-probability assumption. The sharpest theory, but tied to N×N×L tensors of comparable, reasonably dense layers.

**Geometric / latent-space multiplex models.** [Kleineberg et al. 2016](https://arxiv.org/abs/1601.04071) show real multiplexes carry correlated hyperbolic coordinates across layers and build the Geometric Multiplex Model on it; [Papadopoulos & Kleineberg 2019](https://arxiv.org/abs/1807.01190) show geometry alone cannot explain edge overlap and add link persistence. These reproduce degree heterogeneity, clustering and self-similar tails — the natural home for a CiGRAM-like construction — but are fitted by simulation/moments, not likelihood. The Bayesian latent-position alternative is [Jin et al. 2024](https://arxiv.org/abs/2301.00055): Gaussian-mixture latent positions, layer-specific distance scaling, covariates, MCMC.

**Configuration-model / ensemble nulls.** [Bianconi 2013](https://doi.org/10.1103/PhysRevE.87.062806) gives the statistical mechanics — entropy, degree constraints, overlap as order parameter; [Battiston et al. 2014](https://arxiv.org/abs/1308.3182) supply the multiplex structural measures. They fix layer degree sequences and answer "is this overlap surprising?", but generate no communities, and not every (degree, overlap) pair is realisable.

CiGRAM sits in the geometric corner: interpretable control (σ_f, σ_s, a, e_k, p_o), no likelihood, PSO on summaries. No family above delivers degree heterogeneity **and** geometry **and** block communities **and** tunable interlayer coupling; SBMs have three of the four, geometric models two.

## 2. Multiplex community detection

**Multilayer modularity** is the default: [Mucha et al. 2010](https://arxiv.org/abs/0911.1824) define multislice modularity with resolution γ and interlayer coupling ω, optimised by GenLouvain. [Pamfil et al. 2018](https://arxiv.org/abs/1804.01964) show it is exactly MAP inference under a particular multilayer SBM — and hence that its communities are artefacts of that model and its null.

The parameters are the weak point. [Amelio & Tagarelli 2019](https://arxiv.org/abs/1709.07253) note γ and ω "can be set arbitrarily, thus discarding any structure information"; [Vaiana & Muldoon 2018](https://arxiv.org/abs/1803.03597) prove an upper bound on ω beyond which community change across layers is undetectable — a purely multilayer resolution limit — plus an explicit γ–ω trade-off. [Bazzi et al. 2016](https://arxiv.org/abs/1501.00040) show the same null network can correspond to different null models, so multilayer modularity is under-specified even with γ and ω fixed. Partitions are therefore not identified.

**Infomap for multiplexes.** [De Domenico et al. 2015](https://arxiv.org/abs/1408.2925) compress flows on the non-aggregated multiplex and recover modules that neither aggregation nor per-layer analysis finds, with more overlap and smaller modules — the strongest flow-based method.

**SBM / belief propagation.** [Weir et al. 2020](https://arxiv.org/abs/1908.04653) extend modularity belief propagation to multilayer networks, build in γ and ω, match GenLouvain on benchmarks, and — the real contribution — expose convergence behaviour and node marginals, turning parameter choice into a diagnostic.

**Bayesian.** [Bian et al. 2026](https://arxiv.org/abs/2301.07386) use a hierarchical Bayesian latent block model with unknown K and between-subject variability, beating multilayer modularity on split-half reproducibility in HCP fMRI. [De Santiago et al. 2024](https://arxiv.org/abs/2401.04682) establish identifiability for a mixture of multilayer SBMs and select K variationally.

**What "community" means across layers** is unsettled: either nodes similar in several layers at once ("multidimensional communities", [Kleineberg et al. 2016](https://arxiv.org/abs/1601.04071)) or a layer-specific partition free to differ by layer. [Peel et al. 2017](https://arxiv.org/abs/1608.05878) prove no algorithm can uniquely solve community detection, so this is a modelling choice, not a discoverable fact.

## 3. Nulls and benchmarks

The principled generator is [Bazzi et al. 2020](https://arxiv.org/abs/1608.06196): a framework unifying multilayer mesoscale generators for ordered and unordered layers, with an *explicit user-specified interlayer dependency structure* — what ad hoc multiplex benchmarks lack. [mABCD 2025](https://doi.org/10.1016/j.eswa.2025.130920) is the current standardised multilayer benchmark, the multilayer successor to ABCD/LFR with controllable community structure.

Flaws in LFR-style multiplex benchmarks: (i) they plant a block partition then measure recovery of it, so "community" is defined by the generator ([Peel et al. 2017](https://arxiv.org/abs/1608.05878)); (ii) most variants copy one partition into every layer with a coupling knob, so they cannot test layer-specific or overlapping communities — exactly Bazzi et al.'s motivation; (iii) degree correction and layer-density heterogeneity are usually absent.

Missing, per [Boukabene & Benbouzid Si Tayeb 2025](https://arxiv.org/abs/2511.23247): benchmarks with layer-specific partitions, tunable dependence, degree heterogeneity, and *known detectability thresholds* analogous to single-layer SBM limits.

## 4. The credible critiques

- **Most multiplexes are reducible.** [De Domenico et al. 2015](https://doi.org/10.1038/ncomms7864) give an information-theoretic procedure showing many empirical multilayer networks collapse to far fewer layers with negligible loss. Many layers are redundant; multiplexity is often overstated by construction.
- **Aggregation is often as good.** [Taylor et al. 2016](https://arxiv.org/abs/1511.05271) show that for L multiplex layers drawn from a common SBM, the detectability limit under plain summation vanishes as O(L^-1/2), and thresholded summation behaves similarly. Multilayer machinery must beat a very cheap baseline, and often does not.
- **Identifiability.** [Peel et al. 2017](https://arxiv.org/abs/1608.05878) prove no unique community solution exists and that metadata are not ground truth. Corollary: when layers correlate with node attributes, "multiplex community structure" can be covariates re-expressed as geometry.
- **Interdependence is assumed, then "discovered".** [De Bacco et al. 2017](https://arxiv.org/abs/1701.01369) is the honest version: it measures how much one layer predicts another, and finds layers redundant given the rest.
- **Parameter degeneracy.** The γ/ω surface ([Vaiana & Muldoon 2018](https://arxiv.org/abs/1803.03597); [Amelio & Tagarelli 2019](https://arxiv.org/abs/1709.07253)) means published multiplex partitions are rarely uniquely determined.

## 5. Open problems (2023–2026) and CiGRAM's niche

[Boukabene & Benbouzid Si Tayeb 2025](https://arxiv.org/abs/2511.23247) list scalability in L, heterogeneous layer semantics, absence of ground truth and realistic benchmarks, dynamic/streaming multiplexes, and metrics that reward more than modularity. The technical literature adds non-stationarity ([Wang et al. 2026](https://arxiv.org/abs/2604.25156)), unbalanced layers and missing data ([Tang et al. 2025](https://arxiv.org/abs/2509.05221)), and identifiability of coupling parameters ([De Santiago et al. 2024](https://arxiv.org/abs/2401.04682)).

The gap: no generative model with a *closed-form likelihood* delivers degree heterogeneity, explicit latent geometry, block communities and tunable interlayer coupling together. Geometric multiplex models own the geometry and lose the likelihood; multilayer SBMs own the likelihood and lose the geometry. A multiplex CiGRAM whose layer coupling is an identifiable parameter — or at minimum a simulation-based-inference route replacing PSO on summary statistics — would occupy empty space. Expect identification problems between σ_f, σ_s and the coupling; the failures on power-law tails and on assortativity outside the sparse regime will persist.

---

**Verification flags.** (1) I could not verify a single canonical "multiplex LFR" paper; the term appears to be an in-paper construction rather than one citable standard, so §3 cites the standardised generators instead. (2) The Gilbert 2015 Nottingham PhD thesis describing CiGRAM is taken from your brief and not independently verified.
