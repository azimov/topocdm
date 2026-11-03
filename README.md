# topocdm

A fixed-density generative model of an OMOP CDM, with a likelihood.

`topocdm` fits a topology model to a patient × concept incidence matrix and
generates synthetic CDMs from it. The same fitted object is used two ways:

- **forwards**, as a privacy-preserving synthetic dataset that reproduces a
  target CDM's prevalence, incidence and joint structure;
- **backwards**, as the null against which observed incidence is judged —
  *what would this database look like if nothing caused anything?*

## The core idea

CiGRAM's generative scheme (Gilbert 2015, Ch. 5) drew a fixed number of edges
without replacement, which has no closed-form likelihood. That is what stopped
it being a statistical model.

It turns out fixed density and a likelihood are not in conflict. Fixed density
means *the total number of events is fixed*. Conditioning a Poisson model on
its total gives a multinomial, and the multinomial likelihood over incidence
cells is closed-form:

```
X_ij | theta  ~  Poisson( mu_ij )        mu_ij = T_i * u_i * lambda_j * B[g_i, h_j]
X    | M      ~  Multinomial( M, mu / sum(mu) ),   M = sum_ij X_ij
```

Two models that differ only in scale have the same conditional MLE, so the
**Poisson MLE is the fixed-density MLE**. Fit Poisson, generate by conditioning
on M.

The second half of the resolution is that a CDM is a multigraph: a patient can
be coded with the same concept many times, so there is no sampling without
replacement. That is exactly the constraint that made the original intractable,
and it does not arise here.

| parameter | meaning |
|---|---|
| `u_i` | patient care-seeking propensity — how many codes they accumulate |
| `lambda_j` | concept intensity — how often it occurs per unit of contact |
| `B[g,h]` | affinity between patient group g and concept group h. This is the multimorbidity structure: the part of the topology that the marginals do not contain. |

## Quick start

```python
import topocdm as t

data, truth = t.simulate(t.SimConfig(n=4000, c=60, seed=0))   # or load a real CDM
fit = t.fit(data, n_patient_groups=4, n_restarts=3)
print(t.aic_mixture(fit.params, data), t.bic_mixture(fit.params, data))

synth = t.sample_cdm(fit.params, data, mode="fresh")
print(t.compare(data, synth)["prevalence"])
```

## What is validated, and what is not

On simulated data with known ground truth (`experiments/01_recovery.py`):

- AIC and BIC on the **mixture** likelihood recover the true number of patient
  groups; the mean structure correlates with the truth at r ≈ 0.96.
- Latent groups are recovered at ~81% accuracy when the signal is clear.

On held-out patients (`experiments/02_validate.py`), fitting on 75% and
predicting the other 25%:

| | prevalence slope | co-occurrence r | held-out loglik/event |
|---|---|---|---|
| topology model | 0.92 | 0.980 | **−2.59** |
| marginals only | 0.80 | 0.946 | −3.00 |

and, critically, **in a world with no block structure, AIC selects a single
group** — the method does not invent structure that is not there.

Not yet validated: any real CDM. The next milestone is fitting to a real
database and checking that the observation model reproduces its visit and
channel structure.

## A warning that is also a result

The likelihood used for *fitting* (classification likelihood, hard assignment)
is not the likelihood used for *model selection* (mixture likelihood,
integrated over groups). Using the former for AIC silently treats every
patient's group assignment as a free parameter that is never counted, and model
selection then marches off towards infinitely many groups. This happened during
development, was fixed by moving to soft EM and `loglik_mixture`, and is
recorded here because it is an easy trap.

## Layout

```
topocdm/          the library
  model.py        mean structure, likelihoods, AIC/BIC
  fit.py          EM / IPF, identifiability constraints, propensity fitting
  generate.py     fixed-density sampling (replay and fresh modes)
  simulate.py     ground-truth CDMs with known topology
  evaluate.py     prevalence, incidence, co-occurrence, held-out prediction
experiments/      01 recovery, 02 validation
tests/            pytest suite
docs/DESIGN.md    the likelihood argument in full
explore/          earlier exploratory work (CiGRAM probes, bias demos)
figs/, results/   generated output
```

`explore/` contains the probes behind `REPORT.md` and `PROGRAMME.md`; it is not
part of the library.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q
MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/02_validate.py
```

## Citation

Gilbert, J.P. (2015) *A probabilistic model for the analysis of modular
structure in complex biological networks*. PhD thesis, University of Nottingham.
Chapter 5.

## Licence

MIT.
