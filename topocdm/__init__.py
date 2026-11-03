"""
topocdm -- a fixed-density generative model of an OMOP CDM, with a likelihood.

    from topocdm import CDMIncidence, fit, sample_cdm, compare

The model is a degree-corrected block model on the patient x concept incidence
matrix, with person-time as a known offset:

    E[X_ij] = T_i * u_i * lambda_j * B[g_i, h_j]

    u_i     patient care-seeking propensity
    lambda_j concept intensity
    B[g,h]  affinity between patient group g and concept group h (the
            multimorbidity structure, i.e. the topology beyond marginals)

Fitting is closed-form coordinate ascent (IPF + classification EM), so the
log-likelihood increases monotonically and converges without an optimiser.
Generation fixes the total event count M, which is what "fixed density" means.

The same object is used two ways: forwards as a privacy-preserving synthetic
CDM, backwards as the null against which observed incidence is judged.
"""

from topocdm.model import (
    CDMIncidence, Params, mean_matrix, loglik_conditional, loglik_poisson,
    deviance_conditional, aic, bic, n_params, expected_prevalence,
    expected_incidence, loglik_mixture, n_params_mixture, aic_mixture,
    bic_mixture,
)
from topocdm.fit import fit, fit_ipf, fit_propensity, PropensityDist, FitResult
from topocdm.generate import sample_cdm, sample_replicates
from topocdm.simulate import simulate, simulate_null, SimConfig
from topocdm.evaluate import (
    prevalence, incidence, degree, compare, gini, predictive_loglik,
)

__version__ = "0.1.0"

__all__ = [
    "CDMIncidence", "Params", "mean_matrix", "loglik_conditional",
    "loglik_poisson", "deviance_conditional", "aic", "bic", "n_params",
    "expected_prevalence", "expected_incidence",
    "loglik_mixture", "n_params_mixture", "aic_mixture", "bic_mixture",
    "fit", "fit_ipf", "fit_propensity", "PropensityDist", "FitResult",
    "sample_cdm", "sample_replicates",
    "simulate", "simulate_null", "SimConfig",
    "prevalence", "incidence", "degree", "compare", "gini",
    "predictive_loglik",
]
