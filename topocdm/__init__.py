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

from topocdm.cohort_incidence import cohort_incidence, monte_carlo_ci, poisson_ci, predictive_ci
from topocdm.distributed import Stats, fit_partitioned, m_step, partition_stats
from topocdm.distributed import merge as merge_stats
from topocdm.evaluate import (
    compare,
    degree,
    gini,
    incidence,
    predictive_loglik,
    prevalence,
)
from topocdm.factors import FactorFit, fit_factors
from topocdm.fit import FitResult, PropensityDist, fit, fit_ipf, fit_propensity
from topocdm.generate import sample_cdm, sample_replicates
from topocdm.model import (
    CDMIncidence,
    Params,
    aic,
    aic_mixture,
    bic,
    bic_mixture,
    deviance_conditional,
    expected_incidence,
    expected_prevalence,
    loglik_conditional,
    loglik_mixture,
    loglik_poisson,
    mean_matrix,
    n_params,
    n_params_mixture,
)
from topocdm.omop import ConceptVocab, generate_omop_tables, write_omop_tables
from topocdm.release import Release, make_release, sample_release
from topocdm.scalable import (
    SparseCDM,
    StreamingFit,
    array_chunks,
    fit_propensity_streaming,
    fit_streaming,
    simulate_sparse,
    to_sparse,
)
from topocdm.simulate import SimConfig, simulate, simulate_null
from topocdm.sql import CheckpointExecutor, Step
from topocdm.sql import plan as sql_plan
from topocdm.temporal import (
    TemporalProfile,
    fit_temporal,
    generate_dated,
    sample_periods,
    year_exposure,
)
from topocdm.visitanchor import VisitAnchorFit, fit_visitanchor, generate_visitanchor
from topocdm.visitmodel import (
    VisitTopicModel,
    fit_topics,
    fit_transitions,
    simulate_topic_counts,
    simulate_visits,
)
from topocdm.visits import VisitModel, assign_to_visits, fit_visit_model

__version__ = "0.1.0"

__all__ = [
    "CDMIncidence",
    "Params",
    "mean_matrix",
    "loglik_conditional",
    "loglik_poisson",
    "deviance_conditional",
    "aic",
    "bic",
    "n_params",
    "expected_prevalence",
    "expected_incidence",
    "loglik_mixture",
    "n_params_mixture",
    "aic_mixture",
    "bic_mixture",
    "fit",
    "fit_ipf",
    "fit_propensity",
    "PropensityDist",
    "FitResult",
    "sample_cdm",
    "sample_replicates",
    "simulate",
    "simulate_null",
    "SimConfig",
    "prevalence",
    "incidence",
    "degree",
    "compare",
    "gini",
    "predictive_loglik",
    "SparseCDM",
    "to_sparse",
    "fit_streaming",
    "array_chunks",
    "StreamingFit",
    "simulate_sparse",
    "fit_propensity_streaming",
    "ConceptVocab",
    "generate_omop_tables",
    "write_omop_tables",
    "Release",
    "make_release",
    "sample_release",
    "sql_plan",
    "CheckpointExecutor",
    "Step",
    "VisitModel",
    "fit_visit_model",
    "assign_to_visits",
    "FactorFit",
    "fit_factors",
    "VisitTopicModel",
    "fit_topics",
    "fit_transitions",
    "simulate_visits",
    "simulate_topic_counts",
    "VisitAnchorFit",
    "fit_visitanchor",
    "generate_visitanchor",
    "partition_stats",
    "merge_stats",
    "m_step",
    "fit_partitioned",
    "Stats",
    "TemporalProfile",
    "fit_temporal",
    "sample_periods",
    "generate_dated",
    "year_exposure",
    "cohort_incidence",
    "poisson_ci",
    "predictive_ci",
    "monte_carlo_ci",
]
