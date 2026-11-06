"""
cdm_sim.py -- a visit-driven simulator of a CDM-like observational database.

The point of this module is not realism of content. It is realism of the
*observation process*: everything a CDM records is an event that happened to
be observed when a patient came into contact with the health system, through
a particular channel (an examination, a laboratory test, a prescription).

Two ideas are encoded explicitly, because they are the two things synthetic
health data normally omits:

  1. A VISIT PROCESS. Patients have a latent care-seeking propensity ("frailty"
     or utilisation, u_i). Visits arrive as a Poisson process with rate
     proportional to u_i. The number of codes a patient accumulates is
     therefore a function of u_i, not of their health.

  2. CHANNELS. A visit is not one observation. An examination happens at every
     visit; a lab test happens only sometimes, at a probability that depends on
     the care site and on the patient's treatment. A finding is only recorded
     if the channel that can see it is open.

The consequences, for a condition that is *incidentally* found rather than
symptom-driven, are:

    observed incidence  ≈  detection probability  x  channel openings

so the observed incidence of an ingrown toenail tracks how often the patient
was examined, and the observed incidence of a mild lab abnormality tracks how
often bloods were taken. Neither tracks disease.

This is the mechanism that makes a CDM's topology a *model of the recording
process*: if you can regenerate the topology, you have regenerated the bias.

Terminology follows the thesis where it applies: nodes (patients) carry a
latent propensity that sets their degree, and the graph is generated at a
fixed event budget rather than by conditioning on a degree sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    """Parameters of the simulated health system."""

    n: int = 20000
    years: float = 5.0

    # population
    frailty_sigma: float = 0.8  # lognormal SD of care-seeking propensity
    n_sites: int = 40
    site_sigma: float = 0.5  # lognormal SD of site-level lab propensity

    # visit process
    base_visit_rate: float = 1.2  # visits per person-year at u = 1
    p_lab: float = 0.35  # labs per visit at site multiplier 1

    # incidental detection
    eps_exam: float = 0.02  # P(record | exam)
    eps_lab: float = 0.08  # P(record | lab)

    # exposure
    exposure_frac: float = 0.30
    t_exposure: float = 2.5  # years from start
    visit_mult: float = 1.2  # exposure raises visit rate
    lab_mult: float = 2.0  # exposure raises lab propensity

    # a true causal effect, on a condition that is detected through labs
    rr_true: float = 1.5
    lam_background: float = 0.10  # episodes per person-year
    tau_episode: float = 0.25  # years an episode remains detectable

    seed: int = 0


@dataclass
class Population:
    """The latent population and its realised contact history."""

    frailty: np.ndarray
    site_mult: np.ndarray
    exposed: np.ndarray
    visits_pre: np.ndarray
    visits_post: np.ndarray
    labs_pre: np.ndarray
    labs_post: np.ndarray
    person_time: np.ndarray
    exposed_time: np.ndarray
    unexposed_time: np.ndarray
    scn: Scenario = field(repr=False)

    @property
    def visits(self):
        return self.visits_pre + self.visits_post

    @property
    def labs(self):
        return self.labs_pre + self.labs_post

    @property
    def exams(self):
        return self.visits


def simulate_population(scn: Scenario | None = None, seed: int | None = None) -> Population:
    """Generate the latent population and its realised visits and labs."""
    scn = scn or Scenario()
    rng = np.random.default_rng(scn.seed if seed is None else seed)
    n = scn.n

    frailty = rng.lognormal(0.0, scn.frailty_sigma, size=n)
    frailty /= np.exp(0.5 * scn.frailty_sigma**2)  # mean 1, so rates are interpretable
    site_id = rng.integers(0, scn.n_sites, size=n)
    site_mult = rng.lognormal(0.0, scn.site_sigma, size=scn.n_sites)[site_id]

    u = scn.base_visit_rate * frailty
    exposed = rng.random(n) < scn.exposure_frac

    pre = scn.t_exposure * np.ones(n)
    post = (scn.years - scn.t_exposure) * np.ones(n)

    # exposure multiplies rates only after t_exposure (time-fixed exposure)
    v_mult = np.where(exposed, scn.visit_mult, 1.0)
    l_mult = np.where(exposed, scn.lab_mult, 1.0)

    visits_pre = rng.poisson(u * pre)
    visits_post = rng.poisson(u * v_mult * post)

    p_lab_site = scn.p_lab * site_mult
    labs_pre = rng.binomial(visits_pre, np.minimum(p_lab_site, 1.0))
    labs_post = rng.binomial(visits_post, np.minimum(p_lab_site * l_mult, 1.0))

    person_time = pre + post
    exposed_time = np.where(exposed, post, 0.0)
    unexposed_time = person_time - exposed_time

    return Population(
        frailty=frailty,
        site_mult=site_mult,
        exposed=exposed,
        visits_pre=visits_pre,
        visits_post=visits_post,
        labs_pre=labs_pre,
        labs_post=labs_post,
        person_time=person_time,
        exposed_time=exposed_time,
        unexposed_time=unexposed_time,
        scn=scn,
    )


def incidental(pop: Population, eps: float, channel: str, rng) -> tuple:
    """Count an incidentally-detected condition, separately by period.

    Each opening of the relevant channel is a Bernoulli trial. Nothing about
    the patient's health enters, only how often they were looked at.
    Returns (count_before_exposure, count_after_exposure).
    """
    if channel == "exam":
        opens = (pop.visits_pre, pop.visits_post)
    elif channel == "lab":
        opens = (pop.labs_pre, pop.labs_post)
    else:
        raise ValueError("channel must be 'exam' or 'lab'")
    return tuple(rng.binomial(o, eps) for o in opens)


def true_effect(pop: Population, rng) -> dict:
    """A condition with a genuine exposure effect, detected only through labs.

    Episodes occur at a Poisson rate that the exposure multiplies by rr_true.
    An episode is detected only if a lab falls inside its detectable window, so
    the *observed* count depends on the lab rate as well as on the true rate.
    """
    scn = pop.scn
    pre = scn.t_exposure
    post = scn.years - scn.t_exposure

    rr = np.where(pop.exposed, scn.rr_true, 1.0)

    # lab rate (opens per year) in each period
    lab_rate_pre = scn.base_visit_rate * pop.frailty * scn.p_lab * pop.site_mult
    lab_rate_post = lab_rate_pre * np.where(pop.exposed, scn.visit_mult * scn.lab_mult, 1.0)

    p_det_pre = 1.0 - np.exp(-lab_rate_pre * scn.tau_episode)
    p_det_post = 1.0 - np.exp(-lab_rate_post * scn.tau_episode)

    episodes_pre = rng.poisson(scn.lam_background * pre)
    episodes_post = rng.poisson(scn.lam_background * rr * post)

    rng.binomial(episodes_pre, p_det_pre) + rng.binomial(episodes_post, p_det_post)

    obs_pre = rng.binomial(episodes_pre, p_det_pre)
    obs_post = rng.binomial(episodes_post, p_det_post)
    return {
        "observed_pre": obs_pre,
        "observed_post": obs_post,
        "true_pre": episodes_pre,
        "true_post": episodes_post,
        "p_det_pre": p_det_pre,
        "p_det_post": p_det_post,
    }


def to_event_table(pop: Population, concepts: dict, rng) -> np.ndarray:
    """Expand to a long (patient, concept, count) table, CDM-style.

    Only counts are returned here; a real CDM would carry dates and visits.
    """
    rows = []
    for name, counts in concepts.items():
        idx = np.nonzero(counts > 0)[0]
        for i in idx:
            rows.append((i, name, int(counts[i])))
    return np.array(rows, dtype=object) if rows else np.empty((0, 3), dtype=object)
