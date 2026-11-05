# Handoff prompt

*Paste the block below to the next agent. It assumes no knowledge of this conversation.*

---

You are picking up a research codebase at `/Users/jamie/PycharmProjects/networks_research` (private repo `github.com/azimov/topocdm`).

**Who you are working with.** Jamie Azimov — network-science PhD (2015), author of CiGRAM (`github.com/azimov/Cigram`), now doing observational epidemiology on the OMOP CDM. The project asks whether his topology-modelling approach transfers to CDM data. He has strong technical judgement and will catch errors; he has already corrected one overclaim in this project. He does not want statistics lectures — explain in network terms: nodes, edges, degree, null models, spectra. He wants honest grading of claims, not encouragement.

**Read in this order, before writing code:**

1. `HANDOFF.md` — context, environment, ten traps that have already cost time, current state. Start here.
2. `docs/MODEL_V2.pdf` (source `docs/MODEL_V2.md`) — the current model specification, written for external review. This supersedes earlier designs.
3. `WHERE_THIS_STANDS.md` — the five claims the work is judged against, graded against evidence.
4. `BRIEFING.md`, `MODEL_AND_READING.md` — earlier framings, partly superseded. Read for history, not for the current design.

**The project.** Two objectives: (1) model the network topology of a CDM, (2) generate synthetic data that looks real. Causality is explicitly out of scope. Five claims are in play (see `WHERE_THIS_STANDS.md`); the strongest is claim 5 (small, shareable, comparable artefact) and the weakest is claim 3 (calibrating incidence and prevalence).

**Hard constraints.**

- The data is **Synthea**, a rule-based patient simulator. Never write "real data", "real CDM" or "real patients". This was corrected once already and leaked into committed artefacts.
- Do not overclaim. Every previous overclaim in this repo traced to a bookkeeping error, not a modelling error. Check levels (generated totals against target) before interpreting anything.
- Do not select a hyperparameter on the criterion you then report.
- Verify every citation against arXiv, Crossref or PubMed before writing it. A guessed DOI produced a paper on concrete creep.

**Environment.** Run Python as `MPLCONFIGDIR=./.mplconfig .venv/bin/python` — the system python cannot load the signed `.so` for the C kernel. Database access is `docker exec ohdsi-lab-postgres psql -U postgres -d postgres`, schemas `cdm` and `vocab`; there is no `psql` on the host. Build PDFs with `scripts/md2pdf.sh <in.md> <out.pdf>`.

**The immediate task.** Re-extract the event data with **end dates**. `data/events_dated.csv` carries start dates only, but condition and drug are interval processes (medians 364 and 365 days) and that structure is the core of the model — it cannot be fitted without it. Then fit the per-domain temporal model in `docs/MODEL_V2.md` §2: interval processes for condition and drug, point processes for measurement, observation and procedure, an absorbing hazard for death, one shared patient heterogeneity term, and a low-rank cross-domain coupling that is labelled as recording dependence every time it appears.

**Then the experiment that decides everything.** Take a drug–outcome pair with no plausible causal relationship in a real CDM and estimate the association three ways: raw, adjusted for contact volume, and by OHDSI empirical calibration. If the second and third agree, claims 2–4 are live and the method is a contribution. If they do not, claims 2–4 are dead — which is also worth reporting. This is the kill criterion. Do not labour the modelling further until it is answered.

**Done looks like this.** The per-domain model fits and generates; the interval tables are populated (`condition_era` and `drug_era` are empty in the source, and a model that emits intervals can fill them); the spectral comparison is reported alongside the summary statistics rather than buried behind them; and the negative-control experiment has an answer in either direction.

**State of the tree.** Clean and pushed. 15 tests pass. Three models already exist — block/fixed-effects, a factor term, and a visit-anchored topic model — all documented in `HANDOFF.md` §6 with their numbers and their failure modes. Do not rebuild them; build past them.
