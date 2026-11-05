"""
14 -- the visit-anchored model.

A different model from the fixed-effects one: the visit is the unit, concepts at
a visit come from a topic catalogue, and topics follow a Markov chain. The
per-patient quantity is a *count of visits per topic*, not a free parameter.

Parameters: K*C (catalogue) + K*K (transitions) + 2K. Independent of patients.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import topocdm as t
from topocdm.io import load_incidence
from topocdm.visitmodel import fit_topics, fit_transitions, VisitTopicModel, simulate_visits

RES, FIG = "results", "figs"

d, groups, cv = load_incidence()
ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"],
                 usecols=["person_id","concept_id","event_date"])
ev = ev[ev.concept_id.isin(set(d.concept_ids))].copy()
ev["j"] = ev.concept_id.map({c: i for i, c in enumerate(d.concept_ids)})
ev = ev.sort_values(["person_id","event_date"])
ev["vid"] = ev.groupby(["person_id","event_date"], sort=False).ngroup()

V = sparse.csr_matrix((np.ones(len(ev)), (ev["vid"].to_numpy(), ev["j"].to_numpy())),
                      shape=(ev["vid"].nunique(), d.c))
V.sum_duplicates()
vperson = ev.groupby("vid")["person_id"].first().to_numpy()
vdate = ev.groupby("vid")["event_date"].first().to_numpy()
print(f"visits {V.shape[0]:,} x concepts {V.shape[1]:,}: {int(V.sum()):,} records")

def spectrum(M):
    M = M if sparse.issparse(M) else sparse.csr_matrix(np.asarray(M, float))
    r = np.asarray(M.sum(1)).ravel(); c = np.asarray(M.sum(0)).ravel()
    N = sparse.diags(1/np.sqrt(np.maximum(r,1e-9))) @ M @ sparse.diags(1/np.sqrt(np.maximum(c,1e-9)))
    e = np.sort(np.linalg.eigvalsh((N.T@N).toarray()))[::-1]
    return np.sqrt(np.maximum(e,0))
s_real = spectrum(d.X)
grid = np.linspace(0, 1, 200)
cr = np.searchsorted(np.sort(s_real), grid, side="right")/len(s_real)
def ks(x): return float(np.abs(cr - np.searchsorted(np.sort(x),grid,side="right")/len(x)).max())
print(f"real: rank>0.1 {int((s_real>0.1).sum())}")

vc = pd.Series(vperson).value_counts().reindex(d.person_ids, fill_value=0).to_numpy().astype(int)
vs = float(np.mean(np.asarray(V.sum(1)).ravel()))
print(f"visits/patient mean {vc.mean():.1f}  visit size {vs:.2f}")

rows = []
for K in [80, 120, 200]:
    t0 = time.time()
    R, H, pi, ll = fit_topics(V, K=K, n_iter=30)
    topic = R.argmax(1)
    T, rho = fit_transitions(topic, vperson)
    m = VisitTopicModel(theta=H, trans=T, rho=rho,
                        visits_per_patient=vc.mean(), gap_mean=0.0,
                        visit_size=vs, visit_count_pool=vc)
    Xs = simulate_visits(m, d.n, np.random.default_rng(7))
    s = spectrum(Xs)
    rows.append(dict(K=K, params=m.n_parameters, ks=ks(s),
                     rank=int((s>0.1).sum()), loglik=float(ll), secs=time.time()-t0))
    print(f"K={K:3d}  params {m.n_parameters:7,}  KS {rows[-1]['ks']:.3f}  "
          f"rank>0.1 {rows[-1]['rank']:3d}  ({time.time()-t0:.0f}s)")
    np.savez_compressed(f"{RES}/visittopic_K{K}.npz", H=H, pi=pi, T=T, rho=rho, topic=topic)

import json; json.dump(dict(rows=rows, real_rank=int((s_real>0.1).sum()),
                           n_patients=int(d.n), n_visits=int(V.shape[0]),
                           n_concepts=int(d.c)), open(f"{RES}/visitmodel.json","w"), indent=1)
