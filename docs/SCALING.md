# Scaling out on a warehouse / Databricks

Two separate problems, solved separately:

1. **Collection** — getting events out of the CDM. Push it into SQL, decompose it
   into checkpointed steps, let the warehouse parallelise.
2. **Fitting** — the EM loop. Split it into map / reduce / update so partitions
   run independently and only kilobytes are shuffled.

## 1. Collection: checkpointed SQL

The pattern follows OHDSI/Circepy#60 (decomposable cohort sets, shared
checkpoint execution): every step is a self-contained query, content-addressed
by the hash of its SQL, materialised once, and read by downstream steps instead
of the raw CDM.

```
ev_condition    condition_occurrence  -> (person_id, concept_id, n)
ev_drug         drug_exposure         -> (person_id, concept_id, n)
ev_procedure    procedure_occurrence  -> ...
ev_measurement  measurement           -> ...
ev_observation  observation           -> ...
person_time     observation_period    -> (person_id, days_observed)
concept_group   vocab.concept         -> (concept_id, domain, class)
incidence       union of the domain checkpoints
```

Why this matters at scale, beyond parallelism:

- **Resumability.** If the measurement job fails after two hours, the other four
  domain checkpoints survive. Re-running skips them — measured at 51 s cold and
  1.1 s warm on the local 10k-person CDM.
- **Reuse.** The domain aggregations are the same shape any cohort or
  characterisation analysis wants. They are worth materialising once.
- **Dialect independence.** The aggregation SQL is plain ANSI and hashes the
  same on every platform, so a checkpoint computed on Postgres is described by
  the same content hash as one computed on Databricks. Only date arithmetic
  differs, and it is isolated in `topocdm.sql.DIALECTS`.
- **Validation.** On the local CDM the SQL path reproduces the Python extractor
  exactly: 750,053 rows, identical counts, 4,925,516 events.

Portability caveats worth knowing:

- `CREATE OR REPLACE TABLE` is not Postgres; the executor emits
  `DROP … IF EXISTS; CREATE TABLE … AS` there. See `CREATE_REPLACE`.
- `standard_concept` has no value `'C'` in every vocabulary version; if the
  concept set comes back short, check that filter first.
- Databricks needs `CREATE DATABASE`, not `CREATE SCHEMA`. See `SCHEMA_DDL`.

## 2. Fitting: map / reduce / update

The global parameters depend on the data only through sums over patients:

```
colsum_j  = sum_i X_ij                 events per concept
num_gh    = sum_i R_ig n_ih            events per (patient group, concept group)
A_g       = sum_i R_ig T_i u_i         propensity-weighted group mass
pi_num_g  = sum_i R_ig                 responsibility mass
ll        = sum of per-patient log-likelihoods
```

Nothing in that list is indexed by a patient. So:

```
epoch:
    parts  = rdd.mapPartitions(partial_stats)      # needs lambda, B, pi (broadcast)
    stats  = parts.treeAggregate(zero, merge, merge)
    lam, B, pi = m_step(stats)                     # driver
```

Broadcast is a few kilobytes (`C + G*H`). The shuffle is a single record per
partition. Patient-level data never leaves its partition.

`topocdm/distributed.py` implements exactly this, and
`tests/test_model.py::test_partitioned_fit_matches_sequential` asserts that the
reduced form reproduces the sequential fit **bit for bit** (`lam` and `B` differ
by 0.0e+00).

### Databricks wiring

```python
from topocdm.distributed import partition_stats, merge, m_step, Stats

# 1. read the checkpoint in parallel, one partition per person_id range
df = (spark.read.format("jdbc")
      .option("url", jdbc_url).option("dbtable", "topocdm.incidence")
      .option("partitionColumn", "person_id")
      .option("numPartitions", 512)
      .option("lowerBound", lo).option("upperBound", hi)
      .load())
df.cache()          # the fit makes several passes

# 2. one epoch
def part(rows):
    X, T = build_csr(rows)                      # chunk -> scipy CSR
    return iter([partition_stats(X, T, lam_b.value, B_b.value, pi_b.value,
                                 cg_b.value, G)])

stats = (df.rdd.mapPartitions(part)
           .treeAggregate(Stats.zeros(C, G, H), merge, merge))
lam, B, pi = m_step(stats, lam, B, pi, cg, G)
```

Two practical notes:

- **Cache the checkpoint.** EM makes several passes; without caching you pay the
  JDBC read every epoch.
- **Balance partitions by person, not by rows.** `partitionColumn=person_id`
  gives each task a contiguous patient range, which is what the chunk contract
  expects. Skewed humans (a few patients with thousands of events) can unbalance
  a partition; sample the partition boundaries from the person_id distribution
  rather than using a uniform range if that shows up.

## 3. What to expect at 300M patients

| term | size | scales with |
|---|---|---|
| `u_all` if kept | 2.4 GB | **N** — drops to zero with `keep_u=False` |
| chunk working set | ~250 MB | chunk size × rows per patient |
| global parameters | <1 MB | C, G·H |
| broadcast per epoch | <1 MB | C, G·H |

Memory is chunk-bound, not N-bound. **The constraint is I/O**: at ~100 distinct
concepts per patient, 300M patients is ~30B cells, and extraction dominates.
Subsample — 5% is 15M patients, still far more than the release needs, and the
release contains no per-patient parameters so nothing is lost.

## 4. What is *not* parallelised

- **The M-step.** One per epoch on the driver. It is `O(C + G·H)` arithmetic,
  microseconds.
- **The warm start.** `_init_from_sample_fit` densifies a bounded sample and
  runs the exact EM; it is deliberately small and runs once.
- **`lgamma`.** Done once per nonzero cell in the C kernel, parallel across
  partitions by construction.
