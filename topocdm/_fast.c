/*
 * topocdm/_fast.c -- CSR mixture log-likelihood kernel.
 *
 * Evaluates
 *
 *     sum_i log sum_g pi_g prod_j Poisson(x_ij | T_i u_i lambda_j B[g,h_j])
 *
 * directly on the CSR arrays, so the cost is O(nnz * G + n * G) and no dense
 * chunk is built. The Python fallback densifies the chunk and calls lgamma
 * over the whole block once per group, which is O(chunk * C * G) with large
 * temporaries.
 *
 * The subtlety that makes this correct: a *zero* cell still contributes
 *
 *     -mu_ij = -T_i u_i lambda_j B[g,h_j]
 *
 * but CSR does not store zeros. The total is therefore split into
 *
 *     sum_{x>0} [x log mu - lgamma(x+1)]   (nonzero entries only)
 *   - T_i u_i (B Lambda)_g                 (all cells, closed form)
 *
 * with Lambda_h = sum_{j in h} lambda_j. Getting this wrong silently inflates
 * the likelihood by roughly sum(T u Lambda).
 *
 * Build: scripts/build_fast.sh
 */

#include <math.h>
#include <stddef.h>

static double logsumexp_small(const double *a, long G)
{
    double m = a[0];
    for (long g = 1; g < G; ++g)
        if (a[g] > m) m = a[g];
    if (!isfinite(m)) return m;
    double s = 0.0;
    for (long g = 0; g < G; ++g) s += exp(a[g] - m);
    return m + log(s);
}

/*
 * chunk_loglik_csr
 *
 *   indptr, indices, data : CSR arrays for the chunk (n x c)
 *   T, u                  : length n
 *   lam                   : length c
 *   B                     : G x H, row-major
 *   pi                    : length G
 *   cg                    : length c, concept group per column
 *   acc, lam_h, blam      : scratch buffers, lengths G, H, G
 */
double chunk_loglik_csr(const long *indptr, const long *indices,
                        const double *data,
                        const double *T, const double *u,
                        const double *lam, const double *B,
                        const double *pi, const long *cg,
                        long n, long c, long G, long H,
                        double *acc, double *lam_h, double *blam)
{
    if (G <= 0 || H <= 0) return NAN;

    for (long h = 0; h < H; ++h) lam_h[h] = 0.0;
    for (long j = 0; j < c; ++j) lam_h[cg[j]] += lam[j];
    for (long g = 0; g < G; ++g) {
        double s = 0.0;
        for (long h = 0; h < H; ++h) s += B[g * H + h] * lam_h[h];
        blam[g] = s;
    }

    double total = 0.0;
    for (long i = 0; i < n; ++i) {
        const double base_i = T[i] * u[i];
        for (long g = 0; g < G; ++g) acc[g] = 0.0;

        for (long k = indptr[i]; k < indptr[i + 1]; ++k) {
            const long j = indices[k];
            const double x = data[k];
            if (x == 0.0) continue;
            const double base = base_i * lam[j];
            const double lg = lgamma(x + 1.0);
            const double xl = x * log(base);
            const long h = cg[j];
            for (long g = 0; g < G; ++g)
                acc[g] += xl + x * log(B[g * H + h]) - lg;
        }

        for (long g = 0; g < G; ++g)
            acc[g] += -base_i * blam[g] + log(pi[g]);
        total += logsumexp_small(acc, G);
    }
    return total;
}

/* Single-group special case: no group loop, no logsumexp. */
double chunk_loglik_csr_g1(const long *indptr, const long *indices,
                           const double *data,
                           const double *T, const double *u,
                           const double *lam, long n, long c,
                           const double *lam_total)
{
    (void)c;
    double total = 0.0;
    for (long i = 0; i < n; ++i) {
        const double base_i = T[i] * u[i];
        double acc = 0.0;
        for (long k = indptr[i]; k < indptr[i + 1]; ++k) {
            const long j = indices[k];
            const double x = data[k];
            if (x == 0.0) continue;
            acc += x * log(base_i * lam[j]) - lgamma(x + 1.0);
        }
        total += acc - base_i * (*lam_total);
    }
    return total;
}
