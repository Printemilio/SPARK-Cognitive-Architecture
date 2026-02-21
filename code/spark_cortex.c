/*
 * ======================================================================================
 * SPARK AI CORE - Synaptic Plasticity & Adaptive Reinforcement Knowledge
 * ======================================================================================
 *
 * Copyright (c) 2025 Emilio Decaix Massiani
 *
 * Licensed under the MIT License. See LICENSE file in the project root for full terms.
 *
 * --------------------------------------------------------------------------------------
 * DEVELOPMENT METHODOLOGY DISCLOSURE:
 * This software represents an experiment in AI-Augmented Research.
 *
 * - CONCEPT & ARCHITECTURE: Conceived and designed by Emilio Decaix Massiani.
 * - IMPLEMENTATION: The C kernels were generated, refined, and optimized through
 * an iterative process using LLM-based agents under strict human supervision.
 *
 * The intellectual property of the architectural logic and theoretical framework
 * remains the sole work of the author.
 * --------------------------------------------------------------------------------------
 */

// spark_cortex.c

#include "spark_cortex.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "spark_utils.h"

#define SPARK_CORTEX_HISTORY 8
#define SPARK_CORTEX_ALPHA_DEFAULT 0.5
#define SPARK_CORTEX_BETA_DEFAULT 1.5
#define SPARK_CORTEX_INERTIA_DECAY 0.7
#define SPARK_CORTEX_STEP_DEFAULT 0.15
#define SPARK_CORTEX_DAMPING 0.12f

static size_t cortex_effective_dim(const CortexModule *ctx) {
    if (ctx && ctx->memory && ctx->memory->d > 0) return ctx->memory->d;
    if (ctx && ctx->dim > 0) return ctx->dim;
    return 0;
}

static void cortex_to_f32(const double *src, float *dst, size_t n) {
    if (!src || !dst) return;
#if defined(__AVX__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_set_ps((float)src[i + 7], (float)src[i + 6], (float)src[i + 5], (float)src[i + 4],
                                 (float)src[i + 3], (float)src[i + 2], (float)src[i + 1], (float)src[i]);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) dst[i] = (float)src[i];
#else
    for (size_t i = 0; i < n; ++i) dst[i] = (float)src[i];
#endif
}

static void cortex_copy_f32(float *dst, const float *src, size_t n) {
    if (!dst || !src) return;
    memcpy(dst, src, n * sizeof(float));
}

static float cortex_dot_f32(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
#if defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float lane[8];
    _mm256_storeu_ps(lane, acc);
    float sum = lane[0] + lane[1] + lane[2] + lane[3] + lane[4] + lane[5] + lane[6] + lane[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

static float cortex_norm_sq_f32(const float *v, size_t n) {
    return cortex_dot_f32(v, v, n);
}

static void cortex_scale_f32(float *v, float scale, size_t n) {
    if (!v) return;
#if defined(__AVX__)
    __m256 vs = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        x = _mm256_mul_ps(x, vs);
        _mm256_storeu_ps(v + i, x);
    }
    for (; i < n; ++i) v[i] *= scale;
#else
    for (size_t i = 0; i < n; ++i) v[i] *= scale;
#endif
}

static void cortex_axpy_f32(float *dst, float a, const float *x, size_t n) {
    if (!dst || !x) return;
#if defined(__AVX__)
    __m256 va = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        d = _mm256_add_ps(d, _mm256_mul_ps(va, xv));
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) dst[i] += a * x[i];
#else
    for (size_t i = 0; i < n; ++i) dst[i] += a * x[i];
#endif
}

static int cortex_normalize_f32(float *v, size_t n) {
    if (!v || n == 0) return 0;
    float nrm_sq = cortex_norm_sq_f32(v, n);
    if (nrm_sq < 1e-12f) {
        memset(v, 0, n * sizeof(float));
        return 0;
    }
    float inv = 1.0f / (sqrtf(nrm_sq) + 1e-8f);
    cortex_scale_f32(v, inv, n);
    return 1;
}

static float cortex_cosine_f32(const float *a, const float *b, size_t n) {
    float dot = cortex_dot_f32(a, b, n);
    float na = cortex_norm_sq_f32(a, n);
    float nb = cortex_norm_sq_f32(b, n);
    float denom = sqrtf(fmaxf(na * nb, 0.0f)) + 1e-8f;
    return dot / denom;
}

static float cortex_cosine_f32_f64(const float *a, const double *b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
#if defined(__AVX__)
    __m256 dotv = _mm256_setzero_ps();
    __m256 nav = _mm256_setzero_ps();
    __m256 nbv = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_set_ps((float)b[i + 7], (float)b[i + 6], (float)b[i + 5], (float)b[i + 4],
                                  (float)b[i + 3], (float)b[i + 2], (float)b[i + 1], (float)b[i]);
        dotv = _mm256_add_ps(dotv, _mm256_mul_ps(av, bv));
        nav = _mm256_add_ps(nav, _mm256_mul_ps(av, av));
        nbv = _mm256_add_ps(nbv, _mm256_mul_ps(bv, bv));
    }
    float dlanes[8], nalanes[8], nblanes[8];
    _mm256_storeu_ps(dlanes, dotv);
    _mm256_storeu_ps(nalanes, nav);
    _mm256_storeu_ps(nblanes, nbv);
    float dot = dlanes[0] + dlanes[1] + dlanes[2] + dlanes[3] + dlanes[4] + dlanes[5] + dlanes[6] + dlanes[7];
    float na = nalanes[0] + nalanes[1] + nalanes[2] + nalanes[3] + nalanes[4] + nalanes[5] + nalanes[6] + nalanes[7];
    float nb = nblanes[0] + nblanes[1] + nblanes[2] + nblanes[3] + nblanes[4] + nblanes[5] + nblanes[6] + nblanes[7];
    for (; i < n; ++i) {
        float bi = (float)b[i];
        dot += a[i] * bi;
        na += a[i] * a[i];
        nb += bi * bi;
    }
    float denom = sqrtf(fmaxf(na * nb, 0.0f)) + 1e-8f;
    return dot / denom;
#else
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float bi = (float)b[i];
        dot += a[i] * bi;
        na += a[i] * a[i];
        nb += bi * bi;
    }
    float denom = sqrtf(fmaxf(na * nb, 0.0f)) + 1e-8f;
    return dot / denom;
#endif
}

static void cortex_project_state_f32(const CortexModule *ctx, const float *xn, double sx[2]) {
    sx[0] = 0.0;
    sx[1] = 0.0;
    if (!ctx || !ctx->memory || !xn) return;
    size_t dim = cortex_effective_dim(ctx);
    size_t proj_dim = dim < 64 ? dim : 64;
    for (size_t i = 0; i < proj_dim; ++i) {
        double xi = (double)xn[i];
        sx[0] += ctx->memory->U[0][i] * xi;
        sx[1] += ctx->memory->U[1][i] * xi;
    }
}

static int cortex_memory_max_n(const BubbleShadowMemory *mem) {
    if (!mem || mem->count == 0) return 1;
    int max_n = 1;
    for (size_t i = 0; i < mem->count; ++i) {
        if (mem->bubbles[i].n > max_n) max_n = mem->bubbles[i].n;
    }
    return max_n;
}

static double cortex_memory_strength(int n, int max_n) {
    if (n <= 0 || max_n <= 0) return 0.0;
    double denom = log(1.0 + (double)max_n);
    if (denom < 1e-9) return 0.0;
    return log(1.0 + (double)n) / denom;
}

static void cortex_push_history(CortexModule *ctx, const double *xn, size_t dim) {
    if (!ctx || !xn || !ctx->history || ctx->history_cap == 0) return;
    if (ctx->last_state && ctx->has_last_state) {
        double *slot = &ctx->history[ctx->history_pos * dim];
        for (size_t i = 0; i < dim; ++i) {
            slot[i] = xn[i] - ctx->last_state[i];
        }
        double nrm = spark_norm(slot, dim);
        if (nrm > 1e-8) {
            for (size_t i = 0; i < dim; ++i) {
                slot[i] /= nrm;
            }
        } else {
            memset(slot, 0, dim * sizeof(double));
        }
        ctx->history_pos = (ctx->history_pos + 1) % ctx->history_cap;
        if (ctx->history_len < ctx->history_cap) ctx->history_len++;
    }
    if (ctx->last_state) {
        memcpy(ctx->last_state, xn, dim * sizeof(double));
        ctx->has_last_state = 1;
    }
}

static int cortex_compute_inertia(const CortexModule *ctx, double *out, size_t dim) {
    if (!ctx || !out || !ctx->history || ctx->history_len == 0) return 0;
    memset(out, 0, dim * sizeof(double));
    double wsum = 0.0;
    for (size_t age = 0; age < ctx->history_len; ++age) {
        size_t idx = (ctx->history_pos + ctx->history_cap - 1 - age) % ctx->history_cap;
        const double *h = &ctx->history[idx * dim];
        double w = pow(SPARK_CORTEX_INERTIA_DECAY, (double)age);
        wsum += w;
        for (size_t i = 0; i < dim; ++i) {
            out[i] += w * h[i];
        }
    }
    if (wsum > 0.0) {
        for (size_t i = 0; i < dim; ++i) {
            out[i] /= wsum;
        }
    }
    double nrm = spark_norm(out, dim);
    if (nrm < 1e-8) {
        memset(out, 0, dim * sizeof(double));
        return 0;
    }
    for (size_t i = 0; i < dim; ++i) {
        out[i] /= nrm;
    }
    return 1;
}

static int cortex_eval_memory_f32(const CortexModule *ctx, const float *xn, size_t dim, int max_n, double *out_affect, double *out_mem) {
    if (!ctx || !ctx->memory || ctx->memory->count == 0 || !xn) {
        if (out_affect) *out_affect = 0.0;
        if (out_mem) *out_mem = 0.0;
        return 0;
    }
    double sx[2];
    cortex_project_state_f32(ctx, xn, sx);
    size_t best_idx = (size_t)-1;
    double best_metric_sq = 0.0;
    double best_dist_sq = 0.0;
    double best_dir = 0.0;
    for (size_t i = 0; i < ctx->memory->count; ++i) {
        const Bubble *b = &ctx->memory->bubbles[i];
        double dir = (double)cortex_cosine_f32_f64(xn, b->c, dim);
        if (dir < ctx->memory->dir_tau) continue;
        double dx = sx[0] - b->s[0];
        double dy = sx[1] - b->s[1];
        double dist_sq = dx * dx + dy * dy;
        double dir_safe = fmax(dir, 1e-3);
        double metric_sq = dist_sq / (dir_safe * dir_safe);
        if (best_idx == (size_t)-1 || metric_sq < best_metric_sq) {
            best_idx = i;
            best_metric_sq = metric_sq;
            best_dist_sq = dist_sq;
            best_dir = dir;
        }
    }
    if (best_idx == (size_t)-1) {
        if (out_affect) *out_affect = 0.0;
        if (out_mem) *out_mem = 0.0;
        return 0;
    }
    const Bubble *b = &ctx->memory->bubbles[best_idx];
    double closeness = (1.0 / (1.0 + best_dist_sq)) * fmax(0.0, best_dir);
    if (out_affect) *out_affect = b->a * closeness;
    if (out_mem) *out_mem = cortex_memory_strength(b->n, max_n) * closeness;
    return 1;
}

static void cortex_field(const float *state, const float *dir, float *out, size_t dim) {
    if (!state || !dir || !out) return;
#if defined(__AVX__)
    __m256 damp = _mm256_set1_ps(SPARK_CORTEX_DAMPING);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 s = _mm256_loadu_ps(state + i);
        __m256 d = _mm256_loadu_ps(dir + i);
        __m256 f = _mm256_sub_ps(d, _mm256_mul_ps(damp, s));
        _mm256_storeu_ps(out + i, f);
    }
    for (; i < dim; ++i) out[i] = dir[i] - SPARK_CORTEX_DAMPING * state[i];
#else
    for (size_t i = 0; i < dim; ++i) out[i] = dir[i] - SPARK_CORTEX_DAMPING * state[i];
#endif
}

static double cortex_rollout(CortexModule *ctx, const float *xn, const float *dir, double a1, int horizon_n, int max_n) {
    if (!ctx || !xn || !dir || horizon_n <= 0) return 0.0;
    size_t dim = cortex_effective_dim(ctx);
    if (!ctx->scratch_tmp_f || !ctx->scratch_mid_f || !ctx->scratch_k1_f || !ctx->scratch_k2_f || !ctx->scratch_k3_f || !ctx->scratch_k4_f) {
        return a1 * (double)horizon_n;
    }

    float *state = ctx->scratch_tmp_f;
    float *mid = ctx->scratch_mid_f;
    float *k1 = ctx->scratch_k1_f;
    float *k2 = ctx->scratch_k2_f;
    float *k3 = ctx->scratch_k3_f;
    float *k4 = ctx->scratch_k4_f;

    cortex_copy_f32(state, xn, dim);

    double step = SPARK_CORTEX_STEP_DEFAULT;
    if (ctx->memory && ctx->memory->r0 > 0.0) step = ctx->memory->r0;
    float h = (float)step;
    float h_half = 0.5f * h;
    float h_six = h / 6.0f;

    double total = 0.0;
    for (int t = 0; t < horizon_n; ++t) {
        cortex_field(state, dir, k1, dim);
        for (size_t i = 0; i < dim; ++i) mid[i] = state[i] + h_half * k1[i];

        cortex_field(mid, dir, k2, dim);
        for (size_t i = 0; i < dim; ++i) mid[i] = state[i] + h_half * k2[i];

        cortex_field(mid, dir, k3, dim);
        for (size_t i = 0; i < dim; ++i) mid[i] = state[i] + h * k3[i];

        cortex_field(mid, dir, k4, dim);
        for (size_t i = 0; i < dim; ++i) {
            state[i] += h_six * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
        }
        cortex_normalize_f32(state, dim);

        double e = a1;
        double m = 0.0;
        if (ctx->memory && ctx->memory->count > 0) {
            double eval_e = 0.0, eval_m = 0.0;
            if (cortex_eval_memory_f32(ctx, state, dim, max_n, &eval_e, &eval_m)) {
                e = eval_e;
                m = eval_m;
            }
        }
        total += e + ctx->alpha * m;
    }
    return total;
}

CortexModule *spark_cortex_create(BubbleShadowMemory *memory, AffectModule *affect_module, size_t dim) {
    CortexModule *ctx = calloc(1, sizeof(CortexModule));
    if (!ctx) return NULL;
    ctx->memory = memory;
    ctx->affect_module = affect_module;
    ctx->dim = dim;
    ctx->alpha = SPARK_CORTEX_ALPHA_DEFAULT;
    ctx->beta = SPARK_CORTEX_BETA_DEFAULT;
    ctx->history_cap = SPARK_CORTEX_HISTORY;
    size_t eff_dim = cortex_effective_dim(ctx);
    if (eff_dim > 0) {
        ctx->scratch_xn = calloc(eff_dim, sizeof(double));
        ctx->scratch_tmp = calloc(eff_dim, sizeof(double));
        ctx->scratch_dir = calloc(eff_dim, sizeof(double));
        ctx->scratch_best = calloc(eff_dim, sizeof(double));
        ctx->history = calloc(eff_dim * ctx->history_cap, sizeof(double));
        ctx->last_state = calloc(eff_dim, sizeof(double));

        ctx->scratch_xn_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_tmp_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_dir_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_best_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_mid_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_k1_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_k2_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_k3_f = calloc(eff_dim, sizeof(float));
        ctx->scratch_k4_f = calloc(eff_dim, sizeof(float));
    }
    return ctx;
}

void spark_cortex_free(CortexModule *ctx) {
    if (!ctx) return;
    free(ctx->scratch_best);
    free(ctx->scratch_dir);
    free(ctx->scratch_tmp);
    free(ctx->scratch_xn);
    free(ctx->history);
    free(ctx->last_state);

    free(ctx->scratch_xn_f);
    free(ctx->scratch_tmp_f);
    free(ctx->scratch_dir_f);
    free(ctx->scratch_best_f);
    free(ctx->scratch_mid_f);
    free(ctx->scratch_k1_f);
    free(ctx->scratch_k2_f);
    free(ctx->scratch_k3_f);
    free(ctx->scratch_k4_f);
    free(ctx);
}

void spark_cortex_reset_history(CortexModule *ctx) {
    if (!ctx) return;
    ctx->history_len = 0;
    ctx->history_pos = 0;
    ctx->has_last_state = 0;
    if (ctx->history && ctx->history_cap > 0) {
        size_t dim = cortex_effective_dim(ctx);
        if (dim > 0) {
            memset(ctx->history, 0, dim * ctx->history_cap * sizeof(double));
        }
    }
    if (ctx->last_state) {
        size_t dim = cortex_effective_dim(ctx);
        if (dim > 0) {
            memset(ctx->last_state, 0, dim * sizeof(double));
        }
    }
}

double spark_cortex_predict_trajectory(CortexModule *ctx, const double *state_vec, int horizon_n) {
    if (!ctx || !state_vec || horizon_n <= 0) return 0.0;
    size_t dim = cortex_effective_dim(ctx);
    if (dim == 0 || !ctx->scratch_xn || !ctx->scratch_dir || !ctx->scratch_best || !ctx->scratch_xn_f ||
        !ctx->scratch_tmp_f || !ctx->scratch_dir_f || !ctx->scratch_best_f || !ctx->scratch_mid_f || !ctx->scratch_k1_f ||
        !ctx->scratch_k2_f || !ctx->scratch_k3_f || !ctx->scratch_k4_f) {
        return 0.0;
    }

    double a1 = 0.0;
    if (ctx->affect_module) {
        a1 = affect_query(ctx->affect_module, state_vec, dim);
    }

    double *xn = ctx->scratch_xn;
    memcpy(xn, state_vec, dim * sizeof(double));
    double nrm = spark_norm(xn, dim);
    if (nrm < 1e-12) return 0.0;
    spark_normalize(xn, dim);

    cortex_push_history(ctx, xn, dim);
    int inertia_valid = cortex_compute_inertia(ctx, ctx->scratch_dir, dim);

    float *xn_f = ctx->scratch_xn_f;
    cortex_to_f32(xn, xn_f, dim);

    if (inertia_valid) {
        cortex_to_f32(ctx->scratch_dir, ctx->scratch_dir_f, dim);
        cortex_normalize_f32(ctx->scratch_dir_f, dim);
    } else {
        memset(ctx->scratch_dir_f, 0, dim * sizeof(float));
    }

    if (!ctx->memory || ctx->memory->count == 0) {
        if (!inertia_valid) return a1 * (double)horizon_n;
        cortex_copy_f32(ctx->scratch_best_f, ctx->scratch_dir_f, dim);
        return cortex_rollout(ctx, xn_f, ctx->scratch_best_f, a1, horizon_n, 1);
    }

    int max_n = cortex_memory_max_n(ctx->memory);
    double sx[2];
    cortex_project_state_f32(ctx, xn_f, sx);
    double best_score = -1e18;
    int best_found = 0;
    float *best_vec = ctx->scratch_best_f;

    for (size_t idx = 0; idx < ctx->memory->count; ++idx) {
        const Bubble *b = &ctx->memory->bubbles[idx];
        double dir = (double)cortex_cosine_f32_f64(xn_f, b->c, dim);
        if (dir < ctx->memory->dir_tau) continue;

        float *m = ctx->scratch_tmp_f;
        for (size_t i = 0; i < dim; ++i) {
            m[i] = (float)b->c[i] - xn_f[i];
        }
        float mnrm_sq = cortex_norm_sq_f32(m, dim);
        if (mnrm_sq < 1e-8f) {
            for (size_t i = 0; i < dim; ++i) m[i] = (float)b->c[i];
            if (!cortex_normalize_f32(m, dim)) continue;
        } else {
            float inv = 1.0f / (sqrtf(mnrm_sq) + 1e-8f);
            cortex_scale_f32(m, inv, dim);
        }

        double dx = sx[0] - b->s[0];
        double dy = sx[1] - b->s[1];
        double dist_sq = dx * dx + dy * dy;
        double closeness = 1.0 / (1.0 + dist_sq);
        double e = b->a * closeness;
        double mstrength = cortex_memory_strength(b->n, max_n) * closeness;
        double s = inertia_valid ? (double)cortex_cosine_f32(m, ctx->scratch_dir_f, dim) : 0.0;
        double score = e + ctx->alpha * mstrength + ctx->beta * s;
        if (!best_found || score > best_score) {
            best_score = score;
            cortex_copy_f32(best_vec, m, dim);
            best_found = 1;
        }
    }

    if (inertia_valid) {
        double e = a1;
        double mstrength = 0.0;
        float *tmp = ctx->scratch_tmp_f;
        cortex_copy_f32(tmp, xn_f, dim);
        float step = (ctx->memory->r0 > 0.0) ? (float)ctx->memory->r0 : (float)SPARK_CORTEX_STEP_DEFAULT;
        cortex_axpy_f32(tmp, step, ctx->scratch_dir_f, dim);
        cortex_normalize_f32(tmp, dim);

        double eval_e = 0.0, eval_m = 0.0;
        if (cortex_eval_memory_f32(ctx, tmp, dim, max_n, &eval_e, &eval_m)) {
            e = eval_e;
            mstrength = eval_m;
        }
        double score = e + ctx->alpha * mstrength + ctx->beta;
        if (!best_found || score > best_score) {
            best_score = score;
            cortex_copy_f32(best_vec, ctx->scratch_dir_f, dim);
            best_found = 1;
        }
    }

    if (!best_found) return a1 * (double)horizon_n;
    return cortex_rollout(ctx, xn_f, best_vec, a1, horizon_n, max_n);
}
