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


// Spark_Core/spark_utils.h
// Shared utilities (vector ops, RNG).
#ifndef SPARK_UTILS_H
#define SPARK_UTILS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void spark_set_seed(uint32_t seed) {
    srand(seed);
}

static inline double spark_rand_uniform(void) {
    return (double)rand() / (double)RAND_MAX;
}

static inline double spark_norm(const double *v, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += v[i] * v[i];
    }
    return sqrt(acc);
}

static inline void spark_normalize(double *v, size_t n) {
    double nrm = spark_norm(v, n) + 1e-8;
    for (size_t i = 0; i < n; ++i) {
        v[i] /= nrm;
    }
}

static inline double spark_cosine_similarity(const double *a, const double *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    double denom = sqrt(na) * sqrt(nb) + 1e-8;
    return dot / denom;
}

static inline double spark_clip(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline double spark_gauss(double mean, double stddev) {
    // Box-Muller
    double u1 = spark_rand_uniform();
    double u2 = spark_rand_uniform();
    double z0 = sqrt(-2.0 * log(fmax(u1, 1e-12))) * cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}

static inline void spark_zero(double *v, size_t n) {
    memset(v, 0, n * sizeof(double));
}

#endif // SPARK_UTILS_H
