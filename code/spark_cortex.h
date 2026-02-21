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

// spark_cortex.h

#ifndef SPARK_CORTEX_H
#define SPARK_CORTEX_H

#include <stddef.h>

#include "spark19.h"

typedef struct {
    BubbleShadowMemory *memory;
    AffectModule *affect_module;
    size_t dim;
    double *scratch_xn;
    double *scratch_tmp;
    double *scratch_dir;
    double *scratch_best;
    double *history;
    size_t history_cap;
    size_t history_len;
    size_t history_pos;
    double *last_state;
    int has_last_state;
    float *scratch_xn_f;
    float *scratch_tmp_f;
    float *scratch_dir_f;
    float *scratch_best_f;
    float *scratch_mid_f;
    float *scratch_k1_f;
    float *scratch_k2_f;
    float *scratch_k3_f;
    float *scratch_k4_f;
    double alpha;
    double beta;
} CortexModule;

CortexModule *spark_cortex_create(BubbleShadowMemory *memory, AffectModule *affect_module, size_t dim);
void spark_cortex_free(CortexModule *ctx);
void spark_cortex_reset_history(CortexModule *ctx);

double spark_cortex_predict_trajectory(CortexModule *ctx, const double *state_vec, int horizon_n);

#endif
