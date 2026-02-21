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

// Spark_Core/spark_config.h
// C translation of Spark configuration constants.
#ifndef SPARK_CONFIG_H
#define SPARK_CONFIG_H

#include <stddef.h>

typedef struct {
    // Dimensions
    size_t INPUT_DIM;
    size_t WORKING_DIM;

    // Curiosity weights
    double CUR_LOCAL;
    double CUR_GLOBAL;
    double CUR_PROTO;

    // Dynamic/plasticity
    double DYN_BASE_ALPHA;
    double DYN_FATIGUE_DECAY;
    int DYN_NEIGHBORHOOD;
    double DYN_INIT_CONNECTIVITY;
    double DYN_THRESHOLD;

    // Memory unified
    size_t GLOBAL_MAXLEN;

    // Affect
    size_t AFFECT_BUF;
    double AFFECT_DECAY_TAU;
    double AFFECT_MIN_SIM;

    // Dreamer
    int DREAM_MAX;

    // Safety / debug
    int SAFE_NUMPY;
} SparkConfig;

static inline SparkConfig spark_default_config(void) {
    SparkConfig cfg;
    cfg.INPUT_DIM = 20;
    cfg.WORKING_DIM = 30;
    cfg.CUR_LOCAL = 0.5;
    cfg.CUR_GLOBAL = 0.3;
    cfg.CUR_PROTO = 0.2;
    cfg.DYN_BASE_ALPHA = 0.1;
    cfg.DYN_FATIGUE_DECAY = 0.95;
    cfg.DYN_NEIGHBORHOOD = 2;
    cfg.DYN_INIT_CONNECTIVITY = 0.3;
    cfg.DYN_THRESHOLD = 0.01;
    cfg.GLOBAL_MAXLEN = 200;
    cfg.AFFECT_BUF = 50;
    cfg.AFFECT_DECAY_TAU = 5.0;
    cfg.AFFECT_MIN_SIM = 0.1;
    cfg.DREAM_MAX = 5;
    cfg.SAFE_NUMPY = 1;
    return cfg;
}

#endif // SPARK_CONFIG_H
