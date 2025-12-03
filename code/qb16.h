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

// Spark_Core/qb16.h
#ifndef QB16_H
#define QB16_H

#include <stddef.h>

typedef struct {
    int NUM_QUBITS;
    double MIN_FREQ;
    double MAX_FREQ;
    int HISTORY_LEN;
    int PRINT_LOOP;
} QuantumConfig;

typedef struct {
    QuantumConfig cfg;
    int *states;
    int running;
    void **threads;
    size_t thread_count;
    int stop_flag;
    int **history;
    int history_len;
    int history_cap;
} QuantumSignalInterface;

QuantumSignalInterface *qiface_create(QuantumConfig cfg);
void qiface_free(QuantumSignalInterface *q);
void qiface_start(QuantumSignalInterface *q);
void qiface_stop(QuantumSignalInterface *q);
void qiface_get_state_snapshot(QuantumSignalInterface *q, int *out_states);

// pseudo algorithms
int qiface_pseudo_qft_multiwindow(QuantumSignalInterface *q, int windows, int step, double ***out_mag, double ***out_phase, int *out_windows, int *out_bins);
int qiface_pseudo_deutsch_jozsa(QuantumSignalInterface *q, int (*oracle)(const int *, int), int trials, const char **out_label, int *out_value);
int qiface_pseudo_amplitude_estimation(QuantumSignalInterface *q, const char *pattern, int trials, double *constructive, double *destructive);

#endif // QB16_H
