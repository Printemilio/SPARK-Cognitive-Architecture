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


// Spark_Core/spark_bridge.h
// Sensory bridge acting as a thalamus for SPARK: dynamic retina + feature extraction.
#ifndef SPARK_BRIDGE_H
#define SPARK_BRIDGE_H

#include <stddef.h>

#include "spark19.h"

typedef struct {
    SparkSystem *sys;
    size_t frame_width;
    size_t frame_height;
    size_t patch_size;    // sqrt(state_dim) sampled size
    size_t retina_w;      // number of patches horizontally
    size_t retina_h;      // number of patches vertically
    size_t feature_dim;   // per-patch feature vector length
    size_t visual_offset; // node index where visual cortex starts
    size_t scalar_base;   // base index for peripheral scalar nodes
    int *visual_ids;      // node ids last used for visual patches
    size_t visual_ids_len;
    size_t visual_ids_cap;
    double *prev_frame;
    size_t prev_frame_len;
    int has_prev;
    struct SparkBridgeOutputNas *motor_nas;
} SparkBridge;

typedef struct {
    size_t num_channels;
    size_t inputs_per_channel;
    size_t population;
    size_t iterations;
    size_t pool_size;
    size_t elite;
    double mutation_rate;
    double mutation_sigma;
    double activity_weight;
    double affect_weight;
    double curiosity_weight;
    double signal_gain;
} SparkBridgeOutputConfig;

// ---------- Inputs ----------
// Build a bridge with retina sized from state_dim/num_nodes and frame geometry.
// visual_offset sets where to start injecting into the graph.
SparkBridge *spark_bridge_create(SparkSystem *sys, size_t frame_width, size_t frame_height, size_t visual_offset);
void spark_bridge_free(SparkBridge *bridge);
void spark_bridge_reset_history(SparkBridge *bridge);
size_t spark_bridge_patch_count(const SparkBridge *bridge);
void spark_bridge_inject_frame(SparkBridge *bridge, const double *frame);
void spark_bridge_inject_frame_with_ids(SparkBridge *bridge, const double *frame, int *out_ids, size_t out_len);
void spark_bridge_inject_scalars(SparkBridge *bridge, double pain, double pleasure, double sound);

// ---------- Outputs ----------
SparkBridgeOutputConfig spark_bridge_motor_nas_default_config(size_t num_channels);
void spark_bridge_enable_motor_nas(SparkBridge *bridge, const SparkBridgeOutputConfig *cfg);
void spark_bridge_disable_motor_nas(SparkBridge *bridge);
void spark_bridge_resolve_outputs(SparkBridge *bridge, double *out_signals, size_t signal_len);

#endif // SPARK_BRIDGE_H
