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

// Spark_Core/spark19.h
#ifndef SPARK19_H
#define SPARK19_H

#include <stddef.h>

#include "cg19.h"
#include "qb16.h"
#include "spark_config.h"

typedef struct {
    double *c; // center
    double r;  // shadow radius
    double s[2]; // projected center
    int n;
    double a; // opacity
    double dr; // reward trend (slope proxy)
    double S[4]; // 2x2 covariance stored row-major
} Bubble;

typedef struct {
    int index;
    int left;
    int right;
    int axis;
    double bbox_min[3];
    double bbox_max[3];
} KDNode;

typedef struct {
    size_t d;
    double U[2][64]; // max projection size support up to 64 dims by default
    Bubble *bubbles;
    size_t count;
    size_t capacity;
    KDNode *kd_nodes;
    size_t kd_count;
    int kd_root;
    int kd_dirty;
    double r0;
    double match_tau;
    double dir_tau;
    double emotion_gain;
    double split_emotion;
    size_t max_bubbles;
    double *scratch_v;
    size_t scratch_dim;
    long stores;
    long recalls;
} BubbleShadowMemory;

typedef struct {
    size_t dim;
    BubbleShadowMemory *mem;
} MemoryShim;

typedef struct {
    size_t node_id;
    size_t tick;
} DeferredActivationEvent;

typedef struct {
    DeferredActivationEvent *events;
    size_t cap;
    size_t len;
    size_t head;
    size_t latest_tick;
    double temporal_tau;
    double goal_floor;
    double connectivity_bonus;
    double credit_persistence;
} DeferredRewardModule;

typedef struct {
    double *memory;   // ring buffer flattened
    size_t buffer_len;
    size_t size;
    MemoryShim *global_memory;
    double local_w;
    double global_w;
    double proto_w;
    double *prototypes;
    size_t proto_count;
    double *novelty_history;
    size_t novelty_len;
    size_t novelty_cap;
    double threshold_min;
    double *scratch_xn;
    double *scratch_mean_sim;
    size_t scratch_mean_cap;
} CuriosityModule;

typedef struct {
    size_t size;
    double *W;
    int *mask;
    double *usage;
    double alpha;
    double base_alpha;
    double fatigue_decay;
    double threshold;
    int neighborhood;
    double *memory_buffer;
    double *tmp_xn;   // workspace: normalized input
    double *tmp_pred; // workspace: prediction buffer
    size_t memory_len;
    double *recent_activations;
    size_t recent_len;
    double uncertainty;
    double energy;
    double confidence;
    int usage_count;
    double *error_history;
    size_t error_len;
    size_t error_cap;
    double last_dj_mod;
    QuantumSignalInterface *q;
} DynamicModule;

typedef struct {
    double *affect_buffer;
    size_t buf_len;
    double decay_tau;
    double min_similarity;
    size_t dim;
    CognitiveGraph *graph;
    DeferredRewardModule *deferred_reward;
    MemoryShim *global_memory;
    double *stim_buffer;   // flattened stimuli history (buf_len x dim)
    size_t *stim_time;     // timestamps for stimuli
    size_t stim_len;
    double *reward_buffer; // recent rewards
    size_t reward_len;
    size_t time;
    double maturation;
    double matur_rate;
    double running_valence;
    double reward_decay;
    double pending_reward;
    QuantumSignalInterface *q;
    double *scratch_xn;
    double *scratch_weights;
    size_t scratch_weights_cap;
} AffectModule;

typedef struct {
    AffectModule *affect_module;
    MemoryShim *global_memory;
} AffectivePredictor;

typedef struct {
    CuriosityModule *curiosity;
    DynamicModule *dynamic_module;
    AffectModule *affect_module;
    MemoryShim *global_memory;
    QuantumSignalInterface *qiface;
    AffectivePredictor *affective_predictor;
    DeferredRewardModule *deferred_reward;
    CognitiveGraph *graph;
    SparkConfig cfg;
} SparkSystem;

// Utilities
void normalize_vec(double *v, size_t n);
double cosine_similarity_vec(const double *a, const double *b, size_t n);

// Memory
BubbleShadowMemory *bubble_memory_create(size_t dim, double r0, double match_tau, double dir_tau, double emotion_gain, double split_emotion, size_t max_bubbles);
void bubble_memory_free(BubbleShadowMemory *m);
void bubble_memory_store(BubbleShadowMemory *m, const double *x, size_t dim, double emotion);
double bubble_memory_project(BubbleShadowMemory *m, const double *x, size_t dim);
void bubble_memory_feedback(BubbleShadowMemory *m, const double *x, size_t dim, double reward);
void bubble_memory_forget(BubbleShadowMemory *m, double decay, double min_radius, double min_opacity);
void bubble_memory_merge_close(BubbleShadowMemory *m, double shadow_thresh, double dir_thresh);
void bubble_memory_stats(BubbleShadowMemory *m, long *stores, long *recalls, size_t *count);

MemoryShim *memory_shim_create(size_t dim, size_t max_bubbles);
void memory_shim_free(MemoryShim *shim);
void memory_shim_store(MemoryShim *shim, const double *x, size_t dim, double utility, double emotion);
double memory_shim_project(MemoryShim *shim, const double *x, size_t dim);
void memory_shim_feedback(MemoryShim *shim, const double *x, size_t dim, double reward);
void memory_shim_forget(MemoryShim *shim, double decay, double min_radius, double min_opacity);
void memory_shim_merge_close(MemoryShim *shim, double shadow_thresh, double dir_thresh);

// Deferred reward
DeferredRewardModule *deferred_reward_create(size_t history_cap, double temporal_tau, double connectivity_bonus, double credit_persistence);
void deferred_reward_free(DeferredRewardModule *m);
void deferred_reward_record_batch(DeferredRewardModule *m, const size_t *active_ids, size_t count, size_t tick);
void deferred_reward_apply(DeferredRewardModule *m, CognitiveGraph *graph, double reward, const double *final_stimulus, size_t dim);

// Modules
CuriosityModule *curiosity_create(size_t size, MemoryShim *global_memory, const SparkConfig *cfg);
void curiosity_free(CuriosityModule *c);
double curiosity_compute(CuriosityModule *c, const double *x);
void curiosity_update_memory(CuriosityModule *c, const double *x);

DynamicModule *dynamic_create(size_t size, const SparkConfig *cfg, QuantumSignalInterface *q);
void dynamic_free(DynamicModule *d);
void dynamic_update(DynamicModule *d, const double *x, size_t dim, double affect, double cognitive_load);

AffectModule *affect_create(const SparkConfig *cfg, QuantumSignalInterface *q);
void affect_free(AffectModule *a);
void affect_register_feedback(AffectModule *a, double reward, const double *state, size_t dim);
double affect_query(AffectModule *a, const double *x, size_t dim);

AffectivePredictor *affective_predictor_create(AffectModule *a, MemoryShim *mem);
void affective_predictor_free(AffectivePredictor *p);

// Graph building
SparkSystem *build_spark_system(int seed);
void spark_system_free(SparkSystem *sys);
void inject_and_propagate_c(SparkSystem *sys, size_t *indices, double **patterns, size_t count, int steps);
size_t spark_graph_num_nodes(const SparkSystem *sys);
size_t spark_graph_state_dim(const SparkSystem *sys);
void spark_graph_snapshot(const SparkSystem *sys, double *out_matrix); // out size num_nodes*state_dim
void spark_global_memory_stats(const SparkSystem *sys, long *stores, long *recalls, size_t *count);

#endif // SPARK19_H
