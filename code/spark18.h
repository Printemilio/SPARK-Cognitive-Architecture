// Spark_Core/spark18.h
// C translation of Spark V18 core modules.
#ifndef SPARK18_H
#define SPARK18_H

#include <stddef.h>

#include "cg18.h"
#include "qb16.h"
#include "spark_config.h"

typedef struct {
    double *c; // center
    double r;  // shadow radius
    double s[2]; // projected center
    int n;
    double a; // opacity
    double S[4]; // 2x2 covariance stored row-major
} Bubble;

typedef struct {
    size_t d;
    double U[2][64]; // max projection size support up to 64 dims by default
    Bubble *bubbles;
    size_t count;
    size_t capacity;
    double r0;
    double match_tau;
    double dir_tau;
    double emotion_gain;
    double split_emotion;
    size_t max_bubbles;
    long stores;
    long recalls;
} BubbleShadowMemory;

typedef struct {
    size_t dim;
    BubbleShadowMemory *mem;
} MemoryShim;

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
void bubble_memory_forget(BubbleShadowMemory *m, double decay, double min_radius, double min_opacity);
void bubble_memory_merge_close(BubbleShadowMemory *m, double shadow_thresh, double dir_thresh);
void bubble_memory_stats(BubbleShadowMemory *m, long *stores, long *recalls, size_t *count);

MemoryShim *memory_shim_create(size_t dim, size_t max_bubbles);
void memory_shim_free(MemoryShim *shim);
void memory_shim_store(MemoryShim *shim, const double *x, size_t dim, double utility, double emotion);
double memory_shim_project(MemoryShim *shim, const double *x, size_t dim);
void memory_shim_forget(MemoryShim *shim, double decay, double min_radius, double min_opacity);
void memory_shim_merge_close(MemoryShim *shim, double shadow_thresh, double dir_thresh);

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

#endif // SPARK18_H
