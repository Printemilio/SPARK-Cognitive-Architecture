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



// Spark_Core/spark_bridge.c
// Sensory bridge acting as a thalamus for SPARK. It scales retinal density
// from available cognitive capacity and injects V1-like feature vectors.
#include "spark_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spark_utils.h"

#define SPARK_BRIDGE_FEATURES 4

typedef struct SparkBridgeOutputNas {
    SparkBridgeOutputConfig cfg;
    uint32_t rng_state;
    int has_spare;
    double spare;
    size_t last_pool_size;
    size_t last_num_nodes;
    int initialized;

    double *node_activity;
    double *node_score;
    int *node_order;
    int *pool_indices;

    int *conn_indices;
    double *conn_weights;
    int *next_indices;
    double *next_weights;
    double *arch_scores;
    int *arch_order;

    int *best_indices;
    double *best_weights;
} SparkBridgeOutputNas;

static uint32_t nas_xorshift32(SparkBridgeOutputNas *nas) {
    uint32_t x = nas->rng_state ? nas->rng_state : 1u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    nas->rng_state = x;
    return x;
}

static double nas_rand_uniform(SparkBridgeOutputNas *nas) {
    return (double)nas_xorshift32(nas) / (double)UINT32_MAX;
}

static double nas_rand_normal(SparkBridgeOutputNas *nas) {
    if (nas->has_spare) {
        nas->has_spare = 0;
        return nas->spare;
    }
    double u1 = nas_rand_uniform(nas);
    double u2 = nas_rand_uniform(nas);
    if (u1 < 1e-12) u1 = 1e-12;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    nas->spare = r * sin(theta);
    nas->has_spare = 1;
    return r * cos(theta);
}

static size_t nas_total_connections(const SparkBridgeOutputNas *nas) {
    return nas->cfg.num_channels * nas->cfg.inputs_per_channel;
}

static int ensure_node_buffers(SparkBridgeOutputNas *nas, size_t num_nodes) {
    if (num_nodes == 0) return 1;
    if (nas->last_num_nodes == num_nodes && nas->node_activity) return 1;
    free(nas->node_activity);
    free(nas->node_score);
    free(nas->node_order);
    nas->node_activity = calloc(num_nodes, sizeof(double));
    nas->node_score = calloc(num_nodes, sizeof(double));
    nas->node_order = calloc(num_nodes, sizeof(int));
    if (!nas->node_activity || !nas->node_score || !nas->node_order) {
        return 0;
    }
    nas->last_num_nodes = num_nodes;
    return 1;
}

static int ensure_pool_buffers(SparkBridgeOutputNas *nas, size_t pool_size) {
    if (pool_size == 0) return 1;
    if (nas->last_pool_size == pool_size && nas->pool_indices) return 1;
    free(nas->pool_indices);
    nas->pool_indices = calloc(pool_size, sizeof(int));
    if (!nas->pool_indices) return 0;
    nas->last_pool_size = pool_size;
    nas->initialized = 0;
    return 1;
}

static int ensure_population_buffers(SparkBridgeOutputNas *nas) {
    size_t pop = nas->cfg.population;
    size_t conns = nas_total_connections(nas);
    size_t total = pop * conns;
    free(nas->conn_indices);
    free(nas->conn_weights);
    free(nas->next_indices);
    free(nas->next_weights);
    free(nas->arch_scores);
    free(nas->arch_order);
    free(nas->best_indices);
    free(nas->best_weights);

    nas->conn_indices = calloc(total, sizeof(int));
    nas->conn_weights = calloc(total, sizeof(double));
    nas->next_indices = calloc(total, sizeof(int));
    nas->next_weights = calloc(total, sizeof(double));
    nas->arch_scores = calloc(pop, sizeof(double));
    nas->arch_order = calloc(pop, sizeof(int));
    nas->best_indices = calloc(conns, sizeof(int));
    nas->best_weights = calloc(conns, sizeof(double));
    if (!nas->conn_indices || !nas->conn_weights || !nas->next_indices ||
        !nas->next_weights || !nas->arch_scores || !nas->arch_order ||
        !nas->best_indices || !nas->best_weights) {
        return 0;
    }
    nas->initialized = 0;
    return 1;
}

static SparkBridgeOutputNas *spark_bridge_motor_nas_create(const SparkBridgeOutputConfig *cfg) {
    if (!cfg || cfg->num_channels == 0) return NULL;
    SparkBridgeOutputNas *nas = calloc(1, sizeof(SparkBridgeOutputNas));
    if (!nas) return NULL;
    nas->cfg = *cfg;
    if (nas->cfg.inputs_per_channel == 0) nas->cfg.inputs_per_channel = 4;
    if (nas->cfg.population == 0) nas->cfg.population = 64;
    if (nas->cfg.elite == 0 || nas->cfg.elite > nas->cfg.population) {
        nas->cfg.elite = nas->cfg.population / 2;
    }
    if (nas->cfg.pool_size == 0) nas->cfg.pool_size = 32;
    if (nas->cfg.iterations == 0) nas->cfg.iterations = 4;
    if (nas->cfg.mutation_sigma <= 0.0) nas->cfg.mutation_sigma = 0.1;
    nas->rng_state = 1u;
    if (!ensure_population_buffers(nas)) {
        free(nas);
        return NULL;
    }
    return nas;
}

static void spark_bridge_motor_nas_free(SparkBridgeOutputNas *nas) {
    if (!nas) return;
    free(nas->node_activity);
    free(nas->node_score);
    free(nas->node_order);
    free(nas->pool_indices);
    free(nas->conn_indices);
    free(nas->conn_weights);
    free(nas->next_indices);
    free(nas->next_weights);
    free(nas->arch_scores);
    free(nas->arch_order);
    free(nas->best_indices);
    free(nas->best_weights);
    free(nas);
}

static void init_population(SparkBridgeOutputNas *nas, size_t pool_size) {
    size_t pop = nas->cfg.population;
    size_t conns = nas_total_connections(nas);
    for (size_t i = 0; i < pop; ++i) {
        for (size_t c = 0; c < conns; ++c) {
            size_t idx = i * conns + c;
            nas->conn_indices[idx] = (int)(nas_rand_uniform(nas) * (double)pool_size);
            nas->conn_weights[idx] = nas_rand_uniform(nas) * 2.0 - 1.0;
        }
    }
    nas->initialized = 1;
}

static void compute_node_scores(SparkBridgeOutputNas *nas, const SparkBridge *bridge, const SparkSystem *sys, const char *exclude_mask) {
    CognitiveGraph *g = sys->graph;
    size_t num_nodes = g->num_nodes;
    size_t dim = g->state_dim;
    double aw = nas->cfg.activity_weight;
    double affw = nas->cfg.affect_weight;
    double curw = nas->cfg.curiosity_weight;
    size_t visual_limit = 0;
    if (bridge) {
        visual_limit = bridge->visual_offset + (bridge->retina_w * bridge->retina_h);
    }
    for (size_t i = 0; i < num_nodes; ++i) {
        if (exclude_mask && exclude_mask[i]) {
            nas->node_activity[i] = 0.0;
            nas->node_score[i] = -1.0;
            nas->node_order[i] = (int)i;
            continue;
        }
        Node *n = &g->nodes[i];
        double acc = 0.0;
        if (dim > 0 && n->state) {
            for (size_t k = 0; k < dim; ++k) acc += fabs(n->state[k]);
            acc /= (double)dim;
        }
        nas->node_activity[i] = acc;
        double score = aw * acc + affw * fabs(n->affect) + curw * fabs(n->curiosity);
        if (i < visual_limit) score *= 0.1;
        nas->node_score[i] = score;
        nas->node_order[i] = (int)i;
    }
}

static void mark_excluded(char *mask, size_t num_nodes, size_t *count, int id) {
    if (!mask || id < 0) return;
    size_t idx = (size_t)id;
    if (idx >= num_nodes) return;
    if (!mask[idx]) {
        mask[idx] = 1;
        if (count) (*count)++;
    }
}

static char *build_exclusion_mask(const SparkBridge *bridge, const CognitiveGraph *g, size_t *out_excluded) {
    if (!bridge || bridge->visual_ids_len == 0 || !g || g->num_nodes == 0) return NULL;
    char *mask = calloc(g->num_nodes, sizeof(char));
    if (!mask) return NULL;
    size_t excluded = 0;

    for (size_t i = 0; i < bridge->visual_ids_len; ++i) {
        mark_excluded(mask, g->num_nodes, &excluded, bridge->visual_ids[i]);
    }
    for (size_t i = 0; i < bridge->visual_ids_len; ++i) {
        int vid = bridge->visual_ids[i];
        if (vid < 0 || (size_t)vid >= g->num_nodes) continue;
        Node *node = &g->nodes[(size_t)vid];
        for (size_t k = 0; k < node->outgoing.count; ++k) {
            mark_excluded(mask, g->num_nodes, &excluded, node->outgoing.ids[k]);
        }
        for (size_t k = 0; k < node->incoming.count; ++k) {
            mark_excluded(mask, g->num_nodes, &excluded, node->incoming.ids[k]);
        }
    }

    if (out_excluded) *out_excluded = excluded;
    return mask;
}

static const double *g_sort_scores = NULL;

static int score_cmp_desc(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double da = g_sort_scores[ia];
    double db = g_sort_scores[ib];
    if (da < db) return 1;
    if (da > db) return -1;
    return 0;
}

static void sort_indices_desc(int *indices, size_t count, const double *scores) {
    g_sort_scores = scores;
    qsort(indices, count, sizeof(int), score_cmp_desc);
}

static void evaluate_population(SparkBridgeOutputNas *nas, size_t pool_size) {
    size_t pop = nas->cfg.population;
    size_t conns = nas_total_connections(nas);
    for (size_t i = 0; i < pop; ++i) {
        double score = 0.0;
        size_t base = i * conns;
        for (size_t c = 0; c < conns; ++c) {
            int pool_idx = nas->conn_indices[base + c];
            if (pool_idx < 0 || (size_t)pool_idx >= pool_size) {
                pool_idx = pool_idx < 0 ? 0 : (int)(pool_size - 1);
                nas->conn_indices[base + c] = pool_idx;
            }
            int node_id = nas->pool_indices[pool_idx];
            double weight = nas->conn_weights[base + c];
            score += fabs(weight) * nas->node_score[node_id];
        }
        nas->arch_scores[i] = score;
        nas->arch_order[i] = (int)i;
    }
    sort_indices_desc(nas->arch_order, pop, nas->arch_scores);
}

static void evolve_population(SparkBridgeOutputNas *nas, size_t pool_size) {
    size_t pop = nas->cfg.population;
    size_t elite = nas->cfg.elite;
    size_t conns = nas_total_connections(nas);
    double rate = nas->cfg.mutation_rate;
    double sigma = nas->cfg.mutation_sigma;

    for (size_t i = 0; i < elite; ++i) {
        int arch = nas->arch_order[i];
        size_t src = (size_t)arch * conns;
        size_t dst = i * conns;
        memcpy(nas->next_indices + dst, nas->conn_indices + src, sizeof(int) * conns);
        memcpy(nas->next_weights + dst, nas->conn_weights + src, sizeof(double) * conns);
    }

    for (size_t i = elite; i < pop; ++i) {
        int base_arch = nas->arch_order[i % elite];
        size_t src = (size_t)base_arch * conns;
        size_t dst = i * conns;
        for (size_t c = 0; c < conns; ++c) {
            int idx = nas->conn_indices[src + c];
            double w = nas->conn_weights[src + c];
            if (nas_rand_uniform(nas) < rate) {
                idx = (int)(nas_rand_uniform(nas) * (double)pool_size);
            }
            if (nas_rand_uniform(nas) < rate) {
                w += nas_rand_normal(nas) * sigma;
            }
            w = spark_clip(w, -1.0, 1.0);
            nas->next_indices[dst + c] = idx;
            nas->next_weights[dst + c] = w;
        }
    }

    int *tmp_i = nas->conn_indices;
    double *tmp_w = nas->conn_weights;
    nas->conn_indices = nas->next_indices;
    nas->conn_weights = nas->next_weights;
    nas->next_indices = tmp_i;
    nas->next_weights = tmp_w;
}

static void capture_best(SparkBridgeOutputNas *nas) {
    size_t conns = nas_total_connections(nas);
    int best_arch = nas->arch_order[0];
    size_t src = (size_t)best_arch * conns;
    memcpy(nas->best_indices, nas->conn_indices + src, sizeof(int) * conns);
    memcpy(nas->best_weights, nas->conn_weights + src, sizeof(double) * conns);
}

static size_t clamp_size(size_t v, size_t lo, size_t hi) {
    if (v < lo) return lo;
    return v > hi ? hi : v;
}

static size_t compute_patch_size(size_t state_dim, size_t frame_w, size_t frame_h) {
    size_t base = (size_t)floor(sqrt((double)(state_dim > 0 ? state_dim : 1)));
    if (base < 1) base = 1;
    size_t max_side = frame_w < frame_h ? frame_w : frame_h;
    if (max_side < 1) max_side = 1;
    if (base > max_side) base = max_side;
    return base;
}

static int ensure_visual_capacity(SparkBridge *bridge, size_t need) {
    if (!bridge) return 0;
    if (need <= bridge->visual_ids_cap) return 1;
    size_t cap = bridge->visual_ids_cap ? bridge->visual_ids_cap * 2 : 8;
    while (cap < need) cap *= 2;
    int *ids = realloc(bridge->visual_ids, cap * sizeof(int));
    if (!ids) return 0;
    bridge->visual_ids = ids;
    bridge->visual_ids_cap = cap;
    return 1;
}

static void record_visual_id(SparkBridge *bridge, size_t idx, int node_id) {
    if (!bridge || node_id < 0) return;
    if (!ensure_visual_capacity(bridge, idx + 1)) return;
    bridge->visual_ids[idx] = node_id;
    if (idx + 1 > bridge->visual_ids_len) bridge->visual_ids_len = idx + 1;
}

static void compute_retina_geometry(SparkBridge *bridge) {
    CognitiveGraph *g = bridge->sys->graph;
    double capacity_ratio = g->state_dim > 0 ? ((double)g->state_dim / (double)g->num_nodes) : 0.0;
    double density = sqrt(fmax(capacity_ratio, 1e-6)); // more capacity -> denser retina

    size_t min_side = 0;
    const char *min_env = getenv("SPARK_RETINA_MIN");
    if (min_env && min_env[0]) {
        long v = strtol(min_env, NULL, 10);
        if (v > 0) min_side = (size_t)v;
    }

    size_t max_w = bridge->frame_width / bridge->patch_size;
    size_t max_h = bridge->frame_height / bridge->patch_size;
    if (max_w < 1) max_w = 1;
    if (max_h < 1) max_h = 1;

    size_t desired_w = (size_t)ceil((double)max_w * density);
    size_t desired_h = (size_t)ceil((double)max_h * density);
    desired_w = clamp_size(desired_w, min_side > 0 ? min_side : 1, max_w);
    desired_h = clamp_size(desired_h, min_side > 0 ? min_side : 1, max_h);

    size_t available_nodes = g->num_nodes > bridge->visual_offset ? g->num_nodes - bridge->visual_offset : 0;
    size_t reserved_scalar = 3;
    size_t max_patches = available_nodes > reserved_scalar ? available_nodes - reserved_scalar : 0;
    size_t desired_patches = desired_w * desired_h;

    if (max_patches == 0) {
        bridge->retina_w = bridge->retina_h = 0;
        return;
    }

    if (desired_patches > max_patches) {
        double shrink = sqrt((double)max_patches / (double)desired_patches);
        size_t new_w = (size_t)floor((double)desired_w * shrink);
        size_t new_h = (size_t)floor((double)desired_h * shrink);
        new_w = clamp_size(new_w, min_side > 0 ? min_side : 1, max_w);
        new_h = clamp_size(new_h, min_side > 0 ? min_side : 1, max_h);
        while (new_w * new_h > max_patches && new_w > 1) --new_w;
        while (new_w * new_h > max_patches && new_h > 1) --new_h;
        bridge->retina_w = new_w;
        bridge->retina_h = new_h;
    } else {
        bridge->retina_w = desired_w;
        bridge->retina_h = desired_h;
    }
}

SparkBridge *spark_bridge_create(SparkSystem *sys, size_t frame_width, size_t frame_height, size_t visual_offset) {
    if (!sys || !sys->graph || frame_width == 0 || frame_height == 0) return NULL;
    SparkBridge *b = calloc(1, sizeof(SparkBridge));
    b->sys = sys;
    b->frame_width = frame_width;
    b->frame_height = frame_height;
    b->visual_offset = visual_offset;
    b->feature_dim = SPARK_BRIDGE_FEATURES;
    b->patch_size = compute_patch_size(sys->graph->state_dim, frame_width, frame_height);
    compute_retina_geometry(b);
    b->scalar_base = b->visual_offset + (b->retina_w * b->retina_h);
    b->visual_ids = NULL;
    b->visual_ids_len = 0;
    b->visual_ids_cap = 0;
    if (b->retina_w * b->retina_h > 0) {
        ensure_visual_capacity(b, b->retina_w * b->retina_h);
    }
    b->prev_frame_len = frame_width * frame_height;
    b->prev_frame = calloc(b->prev_frame_len, sizeof(double));
    b->has_prev = 0;
    b->motor_nas = NULL;
    return b;
}

void spark_bridge_free(SparkBridge *bridge) {
    if (!bridge) return;
    spark_bridge_motor_nas_free(bridge->motor_nas);
    free(bridge->visual_ids);
    free(bridge->prev_frame);
    free(bridge);
}

void spark_bridge_reset_history(SparkBridge *bridge) {
    if (!bridge) return;
    bridge->has_prev = 0;
    bridge->visual_ids_len = 0;
    if (bridge->prev_frame) spark_zero(bridge->prev_frame, bridge->prev_frame_len);
}

size_t spark_bridge_patch_count(const SparkBridge *bridge) {
    if (!bridge) return 0;
    return bridge->retina_w * bridge->retina_h;
}

// ---------- Inputs ----------
static void extract_patch_features(const SparkBridge *bridge, const double *frame, const double *prev_frame, size_t x0, size_t y0, double *out) {
    size_t W = bridge->frame_width;
    size_t H = bridge->frame_height;
    size_t ps = bridge->patch_size;
    const double sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const double sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    double grad_x = 0.0, grad_y = 0.0, dog = 0.0, motion = 0.0;
    size_t samples = 0;
    for (size_t y = 1; y + 1 < ps && (y0 + y + 1) < H; ++y) {
        for (size_t x = 1; x + 1 < ps && (x0 + x + 1) < W; ++x) {
            size_t gx = x0 + x;
            size_t gy = y0 + y;
            double window[9];
            size_t idx = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    size_t px = gx + (size_t)kx;
                    size_t py = gy + (size_t)ky;
                    window[idx++] = frame[py * W + px];
                }
            }
            for (size_t k = 0; k < 9; ++k) {
                grad_x += sobel_x[k] * window[k];
                grad_y += sobel_y[k] * window[k];
            }
            double center = window[4];
            double surround = 0.25 * (window[1] + window[3] + window[5] + window[7]);
            dog += center - surround; // DoG-style center-surround contrast
            if (prev_frame) {
                double pv = prev_frame[gy * W + gx];
                motion += fabs(center - pv);
            }
            samples++;
        }
    }
    if (samples == 0) samples = 1;
    out[0] = grad_x / (double)samples;
    out[1] = grad_y / (double)samples;
    out[2] = dog / (double)samples;
    out[3] = bridge->has_prev ? (motion / (double)samples) : 0.0;
    spark_normalize(out, bridge->feature_dim);
}

void spark_bridge_inject_frame(SparkBridge *bridge, const double *frame) {
    if (!bridge || !frame || !bridge->sys || !bridge->sys->graph) return;
    if (bridge->retina_w == 0 || bridge->retina_h == 0) return;
    CognitiveGraph *g = bridge->sys->graph;
    size_t dim = g->state_dim;
    size_t stride_x = bridge->frame_width / bridge->retina_w;
    size_t stride_y = bridge->frame_height / bridge->retina_h;
    if (stride_x < bridge->patch_size) stride_x = bridge->patch_size;
    if (stride_y < bridge->patch_size) stride_y = bridge->patch_size;

    double *vec = calloc(dim, sizeof(double));
    double features[SPARK_BRIDGE_FEATURES];
    size_t patch_idx = 0;
    bridge->visual_ids_len = 0;
    for (size_t ry = 0; ry < bridge->retina_h; ++ry) {
        size_t y0 = ry * stride_y;
        if (y0 + bridge->patch_size >= bridge->frame_height) {
            y0 = bridge->frame_height > bridge->patch_size ? bridge->frame_height - bridge->patch_size : 0;
        }
        for (size_t rx = 0; rx < bridge->retina_w; ++rx) {
            size_t x0 = rx * stride_x;
            if (x0 + bridge->patch_size >= bridge->frame_width) {
                x0 = bridge->frame_width > bridge->patch_size ? bridge->frame_width - bridge->patch_size : 0;
            }
            extract_patch_features(bridge, frame, bridge->has_prev ? bridge->prev_frame : NULL, x0, y0, features);
            for (size_t k = 0; k < dim; ++k) vec[k] = features[k % bridge->feature_dim];
            spark_normalize(vec, dim);
            cg_inject(g, vec);
            record_visual_id(bridge, patch_idx, g->hnsw_last);
            patch_idx++;
            if (patch_idx >= spark_bridge_patch_count(bridge)) break;
        }
        if (patch_idx >= spark_bridge_patch_count(bridge)) break;
    }
    free(vec);

    if (bridge->prev_frame_len != bridge->frame_width * bridge->frame_height) {
        bridge->prev_frame_len = bridge->frame_width * bridge->frame_height;
        bridge->prev_frame = realloc(bridge->prev_frame, bridge->prev_frame_len * sizeof(double));
    }
    memcpy(bridge->prev_frame, frame, bridge->prev_frame_len * sizeof(double));
    bridge->has_prev = 1;
}

void spark_bridge_inject_frame_with_ids(SparkBridge *bridge, const double *frame, int *out_ids, size_t out_len) {
    if (!bridge || !frame || !bridge->sys || !bridge->sys->graph) return;
    if (bridge->retina_w == 0 || bridge->retina_h == 0) return;
    CognitiveGraph *g = bridge->sys->graph;
    size_t dim = g->state_dim;
    size_t stride_x = bridge->frame_width / bridge->retina_w;
    size_t stride_y = bridge->frame_height / bridge->retina_h;
    if (stride_x < bridge->patch_size) stride_x = bridge->patch_size;
    if (stride_y < bridge->patch_size) stride_y = bridge->patch_size;

    double *vec = calloc(dim, sizeof(double));
    double features[SPARK_BRIDGE_FEATURES];
    size_t patch_idx = 0;
    bridge->visual_ids_len = 0;
    for (size_t ry = 0; ry < bridge->retina_h; ++ry) {
        size_t y0 = ry * stride_y;
        if (y0 + bridge->patch_size >= bridge->frame_height) {
            y0 = bridge->frame_height > bridge->patch_size ? bridge->frame_height - bridge->patch_size : 0;
        }
        for (size_t rx = 0; rx < bridge->retina_w; ++rx) {
            size_t x0 = rx * stride_x;
            if (x0 + bridge->patch_size >= bridge->frame_width) {
                x0 = bridge->frame_width > bridge->patch_size ? bridge->frame_width - bridge->patch_size : 0;
            }
            extract_patch_features(bridge, frame, bridge->has_prev ? bridge->prev_frame : NULL, x0, y0, features);
            for (size_t k = 0; k < dim; ++k) vec[k] = features[k % bridge->feature_dim];
            spark_normalize(vec, dim);
            cg_inject(g, vec);
            if (out_ids && patch_idx < out_len) {
                out_ids[patch_idx] = g->hnsw_last;
            }
            record_visual_id(bridge, patch_idx, g->hnsw_last);
            patch_idx++;
            if (patch_idx >= spark_bridge_patch_count(bridge)) break;
        }
        if (patch_idx >= spark_bridge_patch_count(bridge)) break;
    }
    free(vec);

    if (bridge->prev_frame_len != bridge->frame_width * bridge->frame_height) {
        bridge->prev_frame_len = bridge->frame_width * bridge->frame_height;
        bridge->prev_frame = realloc(bridge->prev_frame, bridge->prev_frame_len * sizeof(double));
    }
    memcpy(bridge->prev_frame, frame, bridge->prev_frame_len * sizeof(double));
    bridge->has_prev = 1;
}

void spark_bridge_inject_scalars(SparkBridge *bridge, double pain, double pleasure, double sound) {
    if (!bridge || !bridge->sys || !bridge->sys->graph) return;
    CognitiveGraph *g = bridge->sys->graph;
    size_t dim = g->state_dim;
    double signals[3] = {pain, pleasure, sound};
    size_t nodes = g->num_nodes;
    double *vec = calloc(dim, sizeof(double));
    for (size_t i = 0; i < 3; ++i) {
        size_t idx = bridge->scalar_base + i;
        if (idx >= nodes) break;
        spark_zero(vec, dim);
        vec[0] = spark_clip(signals[i], -1.0, 1.0);
        if (dim > 1) vec[1] = (i == 2) ? 0.2 * vec[0] : 0.0; // small secondary channel for sound
        spark_normalize(vec, dim);
        cg_inject(g, vec);
    }
    free(vec);
}

// ---------- Outputs ----------
SparkBridgeOutputConfig spark_bridge_motor_nas_default_config(size_t num_channels) {
    SparkBridgeOutputConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.num_channels = num_channels;
    cfg.inputs_per_channel = 6;
    cfg.population = 96;
    cfg.iterations = 6;
    cfg.pool_size = 64;
    cfg.elite = cfg.population / 4;
    cfg.mutation_rate = 0.2;
    cfg.mutation_sigma = 0.15;
    cfg.activity_weight = 0.5;
    cfg.affect_weight = 0.3;
    cfg.curiosity_weight = 0.2;
    cfg.signal_gain = 1.0;
    return cfg;
}

void spark_bridge_enable_motor_nas(SparkBridge *bridge, const SparkBridgeOutputConfig *cfg) {
    if (!bridge || !cfg || cfg->num_channels == 0) return;
    spark_bridge_motor_nas_free(bridge->motor_nas);
    bridge->motor_nas = spark_bridge_motor_nas_create(cfg);
}

void spark_bridge_disable_motor_nas(SparkBridge *bridge) {
    if (!bridge) return;
    spark_bridge_motor_nas_free(bridge->motor_nas);
    bridge->motor_nas = NULL;
}

void spark_bridge_resolve_outputs(SparkBridge *bridge, double *out_signals, size_t signal_len) {
    if (!bridge || !bridge->sys || !bridge->sys->graph || !out_signals) return;
    spark_zero(out_signals, signal_len);
    SparkBridgeOutputNas *nas = bridge->motor_nas;
    if (!nas) return;
    if (signal_len < nas->cfg.num_channels) return;
    CognitiveGraph *g = bridge->sys->graph;
    size_t num_nodes = g->num_nodes;
    size_t pool_size = nas->cfg.pool_size;
    if (num_nodes == 0 || pool_size == 0) return;
    if (pool_size > num_nodes) pool_size = num_nodes;
    if (!ensure_node_buffers(nas, num_nodes)) return;
    size_t excluded = 0;
    char *exclude_mask = build_exclusion_mask(bridge, g, &excluded);
    if (excluded >= num_nodes) {
        free(exclude_mask);
        return;
    }
    size_t available = num_nodes - excluded;
    if (pool_size > available) pool_size = available;
    if (pool_size == 0) {
        free(exclude_mask);
        return;
    }
    if (!ensure_pool_buffers(nas, pool_size)) {
        free(exclude_mask);
        return;
    }

    compute_node_scores(nas, bridge, bridge->sys, exclude_mask);
    sort_indices_desc(nas->node_order, num_nodes, nas->node_score);
    size_t filled = 0;
    for (size_t i = 0; i < num_nodes && filled < pool_size; ++i) {
        int node_id = nas->node_order[i];
        if (exclude_mask && node_id >= 0 && exclude_mask[node_id]) continue;
        nas->pool_indices[filled++] = node_id;
    }
    free(exclude_mask);
    if (filled < pool_size) pool_size = filled;
    if (pool_size == 0) return;

    if (!nas->initialized) {
        init_population(nas, pool_size);
    }

    for (size_t iter = 0; iter < nas->cfg.iterations; ++iter) {
        evaluate_population(nas, pool_size);
        evolve_population(nas, pool_size);
    }
    evaluate_population(nas, pool_size);
    capture_best(nas);

    size_t inputs = nas->cfg.inputs_per_channel;
    for (size_t m = 0; m < nas->cfg.num_channels; ++m) {
        double acc = 0.0;
        for (size_t k = 0; k < inputs; ++k) {
            size_t idx = m * inputs + k;
            int pool_idx = nas->best_indices[idx];
            if (pool_idx < 0 || (size_t)pool_idx >= pool_size) continue;
            int node_id = nas->pool_indices[pool_idx];
            double weight = nas->best_weights[idx];
            acc += weight * nas->node_activity[node_id];
        }
        out_signals[m] = spark_clip(acc * nas->cfg.signal_gain, -1.0, 1.0);
    }
}
