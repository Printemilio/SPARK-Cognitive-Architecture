// Spark_Core/spark_bridge.c
// Sensory bridge acting as a thalamus for SPARK. It scales retinal density
// from available cognitive capacity and injects V1-like feature vectors.
#include "spark_bridge.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "spark_utils.h"

#define SPARK_BRIDGE_FEATURES 4

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

static void compute_retina_geometry(SparkBridge *bridge) {
    CognitiveGraph *g = bridge->sys->graph;
    double capacity_ratio = g->state_dim > 0 ? ((double)g->state_dim / (double)g->num_nodes) : 0.0;
    double density = sqrt(fmax(capacity_ratio, 1e-6)); // more capacity -> denser retina

    size_t max_w = bridge->frame_width / bridge->patch_size;
    size_t max_h = bridge->frame_height / bridge->patch_size;
    if (max_w < 1) max_w = 1;
    if (max_h < 1) max_h = 1;

    size_t desired_w = (size_t)ceil((double)max_w * density);
    size_t desired_h = (size_t)ceil((double)max_h * density);
    desired_w = clamp_size(desired_w, 1, max_w);
    desired_h = clamp_size(desired_h, 1, max_h);

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
        new_w = clamp_size(new_w, 1, max_w);
        new_h = clamp_size(new_h, 1, max_h);
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
    b->prev_frame_len = frame_width * frame_height;
    b->prev_frame = calloc(b->prev_frame_len, sizeof(double));
    b->has_prev = 0;
    return b;
}

void spark_bridge_free(SparkBridge *bridge) {
    if (!bridge) return;
    free(bridge->prev_frame);
    free(bridge);
}

void spark_bridge_reset_history(SparkBridge *bridge) {
    if (!bridge) return;
    bridge->has_prev = 0;
    if (bridge->prev_frame) spark_zero(bridge->prev_frame, bridge->prev_frame_len);
}

size_t spark_bridge_patch_count(const SparkBridge *bridge) {
    if (!bridge) return 0;
    return bridge->retina_w * bridge->retina_h;
}

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
            cg_inject(g, bridge->visual_offset + patch_idx, vec);
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
        cg_inject(g, idx, vec);
    }
    free(vec);
}
