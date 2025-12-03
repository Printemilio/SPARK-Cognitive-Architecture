// Spark_Core/spark_bridge.h
// Sensory bridge acting as a thalamus for SPARK: dynamic retina + feature extraction.
#ifndef SPARK_BRIDGE_H
#define SPARK_BRIDGE_H

#include <stddef.h>

#include "spark18.h"

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
    double *prev_frame;
    size_t prev_frame_len;
    int has_prev;
} SparkBridge;

// Build a bridge with retina sized from state_dim/num_nodes and frame geometry.
// visual_offset sets where to start injecting into the graph.
SparkBridge *spark_bridge_create(SparkSystem *sys, size_t frame_width, size_t frame_height, size_t visual_offset);
void spark_bridge_free(SparkBridge *bridge);
void spark_bridge_reset_history(SparkBridge *bridge);
size_t spark_bridge_patch_count(const SparkBridge *bridge);
void spark_bridge_inject_frame(SparkBridge *bridge, const double *frame);
void spark_bridge_inject_scalars(SparkBridge *bridge, double pain, double pleasure, double sound);

#endif // SPARK_BRIDGE_H
