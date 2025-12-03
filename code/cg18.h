// Spark_Core/cg18.h
// C translation of CG18 cognitive graph.
#ifndef CG18_H
#define CG18_H

#include <stddef.h>

typedef struct Synapse Synapse;
typedef struct DendriticSegment DendriticSegment;
typedef struct Node Node;
typedef struct CognitiveGraph CognitiveGraph;

typedef void (*RuleFn)(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph);

struct Synapse {
    double weight;
    int segment_id;
    double u;
    double baseline;
    double depression_rate;
    double facilitation_rate;
    double recovery_rate;
    double min_u;
    double max_u;
    double last_activity;
};

struct DendriticSegment {
    int segment_id;
    double threshold;
    int *synapses;
    size_t synapse_count;
    size_t synapse_capacity;
    double last_input;
    int active;
};

typedef struct {
    int *ids;
    Synapse *synapses;
    size_t count;
    size_t capacity;
} SynapseList;

typedef struct {
    double **entries;
    size_t count;
    size_t capacity;
} HistoryBuffer;

struct Node {
    int id;
    size_t dim;
    double *state;
    SynapseList outgoing;
    SynapseList incoming;
    DendriticSegment *segments;
    size_t num_segments;
    int segment_requirement;
    int segment_gate;
    int active_segments;
    double *segment_inputs;
    size_t segment_inputs_count;
    double curiosity;
    double affect;
    double attention;
    HistoryBuffer history;
    double energy;
    int sleeping;
    int inactivity_streak;
};

typedef struct {
    char **names;
    double *values;
    size_t count;
    size_t capacity;
} ModulatorTable;

struct CognitiveGraph {
    Node *nodes;
    size_t num_nodes;
    size_t state_dim;
    size_t num_segments;
    double segment_threshold;
    double base_activation_threshold;
    double activation_threshold;
    int sleep_patience;
    double segment_activation_ratio;
    double global_chem;
    ModulatorTable modulators;
};

// Allocation helpers
CognitiveGraph *cg_create(size_t num_nodes, size_t state_dim, double activation_threshold, int sleep_patience, size_t num_segments, double segment_threshold, double segment_activation_ratio);
void cg_free(CognitiveGraph *g);

// Topology
void cg_connect(CognitiveGraph *g, size_t src, size_t dst, double weight);

// State I/O
void cg_inject(CognitiveGraph *g, size_t idx, const double *pattern);
void cg_snapshot_states(const CognitiveGraph *g, double *out_matrix); // out_matrix size = num_nodes * state_dim

// Modulators
void cg_set_modulator(CognitiveGraph *g, const char *name, double value);
double cg_get_modulator(const CognitiveGraph *g, const char *name, double default_value);
void cg_set_activation_threshold(CognitiveGraph *g, double threshold);
void cg_set_sleep_patience(CognitiveGraph *g, int patience);
void cg_update_weight(CognitiveGraph *g, size_t src, int dst, double delta, double clamp);

// Propagation
void cg_propagate(CognitiveGraph *g, RuleFn *rules, size_t rule_count);
void cg_propagate_steps(CognitiveGraph *g, RuleFn *rules, size_t rule_count, int steps);
void cg_propagate_subset(CognitiveGraph *g, const size_t *indices, size_t count, RuleFn *rules, size_t rule_count);
void cg_dream(CognitiveGraph *g, RuleFn *rules, size_t rule_count, int steps);

// Utility factory wrappers implemented in spark C file
void cg_curiosity_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph);
void cg_affect_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph);
void cg_plasticity_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph);

#endif // CG18_H
