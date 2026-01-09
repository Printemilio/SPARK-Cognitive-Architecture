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

// Spark_Core/cg19.c
#include "cg19.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "spark_utils.h"

static void reset_segments(Node *n);
static void update_node_activation(CognitiveGraph *g, Node *n);
static Node make_node(int id, size_t dim, size_t num_segments, double segment_threshold, int required_segments);

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *ptr, size_t sz) {
    void *p = realloc(ptr, sz);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

static char *cg_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *p = xcalloc(len, 1);
    memcpy(p, s, len);
    return p;
}

// ---------- Synapse helpers ----------
static double synapse_effective_weight(const Synapse *s) {
    return s->weight * s->u;
}

static double synapse_step_stp(Synapse *s, double activity_level) {
    double lvl = fmax(0.0, fmin(1.0, activity_level));
    if (lvl > 0.5) {
        s->u -= s->depression_rate * lvl;
    } else {
        s->u += s->facilitation_rate * lvl;
    }
    s->u = spark_clip(s->u, s->min_u, s->max_u);
    s->u += s->recovery_rate * (s->baseline - s->u);
    s->u = spark_clip(s->u, s->min_u, s->max_u);
    s->last_activity = lvl;
    return synapse_effective_weight(s);
}

// ---------- Synapse list ----------
static void syn_list_reserve(SynapseList *lst, size_t need) {
    if (need <= lst->capacity) return;
    size_t cap = lst->capacity ? lst->capacity * 2 : 8;
    while (cap < need) cap *= 2;
    lst->ids = xrealloc(lst->ids, cap * sizeof(int));
    lst->synapses = xrealloc(lst->synapses, cap * sizeof(Synapse));
    lst->capacity = cap;
}

static ssize_t syn_list_index(const SynapseList *lst, int id) {
    for (size_t i = 0; i < lst->count; ++i) {
        if (lst->ids[i] == id) return (ssize_t)i;
    }
    return -1;
}

static Synapse *syn_list_get_or_add(SynapseList *lst, int id) {
    ssize_t idx = syn_list_index(lst, id);
    if (idx >= 0) {
        return &lst->synapses[idx];
    }
    syn_list_reserve(lst, lst->count + 1);
    size_t pos = lst->count++;
    lst->ids[pos] = id;
    Synapse *s = &lst->synapses[pos];
    s->weight = 0.0;
    s->segment_id = -1;
    s->u = 1.0;
    s->baseline = 1.0;
    s->depression_rate = 0.08;
    s->facilitation_rate = 0.04;
    s->recovery_rate = 0.05;
    s->min_u = 0.2;
    s->max_u = 1.8;
    s->last_activity = 0.0;
    return s;
}

// ---------- Segment helpers ----------
static void segment_reset(DendriticSegment *seg) {
    seg->last_input = 0.0;
    seg->active = 0;
}

static void segment_accumulate(DendriticSegment *seg, double value) {
    seg->last_input += value;
}

static void segment_add_synapse(DendriticSegment *seg, int src_id) {
    if (seg->synapse_count + 1 > seg->synapse_capacity) {
        size_t cap = seg->synapse_capacity ? seg->synapse_capacity * 2 : 4;
        seg->synapses = xrealloc(seg->synapses, cap * sizeof(int));
        seg->synapse_capacity = cap;
    }
    seg->synapses[seg->synapse_count++] = src_id;
}

// ---------- History ----------
static void history_push(HistoryBuffer *h, const double *state, size_t dim) {
    if (h->count + 1 > h->capacity) {
        size_t cap = h->capacity ? h->capacity * 2 : 16;
        h->entries = xrealloc(h->entries, cap * sizeof(double *));
        h->capacity = cap;
    }
    double *copy = xcalloc(dim, sizeof(double));
    memcpy(copy, state, dim * sizeof(double));
    h->entries[h->count++] = copy;
}

static void history_free(HistoryBuffer *h) {
    for (size_t i = 0; i < h->count; ++i) free(h->entries[i]);
    free(h->entries);
    h->entries = NULL;
    h->count = h->capacity = 0;
}

// ---------- Modulators ----------
static void modulators_init(ModulatorTable *m) {
    m->names = NULL;
    m->values = NULL;
    m->count = m->capacity = 0;
}

static void modulators_free(ModulatorTable *m) {
    if (!m) return;
    for (size_t i = 0; i < m->count; ++i) free(m->names[i]);
    free(m->names);
    free(m->values);
}

// ---------- Active set helpers ----------
static int list_contains(const size_t *list, size_t len, size_t id) {
    for (size_t i = 0; i < len; ++i) {
        if (list[i] == id) return 1;
    }
    return 0;
}

static void push_current_active(CognitiveGraph *g, size_t id) {
    if (!g || id >= g->num_nodes) return;
    if (list_contains(g->active_current, g->active_current_len, id)) return;
    if (g->active_current_len < g->num_nodes) {
        g->active_current[g->active_current_len++] = id;
    }
}

static void queue_next_active(CognitiveGraph *g, size_t id) {
    if (!g || id >= g->num_nodes) return;
    if (g->scheduled[id]) return;
    if (g->active_next_len < g->num_nodes) {
        g->active_next[g->active_next_len++] = id;
        g->scheduled[id] = 1;
    }
}

static void ensure_segments_reset(CognitiveGraph *g, size_t node_id) {
    if (!g || node_id >= g->num_nodes) return;
    if (g->touched_flag[node_id]) return;
    reset_segments(&g->nodes[node_id]);
    if (g->touched_len < g->num_nodes) {
        g->touched_nodes[g->touched_len++] = node_id;
    }
    g->touched_flag[node_id] = 1;
}

static size_t finalize_touched_segments(CognitiveGraph *g) {
    size_t active_for_chem = 0;
    for (size_t i = 0; i < g->touched_len; ++i) {
        size_t idx = g->touched_nodes[i];
        Node *n = &g->nodes[idx];
        if (n->num_segments == 0) {
            n->segment_gate = 1;
            n->active_segments = 0;
        } else {
            for (size_t s = 0; s < n->num_segments; ++s) {
                n->segment_inputs[s] = n->segments[s].last_input;
            }
            if (n->num_segments && n->segment_requirement <= 0) {
                n->segment_gate = 1;
                n->active_segments = (int)n->num_segments;
            } else {
                int active = 0;
                for (size_t s = 0; s < n->num_segments; ++s) {
                    DendriticSegment *seg = &n->segments[s];
                    seg->active = seg->last_input >= seg->threshold;
                    if (seg->active) active++;
                }
                n->active_segments = active;
                n->segment_gate = active >= n->segment_requirement;
            }
        }
        update_node_activation(g, n);
        if (n->energy >= g->base_activation_threshold) active_for_chem++;
        g->touched_flag[idx] = 0;
    }
    g->touched_len = 0;
    return active_for_chem;
}

// ---------- Graph growth helpers ----------
static void graph_reserve(CognitiveGraph *g, size_t need) {
    if (g->max_nodes && need > g->max_nodes) need = g->max_nodes;
    if (need <= g->capacity) return;
    size_t cap = g->capacity ? g->capacity * 2 : 16;
    while (cap < need) cap *= 2;
    if (g->max_nodes && cap > g->max_nodes) cap = g->max_nodes;
    g->nodes = xrealloc(g->nodes, cap * sizeof(Node));
    g->active_current = xrealloc(g->active_current, cap * sizeof(size_t));
    g->active_next = xrealloc(g->active_next, cap * sizeof(size_t));
    g->scheduled = xrealloc(g->scheduled, cap * sizeof(char));
    g->touched_nodes = xrealloc(g->touched_nodes, cap * sizeof(size_t));
    g->touched_flag = xrealloc(g->touched_flag, cap * sizeof(char));
    g->nbr_nodes = xrealloc(g->nbr_nodes, cap * sizeof(Node *));
    g->nbr_weights = xrealloc(g->nbr_weights, cap * sizeof(double));
    memset(g->scheduled + g->capacity, 0, (cap - g->capacity) * sizeof(char));
    memset(g->touched_flag + g->capacity, 0, (cap - g->capacity) * sizeof(char));
    g->capacity = cap;
    g->nbr_cap = cap;
}

void cg_reserve(CognitiveGraph *g, size_t need) {
    if (!g) return;
    graph_reserve(g, need);
}

static int hnsw_random_level(void) {
    int level = 0;
    while (spark_rand_uniform() < 0.5 && level < 6) level++;
    return level;
}

static void hnsw_alloc_links(Node *n, int level) {
    if (level < 0) return;
    n->hnsw_links = xcalloc((size_t)(level + 1), sizeof(SynapseList));
    n->hnsw_level = level;
}

static void hnsw_add_link(Node *a, Node *b, int level, int max_m) {
    if (level < 0) return;
    if (level > a->hnsw_level || level > b->hnsw_level) return;
    SynapseList *la = &a->hnsw_links[level];
    SynapseList *lb = &b->hnsw_links[level];
    syn_list_get_or_add(la, b->id);
    syn_list_get_or_add(lb, a->id);
    if ((int)la->count > max_m) la->count = (size_t)max_m;
    if ((int)lb->count > max_m) lb->count = (size_t)max_m;
}

static int hnsw_nearest_at_level(CognitiveGraph *g, const double *vec, int enter, int level) {
    int current = enter;
    int improved = 1;
    while (improved) {
        improved = 0;
        Node *cn = &g->nodes[(size_t)current];
        if (level > cn->hnsw_level || !cn->hnsw_links) return current;
        SynapseList *links = &cn->hnsw_links[level];
        for (size_t i = 0; i < links->count; ++i) {
            int nb_id = links->ids[i];
            if (nb_id < 0 || (size_t)nb_id >= g->num_nodes) continue;
            Node *nb = &g->nodes[(size_t)nb_id];
            double c_dist = 1.0 - spark_cosine_similarity(vec, cn->state, g->state_dim);
            double n_dist = 1.0 - spark_cosine_similarity(vec, nb->state, g->state_dim);
            if (n_dist + 1e-9 < c_dist) {
                current = nb_id;
                improved = 1;
                break;
            }
        }
    }
    return current;
}

static int hnsw_search(CognitiveGraph *g, const double *vec) {
    if (g->hnsw_entry < 0 || g->num_nodes == 0) return -1;
    int enter = g->hnsw_last;
    if (enter < 0 || (size_t)enter >= g->num_nodes) {
        enter = g->hnsw_entry;
    }
    for (int level = g->hnsw_max_level; level > 0; --level) {
        enter = hnsw_nearest_at_level(g, vec, enter, level);
    }
    return hnsw_nearest_at_level(g, vec, enter, 0);
}

static Node *graph_new_node(CognitiveGraph *g, const double *state_vec) {
    if (g->max_nodes && g->num_nodes >= g->max_nodes) return NULL;
    graph_reserve(g, g->num_nodes + 1);
    size_t idx = g->num_nodes;
    Node n = make_node((int)idx, g->state_dim, g->num_segments, g->segment_threshold, (int)ceil(g->num_segments * g->segment_activation_ratio));
    memcpy(n.state, state_vec, g->state_dim * sizeof(double));
    n.hnsw_level = 0;
    n.hnsw_links = NULL;
    g->nodes[idx] = n;
    g->num_nodes++;
    return &g->nodes[idx];
}

static Node *graph_ensure_node(CognitiveGraph *g, size_t id) {
    if (id < g->num_nodes) return &g->nodes[id];
    double *zero = xcalloc(g->state_dim, sizeof(double));
    Node *last = NULL;
    while (g->num_nodes <= id) {
        last = graph_new_node(g, zero);
        int lvl = hnsw_random_level();
        hnsw_alloc_links(last, lvl);
        if (g->hnsw_entry < 0) g->hnsw_entry = last->id;
        if (lvl > g->hnsw_max_level) g->hnsw_max_level = lvl;
    }
    free(zero);
    return last;
}

static Node *hnsw_insert_or_get(CognitiveGraph *g, const double *vec, double create_threshold) {
    if (!g || !vec) return NULL;
    if (g->num_nodes == 0) {
        Node *n = graph_new_node(g, vec);
        int lvl = hnsw_random_level();
        hnsw_alloc_links(n, lvl);
        g->hnsw_entry = n->id;
        g->hnsw_last = n->id;
        g->hnsw_max_level = lvl;
        return n;
    }
    int nearest = hnsw_search(g, vec);
    if (nearest >= 0) {
        Node *nb = &g->nodes[(size_t)nearest];
        double dist = 1.0 - spark_cosine_similarity(vec, nb->state, g->state_dim);
        if (dist <= create_threshold) {
            memcpy(nb->state, vec, g->state_dim * sizeof(double));
            g->hnsw_last = nb->id;
            return nb;
        }
    }
    Node *n = graph_new_node(g, vec);
    if (!n) {
        if (nearest >= 0) {
            Node *nb = &g->nodes[(size_t)nearest];
            memcpy(nb->state, vec, g->state_dim * sizeof(double));
            g->hnsw_last = nb->id;
            return nb;
        }
        return NULL;
    }
    int lvl = hnsw_random_level();
    hnsw_alloc_links(n, lvl);
    if (lvl > g->hnsw_max_level) {
        g->hnsw_max_level = lvl;
        g->hnsw_entry = n->id;
    }
    int cur = g->hnsw_entry;
    for (int level = g->hnsw_max_level; level >= 0; --level) {
        if (level > lvl) {
            cur = hnsw_nearest_at_level(g, vec, cur, level);
            continue;
        }
        cur = hnsw_nearest_at_level(g, vec, cur, level);
        Node *cnode = &g->nodes[(size_t)cur];
        hnsw_add_link(n, cnode, level, g->hnsw_M);
    }
    g->hnsw_last = n->id;
    return n;
}

void cg_set_modulator(CognitiveGraph *g, const char *name, double value) {
    ModulatorTable *m = &g->modulators;
    for (size_t i = 0; i < m->count; ++i) {
        if (strcmp(m->names[i], name) == 0) {
            m->values[i] = value;
            return;
        }
    }
    if (m->count + 1 > m->capacity) {
        size_t cap = m->capacity ? m->capacity * 2 : 8;
        m->names = xrealloc(m->names, cap * sizeof(char *));
        m->values = xrealloc(m->values, cap * sizeof(double));
        m->capacity = cap;
    }
    m->names[m->count] = cg_strdup(name);
    m->values[m->count] = value;
    m->count++;
}

double cg_get_modulator(const CognitiveGraph *g, const char *name, double default_value) {
    const ModulatorTable *m = &g->modulators;
    for (size_t i = 0; i < m->count; ++i) {
        if (strcmp(m->names[i], name) == 0) return m->values[i];
    }
    return default_value;
}

// ---------- Node helpers ----------
static Node make_node(int id, size_t dim, size_t num_segments, double segment_threshold, int required_segments) {
    Node n;
    n.id = id;
    n.dim = dim;
    n.state = xcalloc(dim, sizeof(double));
    n.outgoing.ids = NULL;
    n.outgoing.synapses = NULL;
    n.outgoing.count = n.outgoing.capacity = 0;
    n.incoming.ids = NULL;
    n.incoming.synapses = NULL;
    n.incoming.count = n.incoming.capacity = 0;
    n.segments = xcalloc(num_segments, sizeof(DendriticSegment));
    n.num_segments = num_segments;
    for (size_t i = 0; i < num_segments; ++i) {
        n.segments[i].segment_id = (int)i;
        n.segments[i].threshold = segment_threshold;
        n.segments[i].synapses = NULL;
        n.segments[i].synapse_count = n.segments[i].synapse_capacity = 0;
        n.segments[i].last_input = 0.0;
        n.segments[i].active = 0;
    }
    n.segment_requirement = required_segments;
    n.segment_gate = 1;
    n.active_segments = 0;
    n.segment_inputs = xcalloc(num_segments, sizeof(double));
    n.segment_inputs_count = num_segments;
    n.curiosity = 0.0;
    n.affect = 0.0;
    n.attention = 0.0;
    n.history.entries = NULL;
    n.history.count = n.history.capacity = 0;
    n.energy = 0.0;
    n.sleeping = 0;
    n.inactivity_streak = 0;
    n.hnsw_level = 0;
    n.hnsw_links = NULL;
    return n;
}

static void free_node(Node *n) {
    free(n->state);
    free(n->outgoing.ids);
    free(n->outgoing.synapses);
    free(n->incoming.ids);
    free(n->incoming.synapses);
    for (size_t i = 0; i < n->num_segments; ++i) {
        free(n->segments[i].synapses);
    }
    free(n->segments);
    free(n->segment_inputs);
    history_free(&n->history);
    if (n->hnsw_links) {
        for (int l = 0; l <= n->hnsw_level; ++l) {
            free(n->hnsw_links[l].ids);
            free(n->hnsw_links[l].synapses);
        }
        free(n->hnsw_links);
    }
}

static int select_segment(Node *n) {
    if (n->num_segments == 0) return -1;
    if (n->num_segments == 1) return 0;
    size_t best = 0;
    size_t best_size = SIZE_MAX;
    for (size_t i = 0; i < n->num_segments; ++i) {
        size_t sz = n->segments[i].synapse_count;
        if (sz < best_size) {
            best_size = sz;
            best = i;
        }
    }
    return (int)best;
}

static void node_register_incoming(Node *dst, int src_idx, Synapse *syn) {
    ssize_t existing_idx = syn_list_index(&dst->incoming, src_idx);
    if (existing_idx >= 0 && &dst->incoming.synapses[existing_idx] == syn) {
        return;
    }
    if (existing_idx >= 0) {
        int old_seg = dst->incoming.synapses[existing_idx].segment_id;
        if (old_seg >= 0 && (size_t)old_seg < dst->num_segments) {
            // remove from segment list
            DendriticSegment *seg = &dst->segments[old_seg];
            for (size_t i = 0; i < seg->synapse_count; ++i) {
                if (seg->synapses[i] == src_idx) {
                    memmove(&seg->synapses[i], &seg->synapses[i + 1], (seg->synapse_count - i - 1) * sizeof(int));
                    seg->synapse_count--;
                    break;
                }
            }
        }
    }
    int seg_id = syn->segment_id;
    if (seg_id < 0 || (size_t)seg_id >= dst->num_segments) {
        seg_id = select_segment(dst);
    }
    syn->segment_id = seg_id;
    Synapse *incoming_syn = syn_list_get_or_add(&dst->incoming, src_idx);
    *incoming_syn = *syn;
    if (seg_id >= 0 && (size_t)seg_id < dst->num_segments) {
        DendriticSegment *seg = &dst->segments[seg_id];
        segment_add_synapse(seg, src_idx);
    }
}

static void node_record_state(Node *n) {
    // Lightweight: keep only last state (avoids per-step allocations in fast loops)
    if (n->history.count == 0) {
        history_push(&n->history, n->state, n->dim);
    } else {
        memcpy(n->history.entries[0], n->state, n->dim * sizeof(double));
    }
}

static void update_node_activation(CognitiveGraph *g, Node *n) {
    n->energy = spark_norm(n->state, n->dim);
    if (n->energy >= g->activation_threshold) {
        n->sleeping = 0;
        n->inactivity_streak = 0;
    } else {
        n->inactivity_streak += 1;
        if (n->inactivity_streak >= g->sleep_patience) {
            n->sleeping = 1;
        }
    }
}

static void update_chemistry(CognitiveGraph *g, size_t active_count) {
    if (!g->num_nodes) return;
    double frac_active = active_count / (double)g->num_nodes;
    double target = frac_active;
    double chem = g->global_chem + 0.2 * (target - g->global_chem);
    if (chem < 0.0) chem = 0.0;
    if (chem > 1.0) chem = 1.0;
    g->global_chem = chem;
    cg_set_modulator(g, "chem", chem);
    g->activation_threshold = fmax(0.05 * g->base_activation_threshold,
                                   g->base_activation_threshold * (1.0 - 0.7 * chem));
}

static void reset_segments(Node *n) {
    if (n->num_segments == 0) {
        n->segment_gate = 1;
        n->active_segments = 0;
        return;
    }
    for (size_t i = 0; i < n->num_segments; ++i) {
        segment_reset(&n->segments[i]);
    }
    n->segment_gate = 0;
    n->active_segments = 0;
    for (size_t i = 0; i < n->segment_inputs_count; ++i) n->segment_inputs[i] = 0.0;
}

static size_t neighbors_for_propagation(CognitiveGraph *g, size_t idx, Node **out_nodes, double *out_weights, size_t max_nbr) {
    size_t count = 0;
    Node *node = &g->nodes[idx];
    for (size_t k = 0; k < node->outgoing.count; ++k) {
        int nb_idx = node->outgoing.ids[k];
        Synapse *syn = &node->outgoing.synapses[k];
        Node *nb = &g->nodes[nb_idx];
        if (nb->sleeping) continue;
        if (!nb->segment_gate) continue;
        if (nb->energy < (g->activation_threshold * 0.5)) continue;
        if (count < max_nbr) {
            out_nodes[count] = nb;
            out_weights[count] = synapse_effective_weight(syn);
        }
        count++;
    }
    return count;
}

// ---------- Public API ----------
CognitiveGraph *cg_create(size_t num_nodes, size_t state_dim, double activation_threshold, int sleep_patience, size_t num_segments, double segment_threshold, double segment_activation_ratio) {
    CognitiveGraph *g = xcalloc(1, sizeof(CognitiveGraph));
    g->num_nodes = 0;
    g->capacity = 0;
    g->max_nodes = num_nodes > 0 ? num_nodes : 0;
    g->state_dim = state_dim;
    g->num_segments = num_segments > 0 ? num_segments : 1;
    g->segment_threshold = segment_threshold;
    g->segment_activation_ratio = segment_activation_ratio;
    g->base_activation_threshold = activation_threshold;
    g->activation_threshold = activation_threshold;
    g->sleep_patience = sleep_patience > 0 ? sleep_patience : 1;
    modulators_init(&g->modulators);
    g->global_chem = 0.0;
    g->active_current = NULL;
    g->active_next = NULL;
    g->scheduled = NULL;
    g->touched_nodes = NULL;
    g->touched_flag = NULL;
    g->nbr_nodes = NULL;
    g->nbr_weights = NULL;
    g->nbr_cap = 0;
    g->active_current_len = 0;
    g->active_next_len = 0;
    g->touched_len = 0;
    g->hnsw_entry = -1;
    g->hnsw_last = -1;
    g->hnsw_max_level = 0;
    g->hnsw_M = 8;
    graph_reserve(g, num_nodes > 0 ? num_nodes : 16);
    return g;
}

void cg_free(CognitiveGraph *g) {
    if (!g) return;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        free_node(&g->nodes[i]);
    }
    free(g->nodes);
    modulators_free(&g->modulators);
    free(g->active_current);
    free(g->active_next);
    free(g->scheduled);
    free(g->touched_nodes);
    free(g->touched_flag);
    free(g->nbr_nodes);
    free(g->nbr_weights);
    free(g);
}

void cg_connect(CognitiveGraph *g, size_t src, size_t dst, double weight) {
    if (!g) return;
    graph_ensure_node(g, src);
    graph_ensure_node(g, dst);
    Node *src_node = &g->nodes[src];
    Synapse *syn = syn_list_get_or_add(&src_node->outgoing, (int)dst);
    syn->weight = weight;
    Node *dst_node = &g->nodes[dst];
    node_register_incoming(dst_node, (int)src, syn);
}

void cg_inject(CognitiveGraph *g, const double *pattern) {
    if (!g || !pattern) return;
    static double create_threshold = -1.0;
    if (create_threshold < 0.0) {
        const char *env = getenv("SPARK_INJECT_THRESHOLD");
        if (env && env[0]) {
            create_threshold = strtod(env, NULL);
        }
        if (create_threshold <= 0.0) create_threshold = 0.3;
    }
    Node *n = hnsw_insert_or_get(g, pattern, create_threshold);
    if (!n) return;
    update_node_activation(g, n);
    push_current_active(g, (size_t)n->id);
}

void cg_update_weight(CognitiveGraph *g, size_t src, int dst, double delta, double clamp) {
    if (!g || src >= g->num_nodes) return;
    Node *n = &g->nodes[src];
    Synapse *syn = syn_list_get_or_add(&n->outgoing, dst);
    double w = syn->weight + delta;
    syn->weight = spark_clip(w, -clamp, clamp);
}

void cg_snapshot_states(const CognitiveGraph *g, double *out_matrix) {
    for (size_t i = 0; i < g->num_nodes; ++i) {
        memcpy(out_matrix + i * g->state_dim, g->nodes[i].state, g->state_dim * sizeof(double));
    }
}

void cg_set_activation_threshold(CognitiveGraph *g, double threshold) {
    double val = threshold < 0.0 ? 0.0 : threshold;
    g->base_activation_threshold = val;
    g->activation_threshold = val;
}

void cg_set_sleep_patience(CognitiveGraph *g, int patience) {
    g->sleep_patience = patience > 0 ? patience : 1;
}

void cg_propagate(CognitiveGraph *g, RuleFn *rules, size_t rule_count) {
    if (!g) return;
    if (g->active_current_len == 0) return;

    g->active_next_len = 0;
    g->touched_len = 0;

    // Pass 1: update STP and accumulate dendritic inputs only for active sources and their targets
    for (size_t idx = 0; idx < g->active_current_len; ++idx) {
        size_t node_id = g->active_current[idx];
        if (node_id >= g->num_nodes) continue;
        Node *src = &g->nodes[node_id];
        ensure_segments_reset(g, node_id);
        double src_energy = spark_norm(src->state, src->dim);
        double activity_level = src_energy > 0.0 ? src_energy / (src_energy + 1.0) : 0.0;
        for (size_t k = 0; k < src->outgoing.count; ++k) {
            int dst_idx = src->outgoing.ids[k];
            Synapse *syn = &src->outgoing.synapses[k];
            double eff_weight = synapse_step_stp(syn, activity_level);
            if (dst_idx < 0 || (size_t)dst_idx >= g->num_nodes) continue;
            ensure_segments_reset(g, (size_t)dst_idx);
            Node *dst = &g->nodes[dst_idx];
            if (dst->num_segments == 0) continue;
            if (syn->segment_id < 0 || (size_t)syn->segment_id >= dst->num_segments) {
                node_register_incoming(dst, src->id, syn);
            }
            int seg_id = syn->segment_id;
            if (seg_id < 0 || (size_t)seg_id >= dst->num_segments) continue;
            double contribution = eff_weight * src_energy;
            segment_accumulate(&dst->segments[seg_id], contribution);
        }
    }

    size_t active_for_chem = finalize_touched_segments(g);
    update_chemistry(g, active_for_chem);

    // neighbor cache sized generously
    if (g->nbr_cap < g->num_nodes) graph_reserve(g, g->num_nodes);
    size_t max_neighbors = g->num_nodes;
    Node **nbr_nodes = g->nbr_nodes;
    double *nbr_weights = g->nbr_weights;
    if (!nbr_nodes || !nbr_weights) return;

    double chem = cg_get_modulator(g, "chem", 0.0);
    double noise_std = chem > 0.0 ? 0.02 * (0.3 + 0.7 * chem) : 0.0;

    for (size_t idx = 0; idx < g->active_current_len; ++idx) {
        size_t node_id = g->active_current[idx];
        if (node_id >= g->num_nodes) continue;
        Node *node = &g->nodes[node_id];
        if (node->sleeping) {
            node_record_state(node);
            continue;
        }
        size_t nbr_count = neighbors_for_propagation(g, node_id, nbr_nodes, nbr_weights, max_neighbors);
        for (size_t r = 0; r < rule_count; ++r) {
            RuleFn fn = rules[r];
            if (fn) fn(node, nbr_nodes, nbr_weights, nbr_count, g);
        }
        for (size_t n = 0; n < nbr_count; ++n) {
            queue_next_active(g, (size_t)nbr_nodes[n]->id);
        }
        if (noise_std > 0.0) {
            for (size_t d = 0; d < node->dim; ++d) {
                node->state[d] += spark_gauss(0.0, noise_std);
                node->state[d] = spark_clip(node->state[d], -1.0, 1.0);
            }
        }
        node_record_state(node);
    }
    for (size_t i = 0; i < g->active_next_len; ++i) {
        size_t id = g->active_next[i];
        if (id < g->num_nodes) g->scheduled[id] = 0;
    }
    size_t *tmp = g->active_current;
    g->active_current = g->active_next;
    g->active_next = tmp;
    g->active_current_len = g->active_next_len;
    g->active_next_len = 0;
}

void cg_propagate_steps(CognitiveGraph *g, RuleFn *rules, size_t rule_count, int steps) {
    int s = steps < 1 ? 1 : steps;
    for (int i = 0; i < s; ++i) cg_propagate(g, rules, rule_count);
}

void cg_propagate_subset(CognitiveGraph *g, const size_t *indices, size_t count, RuleFn *rules, size_t rule_count) {
    if (!g) return;
    g->active_current_len = 0;
    g->active_next_len = 0;
    for (size_t k = 0; k < count; ++k) {
        graph_ensure_node(g, indices[k]);
        push_current_active(g, indices[k]);
    }
    cg_propagate(g, rules, rule_count);
}

void cg_dream(CognitiveGraph *g, RuleFn *rules, size_t rule_count, int steps) {
    cg_propagate_steps(g, rules, rule_count, steps < 1 ? 1 : steps);
}
