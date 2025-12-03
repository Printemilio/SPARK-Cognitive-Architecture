// Spark_Core/spark18.c
#include "spark18.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_utils.h"

// rule context (shared across graph callbacks)
static CuriosityModule *g_rule_curiosity = NULL;
static AffectModule *g_rule_affect = NULL;
static DynamicModule *g_rule_dynamic = NULL;
static SparkConfig g_rule_cfg;

static void register_rule_context(SparkSystem *sys) {
    g_rule_curiosity = sys->curiosity;
    g_rule_affect = sys->affect_module;
    g_rule_dynamic = sys->dynamic_module;
    g_rule_cfg = sys->cfg;
}

// ---------- Basic vector helpers ----------
void normalize_vec(double *v, size_t n) { spark_normalize(v, n); }
double cosine_similarity_vec(const double *a, const double *b, size_t n) { return spark_cosine_similarity(a, b, n); }

// ---------- Orthonormal basis (2 x d) ----------
static void orthonormal_basis(size_t dim, double U[2][64]) {
    double l[64], v[64], w[64];
    for (size_t i = 0; i < dim; ++i) l[i] = spark_gauss(0.0, 1.0);
    normalize_vec(l, dim);
    // e1 orthogonal to l
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) v[i] = spark_gauss(0.0, 1.0);
    for (size_t i = 0; i < dim; ++i) dot += v[i] * l[i];
    for (size_t i = 0; i < dim; ++i) v[i] -= dot * l[i];
    if (spark_norm(v, dim) < 1e-9) {
        for (size_t i = 0; i < dim; ++i) {
            v[i] = spark_gauss(0.0, 1.0);
        }
        dot = 0.0;
        for (size_t i = 0; i < dim; ++i) dot += v[i] * l[i];
        for (size_t i = 0; i < dim; ++i) v[i] -= dot * l[i];
    }
    normalize_vec(v, dim);
    // e2 orthogonal to both l and v (Gram-Schmidt)
    double dot_l = 0.0, dot_v = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        w[i] = spark_gauss(0.0, 1.0);
        dot_l += w[i] * l[i];
        dot_v += w[i] * v[i];
    }
    for (size_t i = 0; i < dim; ++i) w[i] -= dot_l * l[i] + dot_v * v[i];
    normalize_vec(w, dim);
    for (size_t i = 0; i < dim; ++i) {
        U[0][i] = v[i];
        U[1][i] = w[i];
    }
}

// ---------- Bubble memory ----------
BubbleShadowMemory *bubble_memory_create(size_t dim, double r0, double match_tau, double dir_tau, double emotion_gain, double split_emotion, size_t max_bubbles) {
    BubbleShadowMemory *m = calloc(1, sizeof(BubbleShadowMemory));
    m->d = dim;
    m->r0 = r0;
    m->match_tau = match_tau;
    m->dir_tau = dir_tau;
    m->emotion_gain = emotion_gain;
    m->split_emotion = split_emotion;
    m->max_bubbles = max_bubbles;
    orthonormal_basis(dim, m->U);
    return m;
}

void bubble_memory_free(BubbleShadowMemory *m) {
    if (!m) return;
    for (size_t i = 0; i < m->count; ++i) free(m->bubbles[i].c);
    free(m->bubbles);
    free(m);
}

static void bubble_memory_add(BubbleShadowMemory *m, const double *x, size_t dim, double emotion) {
    if (m->count + 1 > m->capacity) {
        size_t cap = m->capacity ? m->capacity * 2 : 16;
        m->bubbles = realloc(m->bubbles, cap * sizeof(Bubble));
        m->capacity = cap;
    }
    Bubble *b = &m->bubbles[m->count++];
    b->c = calloc(dim, sizeof(double));
    memcpy(b->c, x, dim * sizeof(double));
    b->r = m->r0;
    b->s[0] = b->s[1] = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        b->s[0] += m->U[0][i] * x[i];
        b->s[1] += m->U[1][i] * x[i];
    }
    b->n = 1;
    b->a = emotion;
    b->S[0] = b->S[3] = 1.0;
    b->S[1] = b->S[2] = 0.0;
    // enforce max
    if (m->max_bubbles > 0 && m->count > m->max_bubbles) {
        free(m->bubbles[0].c);
        memmove(&m->bubbles[0], &m->bubbles[1], (m->count - 1) * sizeof(Bubble));
        m->count--;
    }
}

static size_t bubble_nearest(BubbleShadowMemory *m, const double *sx, double *out_dist) {
    double best = 1e9;
    size_t best_idx = (size_t)-1;
    for (size_t i = 0; i < m->count; ++i) {
        double dx = sx[0] - m->bubbles[i].s[0];
        double dy = sx[1] - m->bubbles[i].s[1];
        double d = sqrt(dx * dx + dy * dy);
        if (d < best) {
            best = d;
            best_idx = i;
        }
    }
    if (out_dist) *out_dist = best;
    return best_idx;
}

void bubble_memory_store(BubbleShadowMemory *m, const double *x, size_t dim, double emotion) {
    double nx = spark_norm(x, dim);
    if (nx < 1e-12) return;
    double *vx = calloc(dim, sizeof(double));
    memcpy(vx, x, dim * sizeof(double));
    normalize_vec(vx, dim);
    double sx[2] = {0.0, 0.0};
    for (size_t i = 0; i < dim; ++i) {
        sx[0] += m->U[0][i] * vx[i];
        sx[1] += m->U[1][i] * vx[i];
    }
    double best_dist;
    size_t idx = bubble_nearest(m, sx, &best_dist);
    Bubble *target = NULL;
    if (idx != (size_t)-1 && best_dist <= m->match_tau) {
        target = &m->bubbles[idx];
        double dir = cosine_similarity_vec(vx, target->c, dim);
        if (dir < m->dir_tau) {
            target = NULL;
        }
    }
    if (!target) {
        bubble_memory_add(m, vx, dim, emotion);
        free(vx);
        m->stores++;
        if (m->stores % 64 == 0) {
            bubble_memory_forget(m, 0.995, 0.08, 1e-3);
            bubble_memory_merge_close(m, 0.10, 0.05);
        }
        return;
    }
    // update bubble
    double alpha = 1.0 / (double)(target->n + 1);
    for (size_t i = 0; i < dim; ++i) {
        target->c[i] = (1.0 - alpha) * target->c[i] + alpha * vx[i];
    }
    target->s[0] = (1.0 - alpha) * target->s[0] + alpha * sx[0];
    target->s[1] = (1.0 - alpha) * target->s[1] + alpha * sx[1];
    target->r = (1.0 - alpha) * target->r + alpha * fmax(m->r0, best_dist);
    target->a += emotion * m->emotion_gain;
    target->n += 1;
    free(vx);
    m->stores++;
    if (m->stores % 64 == 0) {
        bubble_memory_forget(m, 0.995, 0.08, 1e-3);
        bubble_memory_merge_close(m, 0.10, 0.05);
    }
}

double bubble_memory_project(BubbleShadowMemory *m, const double *x, size_t dim) {
    if (m->count == 0) return 0.0;
    double nx = spark_norm(x, dim);
    if (nx < 1e-12) return 0.0;
    double *vx = calloc(dim, sizeof(double));
    memcpy(vx, x, dim * sizeof(double));
    normalize_vec(vx, dim);
    double sx[2] = {0.0, 0.0};
    for (size_t i = 0; i < dim; ++i) {
        sx[0] += m->U[0][i] * vx[i];
        sx[1] += m->U[1][i] * vx[i];
    }
    double best_dist;
    size_t idx = bubble_nearest(m, sx, &best_dist);
    double fam = 0.0;
    if (idx != (size_t)-1) {
        Bubble *b = &m->bubbles[idx];
        double dir = cosine_similarity_vec(vx, b->c, dim);
        fam = fmax(0.0, 1.0 - best_dist) * fmax(0.0, dir);
    }
    free(vx);
    m->recalls++;
    return fam;
}

void bubble_memory_forget(BubbleShadowMemory *m, double decay, double min_radius, double min_opacity) {
    if (!m) return;
    size_t write = 0;
    for (size_t i = 0; i < m->count; ++i) {
        Bubble *b = &m->bubbles[i];
        b->r = fmax(min_radius, b->r * decay);
        b->a = fmax(0.0, b->a * decay);
        int new_n = (int)fmax(1.0, floor((double)b->n * decay + 0.5));
        b->n = new_n;
        for (size_t k = 0; k < 4; ++k) b->S[k] *= decay;
        double score = b->r * (1.0 + b->a);
        if (score <= min_radius * (1.0 + min_opacity)) {
            free(b->c);
            continue;
        }
        if (write != i) m->bubbles[write] = *b;
        write++;
    }
    m->count = write;
}

void bubble_memory_merge_close(BubbleShadowMemory *m, double shadow_thresh, double dir_thresh) {
    if (!m || m->count < 2) return;
    size_t n = m->count;
    char *merged = calloc(n, sizeof(char));
    size_t new_cap = m->capacity ? m->capacity : n;
    Bubble *new_list = calloc(new_cap > 0 ? new_cap : n, sizeof(Bubble));
    size_t new_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (merged[i]) continue;
        Bubble *bi = &m->bubbles[i];
        double acc_n = (double)bi->n;
        double acc_r = bi->r * (double)bi->n;
        double acc_a = bi->a * (double)bi->n;
        double acc_s[2] = {bi->s[0] * (double)bi->n, bi->s[1] * (double)bi->n};
        double acc_S[4] = {bi->S[0] * (double)bi->n, bi->S[1] * (double)bi->n, bi->S[2] * (double)bi->n, bi->S[3] * (double)bi->n};
        double *acc_c = calloc(m->d, sizeof(double));
        for (size_t k = 0; k < m->d; ++k) acc_c[k] = bi->c[k] * (double)bi->n;
        for (size_t j = i + 1; j < n; ++j) {
            if (merged[j]) continue;
            Bubble *bj = &m->bubbles[j];
            double dx = bi->s[0] - bj->s[0];
            double dy = bi->s[1] - bj->s[1];
            double dist = sqrt(dx * dx + dy * dy);
            double dir = spark_cosine_similarity(bi->c, bj->c, m->d);
            if (dist <= shadow_thresh && dir >= 1.0 - dir_thresh) {
                merged[j] = 1;
                acc_n += (double)bj->n;
                acc_r += bj->r * (double)bj->n;
                acc_a += bj->a * (double)bj->n;
                acc_s[0] += bj->s[0] * (double)bj->n;
                acc_s[1] += bj->s[1] * (double)bj->n;
                acc_S[0] += bj->S[0] * (double)bj->n;
                acc_S[1] += bj->S[1] * (double)bj->n;
                acc_S[2] += bj->S[2] * (double)bj->n;
                acc_S[3] += bj->S[3] * (double)bj->n;
                for (size_t k = 0; k < m->d; ++k) acc_c[k] += bj->c[k] * (double)bj->n;
            }
        }
        double inv_n = 1.0 / (acc_n + 1e-12);
        Bubble nb;
        nb.n = (int)acc_n;
        nb.c = calloc(m->d, sizeof(double));
        for (size_t k = 0; k < m->d; ++k) nb.c[k] = acc_c[k] * inv_n;
        normalize_vec(nb.c, m->d);
        nb.s[0] = acc_s[0] * inv_n;
        nb.s[1] = acc_s[1] * inv_n;
        nb.a = acc_a * inv_n;
        nb.r = fmax(m->r0, acc_r * inv_n);
        nb.S[0] = acc_S[0] * inv_n;
        nb.S[1] = acc_S[1] * inv_n;
        nb.S[2] = acc_S[2] * inv_n;
        nb.S[3] = acc_S[3] * inv_n;
        new_list[new_count++] = nb;
        free(acc_c);
    }
    for (size_t i = 0; i < n; ++i) free(m->bubbles[i].c);
    free(m->bubbles);
    free(merged);
    m->bubbles = new_list;
    m->count = new_count;
    m->capacity = new_cap > 0 ? new_cap : new_count;
}

void bubble_memory_stats(BubbleShadowMemory *m, long *stores, long *recalls, size_t *count) {
    if (stores) *stores = m->stores;
    if (recalls) *recalls = m->recalls;
    if (count) *count = m->count;
}

// ---------- Memory shim ----------
MemoryShim *memory_shim_create(size_t dim, size_t max_bubbles) {
    MemoryShim *s = calloc(1, sizeof(MemoryShim));
    s->dim = dim;
    s->mem = bubble_memory_create(dim, 0.15, 1.0, 0.15, 0.6, 0.75, max_bubbles);
    return s;
}

void memory_shim_free(MemoryShim *shim) {
    if (!shim) return;
    bubble_memory_free(shim->mem);
    free(shim);
}

void memory_shim_store(MemoryShim *shim, const double *x, size_t dim, double utility, double emotion) {
    double emo = emotion;
    if (isnan(emo)) emo = 0.0;
    if (isnan(utility) == 0 && emotion == 0.0) emo = utility;
    bubble_memory_store(shim->mem, x, dim, emo);
}

double memory_shim_project(MemoryShim *shim, const double *x, size_t dim) {
    return bubble_memory_project(shim->mem, x, dim);
}

void memory_shim_forget(MemoryShim *shim, double decay, double min_radius, double min_opacity) {
    if (!shim) return;
    bubble_memory_forget(shim->mem, decay, min_radius, min_opacity);
}

void memory_shim_merge_close(MemoryShim *shim, double shadow_thresh, double dir_thresh) {
    if (!shim) return;
    bubble_memory_merge_close(shim->mem, shadow_thresh, dir_thresh);
}

// ---------- Curiosity ----------
CuriosityModule *curiosity_create(size_t size, MemoryShim *global_memory, const SparkConfig *cfg) {
    CuriosityModule *c = calloc(1, sizeof(CuriosityModule));
    c->buffer_len = 200;
    c->memory = calloc(c->buffer_len * size, sizeof(double));
    c->size = size;
    c->global_memory = global_memory;
    c->local_w = cfg->CUR_LOCAL;
    c->global_w = cfg->CUR_GLOBAL;
    c->proto_w = cfg->CUR_PROTO;
    c->proto_count = 5;
    c->prototypes = calloc(c->proto_count * size, sizeof(double));
    for (size_t p = 0; p < c->proto_count; ++p) {
        for (size_t i = 0; i < size; ++i) c->prototypes[p * size + i] = spark_gauss(0.0, 1.0);
        normalize_vec(&c->prototypes[p * size], size);
    }
    c->novelty_cap = 20;
    c->novelty_history = calloc(c->novelty_cap, sizeof(double));
    c->threshold_min = 0.3;
    return c;
}

void curiosity_free(CuriosityModule *c) {
    if (!c) return;
    free(c->memory);
    free(c->prototypes);
    free(c->novelty_history);
    free(c);
}

static size_t curiosity_memory_size(const CuriosityModule *c) {
    return c->buffer_len;
}

double curiosity_compute(CuriosityModule *c, const double *x) {
    double *nx = calloc(c->size, sizeof(double));
    memcpy(nx, x, c->size * sizeof(double));
    normalize_vec(nx, c->size);
    // local novelty
    double local_n = 1.0;
    size_t mem_sz = curiosity_memory_size(c);
    if (mem_sz > 0 && c->memory[0] != 0.0) {
        double *mean_sim = calloc(mem_sz, sizeof(double));
        size_t count = 0;
        for (size_t k = 0; k < mem_sz; ++k) {
            double *mvec = &c->memory[k * c->size];
            if (spark_norm(mvec, c->size) < 1e-12) continue;
            mean_sim[count++] = cosine_similarity_vec(nx, mvec, c->size);
        }
        if (count > 0) {
            double avg = 0.0;
            for (size_t i = 0; i < count; ++i) avg += mean_sim[i];
            avg /= (double)count;
            local_n = fmax(0.0, 1.0 - avg);
        }
        free(mean_sim);
    }
    double global_proj = memory_shim_project(c->global_memory, nx, c->size);
    double global_n = fmax(0.0, 1.0 - global_proj);
    double proto_n = 1.0;
    double best_proto = -1e9;
    for (size_t p = 0; p < c->proto_count; ++p) {
        double sim = cosine_similarity_vec(nx, &c->prototypes[p * c->size], c->size);
        if (sim > best_proto) best_proto = sim;
    }
    proto_n = fmax(0.0, 1.0 - best_proto);
    double novelty = c->local_w * local_n + c->global_w * global_n + c->proto_w * proto_n;
    // adaptive threshold
    memmove(&c->novelty_history[1], &c->novelty_history[0], (c->novelty_cap - 1) * sizeof(double));
    c->novelty_history[0] = novelty;
    double sum = 0.0;
    for (size_t i = 0; i < c->novelty_cap; ++i) sum += c->novelty_history[i];
    double avg = sum / (double)c->novelty_cap;
    c->threshold_min = avg * 0.5;
    double out = novelty > c->threshold_min ? novelty : 0.0;
    free(nx);
    return out;
}

void curiosity_update_memory(CuriosityModule *c, const double *x) {
    // shift ring
    memmove(&c->memory[c->size], &c->memory[0], (c->buffer_len - 1) * c->size * sizeof(double));
    memcpy(&c->memory[0], x, c->size * sizeof(double));
    memory_shim_store(c->global_memory, x, c->size, 0.0, 0.0);
}

// ---------- Dynamic module ----------
static int dj_oracle_parity(const int *bits, int nbits) {
    int sum = 0;
    for (int i = 0; i < nbits; ++i) sum += bits[i];
    return sum % 2;
}

DynamicModule *dynamic_create(size_t size, const SparkConfig *cfg, QuantumSignalInterface *q) {
    DynamicModule *d = calloc(1, sizeof(DynamicModule));
    d->size = size;
    d->W = calloc(size * size, sizeof(double));
    d->mask = calloc(size * size, sizeof(int));
    d->usage = calloc(size * size, sizeof(double));
    d->alpha = cfg->DYN_BASE_ALPHA;
    d->base_alpha = cfg->DYN_BASE_ALPHA;
    d->fatigue_decay = cfg->DYN_FATIGUE_DECAY;
    d->threshold = cfg->DYN_THRESHOLD;
    d->neighborhood = cfg->DYN_NEIGHBORHOOD;
    d->memory_buffer = calloc(5 * size, sizeof(double));
    d->memory_len = 0;
    d->recent_activations = calloc(10, sizeof(double));
    d->recent_len = 0;
    d->uncertainty = 1.0;
    d->energy = 0.0;
    d->confidence = 1.0;
    d->usage_count = 0;
    d->error_cap = 20;
    d->error_history = calloc(d->error_cap, sizeof(double));
    d->error_len = 0;
    d->last_dj_mod = 1.0;
    d->q = q;
    // initialize mask random
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            int idx = (int)abs((long)i - (long)j);
            double connect = spark_rand_uniform();
            if (connect < cfg->DYN_INIT_CONNECTIVITY && idx <= cfg->DYN_NEIGHBORHOOD) {
                d->mask[i * size + j] = 1;
                d->W[i * size + j] = spark_gauss(0.0, 0.1);
            }
        }
    }
    return d;
}

void dynamic_free(DynamicModule *d) {
    if (!d) return;
    free(d->W);
    free(d->mask);
    free(d->usage);
    free(d->memory_buffer);
    free(d->recent_activations);
    free(d->error_history);
    free(d);
}

static double dyn_prediction_error(DynamicModule *d, const double *x) {
    // pred = tanh(W @ x)
    double err = 0.0;
    double *pred = calloc(d->size, sizeof(double));
    for (size_t i = 0; i < d->size; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < d->size; ++j) {
            acc += d->W[i * d->size + j] * x[j];
        }
        pred[i] = tanh(acc);
    }
    for (size_t i = 0; i < d->size; ++i) {
        double diff = pred[i] - x[i];
        err += diff * diff;
    }
    free(pred);
    return sqrt(err);
}

void dynamic_update(DynamicModule *d, const double *x, size_t dim, double affect, double cognitive_load) {
    double *xn = calloc(dim, sizeof(double));
    memcpy(xn, x, dim * sizeof(double));
    normalize_vec(xn, dim);
    // optional proto-signal via Deutsch-Jozsa to modulate plasticity
    double plasticity_mod = 1.0;
    if (d->q) {
        const char *label = NULL;
        int out_val = 0;
        if (qiface_pseudo_deutsch_jozsa(d->q, dj_oracle_parity, 32, &label, &out_val)) {
            if (label && strcmp(label, "constant") == 0) {
                plasticity_mod = 0.75;
            } else if (label && strcmp(label, "balanced") == 0) {
                plasticity_mod = 1.05;
            }
        }
        d->last_dj_mod = plasticity_mod;
    }
    // novelty vs memory
    double novelty = 1.0;
    if (d->memory_len > 0) {
        double sum = 0.0;
        for (size_t i = 0; i < d->memory_len; ++i) {
            sum += cosine_similarity_vec(xn, &d->memory_buffer[i * dim], dim);
        }
        novelty = fmax(0.0, 1.0 - sum / (double)d->memory_len);
    }
    double fatigue_lvl = 0.0;
    for (size_t i = 0; i < d->size * d->size; ++i) fatigue_lvl += d->usage[i];
    fatigue_lvl = (d->size > 0) ? fatigue_lvl / (double)(d->size * d->size) : 0.0;
    double fatigue_gate = pow(d->fatigue_decay, fatigue_lvl);
    double load_gate = spark_clip(1.0 - 0.3 * spark_clip(cognitive_load, 0.0, 2.0), 0.4, 1.0);
    d->alpha = spark_clip(d->base_alpha * (1.0 + novelty) * fatigue_gate * plasticity_mod * load_gate, 0.01, 0.5);
    double mod_factor = 1.0 + 0.3 * affect;
    // Hebbian update
    double delta_energy = 0.0;
    size_t delta_count = 0;
    for (size_t i = 0; i < d->size; ++i) {
        for (size_t j = 0; j < d->size; ++j) {
            if (!d->mask[i * d->size + j]) continue;
            double outer = xn[i] * xn[j];
            if (fabs(outer) <= d->threshold) continue;
            double fatigue_factor = fmax(0.0, 1.0 - 0.5 * d->usage[i * d->size + j]);
            double delta = outer * d->alpha * mod_factor * fatigue_factor;
            if (i == j || j + 1 == i || i + 1 == j) delta *= 0.7;
            double new_w = spark_clip(d->W[i * d->size + j] + delta, -1.0, 1.0);
            d->W[i * d->size + j] = new_w;
            d->usage[i * d->size + j] += 0.05;
            delta_energy += delta * delta;
            delta_count++;
        }
    }
    for (size_t i = 0; i < d->size * d->size; ++i) d->usage[i] *= 0.95;
    if (delta_count > 0) d->energy += sqrt(delta_energy / (double)delta_count);
    // update memory buffer
    if (d->memory_len < 5) d->memory_len++;
    for (size_t i = 0; i < d->memory_len * dim; ++i) d->memory_buffer[i] *= 0.95;
    memmove(&d->memory_buffer[dim], &d->memory_buffer[0], (d->memory_len - 1) * dim * sizeof(double));
    memcpy(&d->memory_buffer[0], xn, dim * sizeof(double));
    // meta metrics
    double pred_err = dyn_prediction_error(d, xn);
    double mean_act = 0.0;
    for (size_t i = 0; i < d->memory_len; ++i) mean_act += cosine_similarity_vec(xn, &d->memory_buffer[i * dim], dim);
    mean_act /= (double)d->memory_len;
    d->recent_activations[d->recent_len % 10] = mean_act;
    d->recent_len++;
    // quick std estimate
    double mu = 0.0;
    size_t rc = d->recent_len < 10 ? d->recent_len : 10;
    for (size_t i = 0; i < rc; ++i) mu += d->recent_activations[i];
    mu /= (double)rc;
    double var = 0.0;
    for (size_t i = 0; i < rc; ++i) {
        double dv = d->recent_activations[i] - mu;
        var += dv * dv;
    }
    var /= (double)rc;
    d->uncertainty = sqrt(var);
    double conf_update = 1.0 / (1.0 + d->uncertainty + pred_err);
    d->confidence = 0.9 * d->confidence + 0.1 * conf_update;
    if (d->error_len < d->error_cap) d->error_len++;
    memmove(&d->error_history[1], &d->error_history[0], (d->error_len - 1) * sizeof(double));
    d->error_history[0] = pred_err;
    d->usage_count++;
    free(xn);
}

// ---------- Affect ----------
static void affect_push_stimulus(AffectModule *a, const double *x) {
    if (!a || !x) return;
    double *dst = a->stim_buffer;
    size_t dim = a->dim;
    if (a->stim_len < a->buf_len) a->stim_len++;
    memmove(&dst[dim], &dst[0], (a->stim_len - 1) * dim * sizeof(double));
    memcpy(dst, x, dim * sizeof(double));
    memmove(&a->stim_time[1], &a->stim_time[0], (a->stim_len - 1) * sizeof(size_t));
    a->stim_time[0] = a->time;
}

static double affect_novelty(const AffectModule *a, const double *x) {
    if (!a || a->stim_len == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < a->stim_len; ++i) {
        const double *s = &a->stim_buffer[i * a->dim];
        sum += cosine_similarity_vec(x, s, a->dim);
    }
    double avg = sum / (double)a->stim_len;
    return fmax(0.0, 1.0 - avg);
}

AffectModule *affect_create(const SparkConfig *cfg, QuantumSignalInterface *q) {
    AffectModule *a = calloc(1, sizeof(AffectModule));
    a->buf_len = cfg->AFFECT_BUF;
    a->affect_buffer = calloc(a->buf_len, sizeof(double));
    a->decay_tau = cfg->AFFECT_DECAY_TAU;
    a->min_similarity = cfg->AFFECT_MIN_SIM;
    a->dim = cfg->WORKING_DIM;
    a->stim_buffer = calloc(a->buf_len * a->dim, sizeof(double));
    a->stim_time = calloc(a->buf_len, sizeof(size_t));
    a->reward_buffer = calloc(a->buf_len, sizeof(double));
    a->reward_len = 0;
    a->stim_len = 0;
    a->time = 0;
    a->maturation = 0.0;
    a->matur_rate = 1e-4;
    a->running_valence = 0.0;
    a->reward_decay = exp(-1.0 / fmax(1e-6, a->decay_tau));
    a->pending_reward = 0.0;
    a->q = q;
    return a;
}

void affect_free(AffectModule *a) {
    if (!a) return;
    free(a->affect_buffer);
    free(a->stim_buffer);
    free(a->stim_time);
    free(a->reward_buffer);
    free(a);
}

void affect_register_feedback(AffectModule *a, double reward, const double *state, size_t dim) {
    if (!a) return;
    a->pending_reward = reward;
    if (!state || dim == 0) return;
    size_t d = dim < a->dim ? dim : a->dim;
    double *xn = calloc(a->dim, sizeof(double));
    memcpy(xn, state, d * sizeof(double));
    normalize_vec(xn, a->dim);
    // credit distribution to recent stimuli
    double total = 0.0;
    double *weights = calloc(a->stim_len, sizeof(double));
    for (size_t i = 0; i < a->stim_len; ++i) {
        const double *s = &a->stim_buffer[i * a->dim];
        double sim = cosine_similarity_vec(xn, s, a->dim);
        if (sim < a->min_similarity) continue;
        size_t dt = a->time > a->stim_time[i] ? a->time - a->stim_time[i] : 0;
        double decay = exp(-(double)dt / fmax(1e-6, a->decay_tau));
        double w = decay * sim;
        weights[i] = w;
        total += w;
    }
    if (total > 0.0) {
        double accum = 0.0;
        for (size_t i = 0; i < a->stim_len; ++i) {
            if (weights[i] <= 0.0) continue;
            double contrib = reward * (weights[i] / total) * (1.0 - 0.5 * a->maturation);
            accum += contrib;
        }
        a->running_valence = 0.9 * a->running_valence + 0.1 * accum;
    }
    free(weights);
    free(xn);
}

double affect_query(AffectModule *a, const double *x, size_t dim) {
    if (!a) return 0.0;
    size_t d = dim < a->dim ? dim : a->dim;
    double *xn = calloc(a->dim, sizeof(double));
    memcpy(xn, x, d * sizeof(double));
    double norm = spark_norm(xn, a->dim);
    if (norm < 1e-12) {
        free(xn);
        return 0.0;
    }
    normalize_vec(xn, a->dim);
    // push stimulus and advance time
    affect_push_stimulus(a, xn);
    a->time++;
    a->maturation = fmin(1.0, a->maturation + a->matur_rate);
    // novelty and intensity estimators
    double novelty = affect_novelty(a, xn);
    double intensity = tanh(norm);
    // decay running valence and mix recent rewards
    a->running_valence *= a->reward_decay;
    if (a->pending_reward != 0.0) {
        memmove(&a->reward_buffer[1], &a->reward_buffer[0], (a->buf_len - 1) * sizeof(double));
        a->reward_buffer[0] = a->pending_reward;
        if (a->reward_len < a->buf_len) a->reward_len++;
        a->pending_reward = 0.0;
    }
    double weighted = 0.0, wsum = 0.0;
    for (size_t i = 0; i < a->reward_len; ++i) {
        double w = pow(a->reward_decay, (double)i);
        weighted += a->reward_buffer[i] * w;
        wsum += w;
    }
    if (wsum > 0.0) {
        double delta = (weighted / wsum) * (1.0 - 0.5 * a->maturation);
        a->running_valence = 0.9 * a->running_valence + 0.1 * delta;
    }
    double valence = spark_clip(a->running_valence, -1.0, 1.0);
    double affect_signal = intensity * (0.6 * valence + 0.25 * (1.0 - novelty) + 0.15 * valence * (1.0 - a->maturation));
    // smooth over buffer
    memmove(&a->affect_buffer[1], &a->affect_buffer[0], (a->buf_len - 1) * sizeof(double));
    a->affect_buffer[0] = affect_signal;
    double sum = 0.0;
    for (size_t i = 0; i < a->buf_len; ++i) sum += a->affect_buffer[i];
    double baseline = sum / (double)a->buf_len;
    free(xn);
    return baseline;
}

// ---------- Affective predictor ----------
AffectivePredictor *affective_predictor_create(AffectModule *a, MemoryShim *mem) {
    AffectivePredictor *p = calloc(1, sizeof(AffectivePredictor));
    p->affect_module = a;
    p->global_memory = mem;
    return p;
}

void affective_predictor_free(AffectivePredictor *p) {
    free(p);
}

// ---------- Rule adapters ----------
void cg_curiosity_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph) {
    (void)neighbors;
    (void)weights;
    (void)neighbor_count;
    (void)graph;
    if (!g_rule_curiosity) {
        node->curiosity = 0.0;
        return;
    }
    double novelty = curiosity_compute(g_rule_curiosity, node->state);
    curiosity_update_memory(g_rule_curiosity, node->state);
    node->curiosity = novelty;
}

void cg_affect_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph) {
    (void)neighbors;
    (void)weights;
    (void)neighbor_count;
    (void)graph;
    if (!g_rule_affect) {
        node->affect = 0.0;
        return;
    }
    node->affect = affect_query(g_rule_affect, node->state, node->dim);
}

void cg_plasticity_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph) {
    if (g_rule_dynamic) {
        dynamic_update(g_rule_dynamic, node->state, node->dim, node->affect, 0.0);
    }
    // Hebbian on inter-node weights
    for (size_t i = 0; i < neighbor_count; ++i) {
        Node *nb = neighbors[i];
        double dot = 0.0;
        for (size_t k = 0; k < node->dim; ++k) dot += node->state[k] * nb->state[k];
        double hebb = 0.03 * dot;
        cg_update_weight(graph, (size_t)node->id, nb->id, hebb, 1.0);
    }
}

static void attention_rule(Node *node, Node **neighbors, double *weights, size_t neighbor_count, CognitiveGraph *graph) {
    (void)graph;
    double att = 0.0;
    for (size_t i = 0; i < neighbor_count; ++i) att += fabs(weights[i]);
    node->attention = att;
    double mod = cg_get_modulator(graph, "attention", 1.0);
    if (!isfinite(mod)) mod = 1.0;
    for (size_t k = 0; k < node->dim; ++k) node->state[k] *= mod;
}

// ---------- Build spark system ----------
SparkSystem *build_spark_system(int seed) {
    spark_set_seed((uint32_t)seed);
    SparkConfig cfg = spark_default_config();
    char *env_nodes = getenv("SPARK_NODES");
    size_t num_nodes = 200;
    if (env_nodes) {
        long v = strtol(env_nodes, NULL, 10);
        if (v > 4 && v < 10000) num_nodes = (size_t)v;
    }
    QuantumSignalInterface *q = NULL;
    char *env_no_q = getenv("SPARK_NO_QUANTUM");
    if (!env_no_q) {
        QuantumConfig qc = {5, 1.0, 5.0, 128, 0};
        q = qiface_create(qc);
        qiface_start(q);
    }

    MemoryShim *global_mem = memory_shim_create(cfg.WORKING_DIM, cfg.GLOBAL_MAXLEN);
    CuriosityModule *curiosity = curiosity_create(cfg.WORKING_DIM, global_mem, &cfg);
    DynamicModule *dyn = dynamic_create(cfg.WORKING_DIM, &cfg, q);
    AffectModule *affect = affect_create(&cfg, q);
    AffectivePredictor *aff_pred = affective_predictor_create(affect, global_mem);

    CognitiveGraph *graph = cg_create(num_nodes, cfg.WORKING_DIM, 0.05, 3, 3, 0.4, 0.5);
    for (int i = 0; i < 4 && i < (int)num_nodes; ++i) cg_connect(graph, (size_t)i, (size_t)((i + 1) % num_nodes), 0.1);

    SparkSystem *sys = calloc(1, sizeof(SparkSystem));
    sys->curiosity = curiosity;
    sys->dynamic_module = dyn;
    sys->affect_module = affect;
    sys->global_memory = global_mem;
    sys->qiface = q;
    sys->affective_predictor = aff_pred;
    sys->graph = graph;
    sys->cfg = cfg;
    register_rule_context(sys);
    return sys;
}

void spark_system_free(SparkSystem *sys) {
    if (!sys) return;
    curiosity_free(sys->curiosity);
    dynamic_free(sys->dynamic_module);
    affect_free(sys->affect_module);
    affective_predictor_free(sys->affective_predictor);
    memory_shim_free(sys->global_memory);
    qiface_stop(sys->qiface);
    qiface_free(sys->qiface);
    cg_free(sys->graph);
    free(sys);
}

void inject_and_propagate_c(SparkSystem *sys, size_t *indices, double **patterns, size_t count, int steps) {
    for (size_t i = 0; i < count; ++i) {
        size_t idx = indices[i];
        double *vec = patterns[i];
        cg_inject(sys->graph, idx, vec);
    }
    RuleFn rules[4];
    rules[0] = cg_curiosity_rule;
    rules[1] = cg_affect_rule;
    rules[2] = cg_plasticity_rule;
    rules[3] = attention_rule;
    cg_propagate_steps(sys->graph, rules, 4, steps);
}

size_t spark_graph_num_nodes(const SparkSystem *sys) {
    return sys && sys->graph ? sys->graph->num_nodes : 0;
}

size_t spark_graph_state_dim(const SparkSystem *sys) {
    return sys && sys->graph ? sys->graph->state_dim : 0;
}

void spark_graph_snapshot(const SparkSystem *sys, double *out_matrix) {
    if (!sys || !sys->graph || !out_matrix) return;
    cg_snapshot_states(sys->graph, out_matrix);
}

void spark_global_memory_stats(const SparkSystem *sys, long *stores, long *recalls, size_t *count) {
    if (!sys || !sys->global_memory) return;
    bubble_memory_stats(sys->global_memory->mem, stores, recalls, count);
}
