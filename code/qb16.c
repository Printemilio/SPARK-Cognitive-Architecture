#define _POSIX_C_SOURCE 199309L
// Spark_Core/qb16.c
#include "qb16.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include "spark_utils.h"

typedef struct {
    QuantumSignalInterface *q;
    int index;
} QubitArgs;

static void sleep_seconds(double seconds) {
    if (seconds <= 0.0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)seconds;
    ts.tv_nsec = (long)((seconds - ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

static void *qubit_loop(void *arg) {
    QubitArgs *qa = (QubitArgs *)arg;
    QuantumSignalInterface *q = qa->q;
    int idx = qa->index;
    double freq = spark_rand_uniform() * (q->cfg.MAX_FREQ - q->cfg.MIN_FREQ) + q->cfg.MIN_FREQ;
    double period = 1.0 / fmax(freq, 1e-6);
    double initial_delay = spark_rand_uniform() * period;
    sleep_seconds(initial_delay);
    while (!q->stop_flag) {
        q->states[idx] = 1 - q->states[idx];
        sleep_seconds(period);
    }
    free(qa);
    return NULL;
}

static void *sampler_loop(void *arg) {
    QuantumSignalInterface *q = (QuantumSignalInterface *)arg;
    while (!q->stop_flag) {
        if (q->history_len + 1 > q->history_cap) {
            int cap = q->history_cap ? q->history_cap * 2 : q->cfg.HISTORY_LEN;
            q->history = realloc(q->history, cap * sizeof(int *));
            q->history_cap = cap;
        }
        int *snap = malloc(q->cfg.NUM_QUBITS * sizeof(int));
        for (int i = 0; i < q->cfg.NUM_QUBITS; ++i) snap[i] = q->states[i];
        q->history[q->history_len++] = snap;
        if (q->history_len > q->cfg.HISTORY_LEN) {
            free(q->history[0]);
            memmove(&q->history[0], &q->history[1], (q->history_len - 1) * sizeof(int *));
            q->history_len--;
        }
        if (q->cfg.PRINT_LOOP) {
            printf("Qubits: ");
            for (int i = 0; i < q->cfg.NUM_QUBITS; ++i) printf("%d", q->states[i]);
            printf("\n");
        }
        sleep_seconds(0.01);
    }
    return NULL;
}

QuantumSignalInterface *qiface_create(QuantumConfig cfg) {
    QuantumSignalInterface *q = calloc(1, sizeof(QuantumSignalInterface));
    q->cfg = cfg;
    q->states = calloc((size_t)cfg.NUM_QUBITS, sizeof(int));
    q->threads = NULL;
    q->thread_count = 0;
    q->stop_flag = 0;
    q->history = NULL;
    q->history_len = 0;
    q->history_cap = 0;
    return q;
}

void qiface_free(QuantumSignalInterface *q) {
    if (!q) return;
    qiface_stop(q);
    free(q->states);
    for (int i = 0; i < q->history_len; ++i) free(q->history[i]);
    free(q->history);
    free(q->threads);
    free(q);
}

void qiface_start(QuantumSignalInterface *q) {
    if (!q || q->running) return;
    q->stop_flag = 0;
    q->thread_count = (size_t)(q->cfg.NUM_QUBITS + 1);
    q->threads = calloc(q->thread_count, sizeof(void *));
    for (int i = 0; i < q->cfg.NUM_QUBITS; ++i) {
        pthread_t *t = malloc(sizeof(pthread_t));
        QubitArgs *qa = malloc(sizeof(QubitArgs));
        qa->q = q;
        qa->index = i;
        pthread_create(t, NULL, qubit_loop, qa);
        q->threads[i] = t;
    }
    pthread_t *sampler = malloc(sizeof(pthread_t));
    pthread_create(sampler, NULL, sampler_loop, q);
    q->threads[q->thread_count - 1] = sampler;
    q->running = 1;
}

void qiface_stop(QuantumSignalInterface *q) {
    if (!q || !q->running) return;
    q->stop_flag = 1;
    for (size_t i = 0; i < q->thread_count; ++i) {
        pthread_t *t = (pthread_t *)q->threads[i];
        if (t) {
            pthread_join(*t, NULL);
            free(t);
        }
    }
    free(q->threads);
    q->threads = NULL;
    q->thread_count = 0;
    q->running = 0;
}

void qiface_get_state_snapshot(QuantumSignalInterface *q, int *out_states) {
    for (int i = 0; i < q->cfg.NUM_QUBITS; ++i) {
        out_states[i] = q->states[i];
    }
}

int qiface_pseudo_qft_multiwindow(QuantumSignalInterface *q, int windows, int step, double ***out_mag, double ***out_phase, int *out_windows, int *out_bins) {
    // Simplified FFT using DFT (costly but avoids external deps)
    if (q->history_len < q->cfg.HISTORY_LEN) return 0;
    int W = windows;
    int bins = step / 2;
    *out_windows = W;
    *out_bins = bins;
    *out_mag = (double **)calloc(W, sizeof(double *));
    *out_phase = (double **)calloc(W, sizeof(double *));
    for (int w = 0; w < W; ++w) {
        int start = w * step;
        int end = start + step;
        if (end > q->cfg.HISTORY_LEN) break;
        double *mag = (double *)calloc((size_t)bins * q->cfg.NUM_QUBITS, sizeof(double));
        double *phase = (double *)calloc((size_t)bins * q->cfg.NUM_QUBITS, sizeof(double));
        for (int qb = 0; qb < q->cfg.NUM_QUBITS; ++qb) {
            for (int k = 0; k < bins; ++k) {
                double real = 0.0, imag = 0.0;
                for (int n = 0; n < step; ++n) {
                    double val = q->history[start + n][qb];
                    double angle = -2.0 * M_PI * k * n / step;
                    real += val * cos(angle);
                    imag += val * sin(angle);
                }
                double m = sqrt(real * real + imag * imag);
                double ph = atan2(imag, real);
                mag[qb * bins + k] = m;
                phase[qb * bins + k] = ph;
            }
        }
        (*out_mag)[w] = mag;
        (*out_phase)[w] = phase;
    }
    return 1;
}

int qiface_pseudo_deutsch_jozsa(QuantumSignalInterface *q, int (*oracle)(const int *, int), int trials, const char **out_label, int *out_value) {
    int unique_outputs[3] = {0, 0, 0};
    int unique_count = 0;
    for (int t = 0; t < trials; ++t) {
        int state[32];
        qiface_get_state_snapshot(q, state);
        int val = oracle(state, q->cfg.NUM_QUBITS);
        int known = 0;
        for (int i = 0; i < unique_count; ++i) {
            if (unique_outputs[i] == val) {
                known = 1;
                break;
            }
        }
        if (!known && unique_count < 3) {
            unique_outputs[unique_count++] = val;
        }
        sleep_seconds(0.002);
    }
    if (unique_count == 1) {
        *out_label = "constant";
        *out_value = unique_outputs[0];
        return 1;
    }
    if (unique_count == 2) {
        *out_label = "balanced";
        *out_value = unique_outputs[0]; // representative
        return 1;
    }
    *out_label = "unknown";
    *out_value = 0;
    return 1;
}

int qiface_pseudo_amplitude_estimation(QuantumSignalInterface *q, const char *pattern, int trials, double *constructive, double *destructive) {
    int pattern_len = (int)strlen(pattern);
    int matches = 0, total = trials > 0 ? trials : 1;
    for (int t = 0; t < total; ++t) {
        int state_buf[32];
        qiface_get_state_snapshot(q, state_buf);
        int found = 0;
        for (int i = 0; i <= q->cfg.NUM_QUBITS - pattern_len; ++i) {
            int ok = 1;
            for (int j = 0; j < pattern_len; ++j) {
                if ((pattern[j] - '0') != state_buf[i + j]) {
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                found = 1;
                break;
            }
        }
        if (found) matches++;
        sleep_seconds(0.002);
    }
    *constructive = matches / (double)total;
    *destructive = (total - matches) / (double)total;
    return 1;
}
