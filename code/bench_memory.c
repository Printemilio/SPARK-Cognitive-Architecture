#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "spark18.h"
#include "spark_utils.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_gauss(double *v, size_t n) {
    for (size_t i = 0; i < n; ++i) v[i] = spark_gauss(0.0, 1.0);
}

int main(int argc, char **argv) {
    size_t dim = 30;
    size_t iters = 20000;
    if (argc > 1) iters = (size_t)strtoul(argv[1], NULL, 10);
    spark_set_seed(42);
    BubbleShadowMemory *m = bubble_memory_create(dim, 0.15, 1.0, 0.15, 0.6, 0.75, 400);
    double *x = calloc(dim, sizeof(double));
    double start = now_sec();
    for (size_t i = 0; i < iters; ++i) {
        fill_gauss(x, dim);
        bubble_memory_store(m, x, dim, 0.1);
        if ((i & 3) == 0) {
            bubble_memory_project(m, x, dim);
        }
    }
    double elapsed = now_sec() - start;
    printf("C bubble_memory: iters=%zu dim=%zu time=%.3f ms (%.2f Kops/s)\n",
           iters, dim, elapsed * 1000.0, (iters / fmax(1e-9, elapsed)) / 1000.0);
    free(x);
    bubble_memory_free(m);
    return 0;
}
