#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#include "hpgemm.h"

static float *A, *B, *C;

/* GFLOPS of the current blocking parameters, averaged over the dimensions. */
static double avg_gflops(int repeat, int start, int end, int step) {
    double total = 0.0;
    int count = 0;

    for(int dim = start; dim <= end; dim += step, count++) {
        double ttook_best = INFINITY;

        for(int i = 0; i < repeat; i++) {
            double ttook = omp_get_wtime();
            hpgemm(false, false, dim, dim, dim, 0.0f, C, dim, 1.0f, A, dim, B,
                dim);
            ttook = omp_get_wtime() - ttook;

            if(ttook < ttook_best)
                ttook_best = ttook;
        }

        total += 2 * (double) dim * dim * dim * 1e-9 / ttook_best;
    }

    return total / count;
}

int main(int argc, char **argv) {
    if(argc < 6) {
        printf("Invalid argument count. <mckc|nc> <repeat_operations> <start_dim> <end_dim> <dim_step>\n");
        return -1;
    }

    int repeat = atoi(argv[2]);
    int start = atoi(argv[3]);
    int end = atoi(argv[4]);
    int step = atoi(argv[5]);

    /* Cache blocking is per core, so find it on a single thread. */
    omp_set_num_threads(1);
    kernel_init();

    /* Matrices are shared by all the runs, `hpgemm()` overwrites `C`. */
    A = hpgemm_malloc((size_t) end * end * sizeof(float), 64);
    B = hpgemm_malloc((size_t) end * end * sizeof(float), 64);
    C = hpgemm_malloc((size_t) end * end * sizeof(float), 64);
    for(int i = 0; i < end * end; i++) {
        A[i] = (float) rand() / RAND_MAX - 0.5f;
        B[i] = (float) rand() / RAND_MAX - 0.5f;
    }

    double best = 0.0;

    if(strcmp(argv[1], "mckc") == 0) {
        FILE *f = fopen("data/empirical/mckc.dat", "w");
        if(f == NULL)
            return -2;

        int best_mc = 0, best_kc = 0;

        /* NC stays default. MC steps by multiple of MR. */
        for(int kc = 32; kc <= 640; kc += 32) {
            for(int mc = hpgemm_mr * 8; mc <= 1024; mc += hpgemm_mr * 8) {
                hpgemm_mc = mc;
                hpgemm_kc = kc;

                double gflops = avg_gflops(repeat, start, end, step);
                fprintf(f, "%d %d %lf\n", kc, mc, gflops);
                fflush(f);

                if(gflops > best) {
                    best = gflops;
                    best_mc = mc;
                    best_kc = kc;
                }
            }

            /* Blank line between scans, so `splot` can draw the surface. */
            fprintf(f, "\n");
        }

        fclose(f);
        printf("Best MCxKC = %dx%d, GFLOPS = %lf\n", best_mc, best_kc, best);
    } else if(strcmp(argv[1], "nc") == 0) {
        FILE *f = fopen("data/empirical/nc.dat", "w");
        if(f == NULL)
            return -2;

        int best_nc = 0;

        /* MC and KC stay default. NC steps by multiple of NR. The effect is
           visible only when the dimensions are larger than NC. */
        for(int nc = hpgemm_nr * 16; nc <= 8192; nc += hpgemm_nr * 16) {
            hpgemm_nc = nc;

            double gflops = avg_gflops(repeat, start, end, step);
            fprintf(f, "%d %lf\n", nc, gflops);
            fflush(f);

            if(gflops > best) {
                best = gflops;
                best_nc = nc;
            }
        }

        fclose(f);
        printf("Best NC = %d, GFLOPS = %lf\n", best_nc, best);
    } else {
        printf("Unknown case, use `mckc` or `nc`.\n");
        return -1;
    }

    hpgemm_free(A);
    hpgemm_free(B);
    hpgemm_free(C);

    return 0;
}
