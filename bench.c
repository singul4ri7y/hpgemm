#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>
#include <mkl.h>

#include "hpgemm.h"

/* Create a random matrix. */
static float *new_mat(int r, int c) {
    float *mat = hpgemm_malloc((size_t) r * c * sizeof(float), 64);
    for(int i = 0; i < r * c; i++)
        mat[i] = (float) rand() / RAND_MAX - 0.5f;

    return mat;
}

/* Bench MKL and hp-gemm with `nt` threads (dimension vs GFLOPS). The data is
   written to `data/perf/{mkl,hpgemm}_<tag>.dat`, two columns, gnuplot ready. */
static void bench(
    const char *tag, int nt,
    int repeat, int start, int end, int step
) {
    char path[64];

    snprintf(path, sizeof(path), "data/perf/mkl_%s.dat", tag);
    FILE *mklf = fopen(path, "w");
    snprintf(path, sizeof(path), "data/perf/hpgemm_%s.dat", tag);
    FILE *hpf = fopen(path, "w");
    if(mklf == NULL || hpf == NULL) {
        printf("Cannot open data files, does `data/perf` exist?\n");
        return;
    }

    omp_set_num_threads(nt);
    mkl_set_num_threads(nt);

    for(int dim = start; dim <= end; dim += step) {
        double gflops = 2 * (double) dim * dim * dim * 1e-9;

        /* Create matrices. */
        float *A = new_mat(dim, dim);
        float *B = new_mat(dim, dim);
        float *C = new_mat(dim, dim);
        float *Cref = new_mat(dim, dim);

        double mkl_best = INFINITY, hp_best = INFINITY;

        /* Intel MKL. */
        for(int i = 0; i < repeat; i++) {
            double ttook = omp_get_wtime();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim,
                dim, 1.0f, A, dim, B, dim, 0.0f, Cref, dim);
            ttook = omp_get_wtime() - ttook;

            if(ttook < mkl_best)
                mkl_best = ttook;
        }

        /* hp-gemm. */
        for(int i = 0; i < repeat; i++) {
            double ttook = omp_get_wtime();
            hpgemm(false, false, dim, dim, dim, 0.0f, C, dim, 1.0f, A, dim, B,
                dim);
            ttook = omp_get_wtime() - ttook;

            if(ttook < hp_best)
                hp_best = ttook;
        }

        /* Check for correctness against MKL. */
        float err = 0.0f;
        for(int i = 0; i < dim * dim; i++) {
            float diff = fabsf(C[i] - Cref[i]) / (fabsf(Cref[i]) + 1.0f);
            if(diff > err)
                err = diff;
        }

        printf("[%s] nxn=%d, MKL = %lf GFLOPS, hp-gemm = %lf GFLOPS, err = %g%s\n",
            tag, dim, gflops / mkl_best, gflops / hp_best, err,
            err > 1e-3f ? " (MISMATCH)" : "");

        fprintf(mklf, "%d    %lf\n", dim, gflops / mkl_best);
        fprintf(hpf, "%d    %lf\n", dim, gflops / hp_best);
        fflush(mklf);
        fflush(hpf);

        hpgemm_free(A);
        hpgemm_free(B);
        hpgemm_free(C);
        hpgemm_free(Cref);
    }

    fclose(mklf);
    fclose(hpf);
}

int main(int argc, char **argv) {
    if(argc < 5) {
        printf("Invalid argument count. <repeat_operations> <start_dim> <end_dim> <dim_step>\n");
        return -1;
    }

    int repeat = atoi(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);

    kernel_init();
    printf("Kernel: %s (%dx%d)\n", hpgemm_kernel_name, hpgemm_mr, hpgemm_nr);

    int nt = omp_get_max_threads();

    /* Single core and multithreaded. */
    bench("1t", 1, repeat, start, end, step);
    bench("mt", nt, repeat, start, end, step);

    return 0;
}
