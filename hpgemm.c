#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "hpgemm.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads()    1
#define omp_get_thread_num()     0
#endif  // _OPENMP

#define min(x, y)    ((x) < (y) ? (x) : (y))
/* Get an element from matrices in row-major order. */
#define X(i, j)      x[(i) * ld_x + (j)]
#define Y(i, j)      y[(i) * ld_y + (j)]
#define Z(i, j)      z[(i) * ld_z + (j)]


int hpgemm_mc = MC;
int hpgemm_kc = KC;
int hpgemm_nc = NC;


void *hpgemm_malloc(size_t size, size_t align) {
    /* Over-allocate and stash the original pointer just before the aligned
       one. Pure C99, hence works everywhere. */
    void *raw = malloc(size + align + sizeof(void *));
    if(raw == NULL)
        return NULL;

    uintptr_t addr = ((uintptr_t) raw + sizeof(void *) + align - 1) &
        ~((uintptr_t) align - 1);
    ((void **) addr)[-1] = raw;

    return (void *) addr;
}

void hpgemm_free(void *ptr) {
    if(ptr != NULL)
        free(((void **) ptr)[-1]);
}


/* NOTE: This packing subroutine favors the row-major access pattern of
 *   matrices while packing (say, when packing for matrix `y`). Hence, to pack
 *   matrix `x`, leverage the `transpose` parameter, because due to outer
 *   product computation in the microkernel, the `x` access pattern is reversed.
 */
HPGEMM_HOT static inline void pack(
    int k, int xn, int stride,
    float *restrict x,
    int ld_x,
    float *restrict x_packed,
    bool transpose
) {
    /* `stride` is either `MR` or `NR`. Try prefetching beforehand. */
    HPGEMM_PREFETCH(x_packed, 1, 1);
    HPGEMM_PREFETCH(x, 0, 1);

    for(int ir = 0; ir < xn; ir += stride) {
        int irb = min(stride, xn - ir);
        for(int p = 0; p < k; p++) {
            int i = 0;

            HPGEMM_UNROLL8
            for(; i < irb; i++)
                *x_packed++ = transpose ? X(ir + i, p) : X(p, ir + i);

            /* Fill remainders with 0. */
            for(; i < stride; i++)
                *x_packed++ = 0.0f;
        }
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop1(
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    hpgemm_ukernel_t ukernel = hpgemm_ukernel;
    int mr = hpgemm_mr, nr = hpgemm_nr;

    /* 1st outer loop to slice block MCxKC and `z` into MRxKC micro-panel. */
    for(int i = 0; i < m; i += mr) {
        int ib = min(mr, m - i);

        if(HPGEMM_LIKELY(ib == mr && n == nr))
            /* Dispatch to microkernel. */
            ukernel(k, beta, &Z(i, 0), ld_z, alpha, &x_packed[i * k],
                y_packed);
        else {
            float dest_shell[MR_MAX * NR_MAX] HPGEMM_ALIGNAS(64);

            /* Copy original C. */
            if(HPGEMM_UNLIKELY(beta != 0.0f)) {
                for(int ir = 0; ir < ib; ir++) {
                    for(int jr = 0; jr < n; jr++)
                        dest_shell[ir * nr + jr] = Z(i + ir, jr);
                }
            }

            ukernel(k, beta, dest_shell, nr, alpha, &x_packed[i * k],
                y_packed);

            /* Copy shell data back to `z`. */
            for(int ir = 0; ir < ib; ir++) {
                for(int jr = 0; jr < n; jr++)
                    Z(i + ir, jr) = dest_shell[ir * nr + jr];
            }
        }
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop2(
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    int nr = hpgemm_nr;

    /* 2nd outer loop to slice panel KCxNC and `z` into KCxNR micro-panel. */
    for(int j = 0; j < n; j += nr) {
        int jb = min(nr, n - j);
        /* Dispatch to loop 1. */
        hpgemm_loop1(m, jb, k, beta, &Z(0, j), ld_z, alpha, x_packed,
            &y_packed[k * j]);
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop3(
    bool transp_x,
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x,
    int ld_x,
    float *restrict y_packed,
    float *restrict x_packed
) {
    int mc = hpgemm_mc, mr = hpgemm_mr;
    int nt = omp_get_num_threads(), id = omp_get_thread_num();

    /* Split `m` evenly among threads in multiples of MR, so every thread gets
       an equal share regardless of how many MC blocks `m` spans. */
    int chunk = ((m + nt - 1) / nt + mr - 1) / mr * mr;
    int is = min(id * chunk, m);
    int ie = min(is + chunk, m);

    /* 3rd outer loop to slice `x` and `z` by MC over m-dimension. Each thread
       packs into its own `x_packed`. */
    for(int i = is; i < ie; i += mc) {
        int ib = min(mc, ie - i);

        /* Pack `MCxKC` block. */
        pack(k, ib, mr, transp_x ? &X(0, i) : &X(i, 0), ld_x, x_packed,
            !transp_x);

        /* Dispatch to 2nd outer loop. */
        hpgemm_loop2(ib, n, k, beta, &Z(i, 0), ld_z, alpha, x_packed,
            y_packed);
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop4(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x,
    int ld_x,
    float *restrict y,
    int ld_y,
    float *restrict x_packed,
    float *restrict y_packed
) {
    int nr = hpgemm_nr;
    int panels = (n + nr - 1) / nr;

    /* 4th outer loop to slice both `x` and `y` by KC over k-dimension. */
    for(int p = 0; p < k; p += hpgemm_kc) {
        int pb = min(hpgemm_kc, k - p);

        /* Pack the panel KCxNC, threads share the KCxNR micro-panels. The
           implicit barrier makes `y_packed` complete before anyone uses it. */
        #pragma omp for schedule(static)
        for(int q = 0; q < panels; q++) {
            int c = q * nr;
            pack(pb, min(nr, n - c), nr, transp_y ? &Y(c, p) : &Y(p, c), ld_y,
                &y_packed[c * pb], transp_y);
        }

        /* Dispatch to loop 3. */
        /* NOTE: `z` must be scaled by `beta` only once. */
        hpgemm_loop3(transp_x, m, n, pb, p == 0 ? beta : 1.0f, z, ld_z, alpha,
            transp_x ? &X(p, 0) : &X(0, p), ld_x, y_packed, x_packed);

        /* Everyone must finish before `y_packed` is overwritten. */
        #pragma omp barrier
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop5(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x,
    int ld_x,
    float *restrict y,
    int ld_y,
    float *restrict x_packed,
    float *restrict y_packed
) {
    int nc = hpgemm_nc;

    /* 5th outer loop to slice `z` and `y` by NC over n-dimension. Every thread
       runs it redundantly, so the worksharing loops below stay in sync. */
    for(int j = 0; j < n; j += nc) {
        int jb = min(nc, n - j);

        /* Dispatch to 4th outer loop. */
        hpgemm_loop4(transp_x, transp_y, m, jb, k, beta, &Z(0, j), ld_z, alpha,
            x, ld_x, transp_y ? &Y(j, 0) : &Y(0, j), ld_y, x_packed, y_packed);
    }
}


void hpgemm(
    bool transp_x, bool transp_y,
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x,
    int ld_x,
    float *restrict y,
    int ld_y
) {
    if(HPGEMM_UNLIKELY(hpgemm_ukernel == NULL))
        kernel_init();

    float *y_packed = NULL;

    #pragma omp parallel
    {
        /* Memory allocation for `MCxKC` block of `x` and `KCxNC` panel of `y`.
           NOTE: `x` block is private, `y` panel is shared. Packing pads upto
           MR and NR. */
        size_t x_size = (size_t) (hpgemm_mc + hpgemm_mr) * hpgemm_kc;
        size_t y_size = (size_t) (hpgemm_nc + hpgemm_nr) * hpgemm_kc;

        float *x_packed = hpgemm_malloc(x_size * sizeof(float), 64);

        /* Implicit barrier at the end of `single`. */
        #pragma omp single
        y_packed = hpgemm_malloc(y_size * sizeof(float), 64);

        /* Dispatch to loop 5. */
        hpgemm_loop5(transp_x, transp_y, m, n, k, beta, z, ld_z, alpha, x, ld_x,
            y, ld_y, x_packed, y_packed);

        hpgemm_free(x_packed);
    }

    hpgemm_free(y_packed);
}
