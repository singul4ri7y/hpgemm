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

/* Position of a thread in the thread grid. The grid is `jc x ic x jr`, 5th,
   3rd and 2nd loops are parallelized (BLIS way). 4th loop carries the
   dependency over `z` and 1st loop is too fine grained. */
typedef struct {
    int jc_id, jc_nt;
    int ic_id, ic_nt;
    int jr_id, jr_nt;

    /* Thread index and count inside a `jc` group, the group shares `y_packed`. */
    int gid, gnt;
} hpgemm_thread_t;


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


/* Split `[0, n)` among `nt` threads in multiples of `unit`. Get the range of
   thread `id`. */
static inline void hpgemm_split(
    int n, int unit, int nt, int id,
    int *start, int *end
) {
    int chunk = (n + nt - 1) / nt;
    chunk = (chunk + unit - 1) / unit * unit;

    *start = min(id * chunk, n);
    *end = min(*start + chunk, n);
}

/* Factor the threads into `jc x ic x jr` grid. Priority is `ic`, then `jc`,
   and the rest goes to `jr`. Bounded by the available blocks. */
static void hpgemm_thread_grid(hpgemm_thread_t *t, int m, int n) {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int ic_blocks = (m + hpgemm_mc - 1) / hpgemm_mc;
    int jc_blocks = (n + hpgemm_nc - 1) / hpgemm_nc;
    int ic = 1, jc = 1;

    for(int d = 1; d <= min(nt, ic_blocks); d++) {
        if(nt % d == 0)
            ic = d;
    }

    for(int d = 1; d <= min(nt / ic, jc_blocks); d++) {
        if((nt / ic) % d == 0)
            jc = d;
    }

    t->ic_nt = ic;
    t->jc_nt = jc;
    t->jr_nt = nt / ic / jc;

    t->gnt = t->ic_nt * t->jr_nt;
    t->gid = tid % t->gnt;

    t->jc_id = tid / t->gnt;
    t->ic_id = t->gid / t->jr_nt;
    t->jr_id = t->gid % t->jr_nt;
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
    const hpgemm_thread_t *t,
    int m, int n, int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    int nr = hpgemm_nr, js, je;

    /* Split the KCxNR micro-panels among `jr` threads. */
    hpgemm_split(n, nr, t->jr_nt, t->jr_id, &js, &je);

    /* 2nd outer loop to slice panel KCxNC and `z` into KCxNR micro-panel. */
    for(int j = js; j < je; j += nr) {
        int jb = min(nr, je - j);
        /* Dispatch to loop 1. */
        hpgemm_loop1(m, jb, k, beta, &Z(0, j), ld_z, alpha, x_packed,
            &y_packed[k * j]);
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop3(
    const hpgemm_thread_t *t,
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
    int mc = hpgemm_mc, is, ie;

    /* Split `m` among `ic` threads. Each thread owns its `x_packed`. */
    hpgemm_split(m, hpgemm_mr, t->ic_nt, t->ic_id, &is, &ie);

    /* 3rd outer loop to slice `x` and `z` by MC over m-dimension. */
    for(int i = is; i < ie; i += mc) {
        int ib = min(mc, ie - i);
        /* Pack `MCxKC` block. */
        pack(k, ib, hpgemm_mr, transp_x ? &X(0, i) : &X(i, 0), ld_x, x_packed,
            !transp_x);

        /* Dispatch to 2nd outer loop. */
        hpgemm_loop2(t, ib, n, k, beta, &Z(i, 0), ld_z, alpha, x_packed,
            y_packed);
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop4(
    const hpgemm_thread_t *t,
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

        /* Pack the panel KCxNC. */
        /* All threads of the `jc` group take part, each packs a few KCxNR
           micro-panels. */
        int qs, qe;
        hpgemm_split(panels, 1, t->gnt, t->gid, &qs, &qe);

        int cs = qs * nr;
        int ce = min(qe * nr, n);
        if(cs < ce)
            pack(pb, ce - cs, nr, transp_y ? &Y(cs, p) : &Y(p, cs), ld_y,
                &y_packed[cs * pb], transp_y);

        /* `y_packed` must be complete before anyone consumes it. */
        #pragma omp barrier

        /* Dispatch to loop 3. */
        /* NOTE: `z` must be scaled by `beta` only once. */
        hpgemm_loop3(t, transp_x, m, n, pb, p == 0 ? beta : 1.0f, z, ld_z,
            alpha, transp_x ? &X(p, 0) : &X(0, p), ld_x, y_packed, x_packed);

        /* Everyone must finish before `y_packed` is overwritten. */
        #pragma omp barrier
    }
}


HPGEMM_FLATTEN HPGEMM_HOT static inline void hpgemm_loop5(
    const hpgemm_thread_t *t,
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
    int nc = hpgemm_nc, js, je, s0, e0;

    /* Split `n` among `jc` groups. */
    hpgemm_split(n, hpgemm_nr, t->jc_nt, t->jc_id, &js, &je);

    /* All the groups must iterate equally (a group may even be empty), as
       barriers in 4th loop are shared by the whole team. Group 0 always has the
       largest share. */
    hpgemm_split(n, hpgemm_nr, t->jc_nt, 0, &s0, &e0);
    int iters = (e0 - s0 + nc - 1) / nc;

    /* 5th outer loop to slice `z` and `y` by NC over n-dimension. */
    for(int it = 0; it < iters; it++) {
        int j = min(js + it * nc, je);
        int jb = min(nc, je - j);

        /* Dispatch to 4th outer loop. */
        hpgemm_loop4(t, transp_x, transp_y, m, jb, k, beta, &Z(0, j), ld_z,
            alpha, x, ld_x, transp_y ? &Y(j, 0) : &Y(0, j), ld_y, x_packed,
            y_packed);
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
        hpgemm_init();

    float *y_packed_all = NULL;

    #pragma omp parallel
    {
        hpgemm_thread_t t;
        hpgemm_thread_grid(&t, m, n);

        /* Memory allocation for `MCxKC` block of `x` and `KCxNC` panel of `y`. */
        /* NOTE: `x` block is private, `y` panel is one per `jc` group. Packing
           pads upto MR and NR. */
        size_t x_size = (size_t) (hpgemm_mc + hpgemm_mr) * hpgemm_kc;
        size_t y_size = (size_t) (hpgemm_nc + hpgemm_nr) * hpgemm_kc;

        float *x_packed = hpgemm_malloc(x_size * sizeof(float), 64);

        /* Implicit barrier at the end of `single`. */
        #pragma omp single
        y_packed_all = hpgemm_malloc(t.jc_nt * y_size * sizeof(float), 64);

        /* Dispatch to loop 5. */
        hpgemm_loop5(&t, transp_x, transp_y, m, n, k, beta, z, ld_z, alpha, x,
            ld_x, y, ld_y, x_packed, y_packed_all + t.jc_id * y_size);

        hpgemm_free(x_packed);
    }

    hpgemm_free(y_packed_all);
}
