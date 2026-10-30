#ifndef HPGEMM_H
#define HPGEMM_H

#include <stdbool.h>
#include <stddef.h>


#if defined(__GNUC__)
#define HPGEMM_FLATTEN                 __attribute__((flatten))
#define HPGEMM_HOT                     __attribute__((hot))

#define HPGEMM_ALIGNAS(b)              __attribute__((aligned(b)))
#define HPGEMM_LIKELY(c)               __builtin_expect(!!(c), 1)
#define HPGEMM_UNLIKELY(c)             __builtin_expect(!!(c), 0)
#define HPGEMM_PREFETCH(p, rw, loc)    __builtin_prefetch(p, rw, loc)
#else
#define HPGEMM_FLATTEN
#define HPGEMM_HOT

#define HPGEMM_ALIGNAS(b)
#define HPGEMM_LIKELY(c)      (c)
#define HPGEMM_UNLIKELY(c)    (c)
#define HPGEMM_PREFETCH(p, rw, loc)
#endif  // __GNUC__

/* Unroll the following loop upto factor of 8. */
#if defined(__clang__)
#define HPGEMM_UNROLL8        _Pragma("clang loop unroll_count(8)")
#elif defined(__GNUC__)
#define HPGEMM_UNROLL8        _Pragma("GCC unroll 4")
#else
#define HPGEMM_UNROLL8
#endif  // unroll 8

/* Maximum register blocking over all the microkernels. */
#define MR_MAX    14
#define NR_MAX    32

/* Default cache blocking parameters, can be tuned at runtime. */
#define MC    112
#define KC    640
#define NC    4608


/* Microkernel signature. Computes `z = beta * z + alpha * x_packed * y_packed`
   for a `MRxNR` tile of `z`. */
typedef void (*hpgemm_ukernel_t)(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
);

/* Microkernel selected at runtime and its register blocking. */
extern hpgemm_ukernel_t hpgemm_ukernel;
extern int hpgemm_mr, hpgemm_nr;
extern const char *hpgemm_kernel_name;

/* Cache blocking parameters. */
extern int hpgemm_mc, hpgemm_kc, hpgemm_nc;


/* Detect the CPU and select the microkernel. Called by `hpgemm()` lazily. */
void kernel_init(void);

/* Platform independent aligned allocation (`align` must be power of 2). */
void *hpgemm_malloc(size_t size, size_t align);
void hpgemm_free(void *ptr);

/* Row-major SGEMM: `z = beta * z + alpha * op(x) * op(y)`, where `z` is
   `m x n`, `op(x)` is `m x k` and `op(y)` is `k x n`. */
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
);

#endif  // HPGEMM_H
