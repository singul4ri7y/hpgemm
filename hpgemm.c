#include <stdbool.h>
#include <immintrin.h>
#include "common.h"


#ifdef __GNUC__
#define HPGEMM_FLATTEN        __attribute__((flatten))
#define HPGEMM_HOT            __attribute__((hot))

#define HPGEMM_ALIGNAS(b)     __attribute__((aligned(b)))
#define HPGEMM_LIKELY(c)      __builtin_expect(!!(c), 1)
#define HPGEMM_UNLIKELY(c)    __builtin_expect(!!(c), 0)
#else
#define HPGEMM_FLATTEN
#define HPGEMM_HOT
#endif  // __GNUC__

#define min(x, y)    ((x) < (y) ? (x) : (y))
/* Get an element from matrices in row-major order. */
#define X(i, j)      x[(i) * ld_x + j]
#define Y(i, j)      y[(i) * ld_y + j]
#define Z(i, j)      z[(i) * ld_z + j]


HPGEMM_HOT void hpgemm_ukernel_6x16(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float *restrict x_packed,
    float *restrict y_packed
) {
    _mm256_zeroall();

    register __m256 c_0;
    register __m256 c_a;

    register __m256 c_1;
    register __m256 c_b;

    register __m256 c_2;
    register __m256 c_c;

    register __m256 c_3;
    register __m256 c_d;

    register __m256 c_4;
    register __m256 c_e;

    register __m256 c_5;
    register __m256 c_f;

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_load_ps(y_packed);
        register __m256 b_l = _mm256_load_ps(y_packed + 8);  // Next row

        register __m256 a = _mm256_set1_ps(x_packed[0]);
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, b_l, c_a);

        a = _mm256_set1_ps(x_packed[1]);
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, b_l, c_b);

        a = _mm256_set1_ps(x_packed[2]);
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, b_l, c_c);

        a = _mm256_set1_ps(x_packed[3]);
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, b_l, c_d);

        a = _mm256_set1_ps(x_packed[4]);
        c_4 = _mm256_fmadd_ps(a, b, c_4);
        c_e = _mm256_fmadd_ps(a, b_l, c_e);

        a = _mm256_set1_ps(x_packed[5]);
        c_5 = _mm256_fmadd_ps(a, b, c_5);
        c_f = _mm256_fmadd_ps(a, b_l, c_f);

        /* 6x16 microkernel. */
        x_packed += 6;
        y_packed += 16;
    }

    /* Store the results. */
    _mm256_store_ps(&gamma(0, 0), c_0);
    _mm256_store_ps(&gamma(0, 8), c_a);

    _mm256_store_ps(&gamma(1, 0), c_1);
    _mm256_store_ps(&gamma(1, 8), c_b);

    _mm256_store_ps(&gamma(2, 0), c_2);
    _mm256_store_ps(&gamma(2, 8), c_c);

    _mm256_store_ps(&gamma(3, 0), c_3);
    _mm256_store_ps(&gamma(3, 8), c_d);

    _mm256_store_ps(&gamma(4, 0), c_4);
    _mm256_store_ps(&gamma(4, 8), c_e);

    _mm256_store_ps(&gamma(5, 0), c_5);
    _mm256_store_ps(&gamma(5, 8), c_f);
}


HPGEMM_HOT static inline void pack(
    int k, int xn, int stride,
    float *restrict x,
    int ld_x,
    float *restrict x_packed,
    bool transpose
) {
    for(int ir = 0; ir < xn; ir += stride) {
        int irb = min(stride, xn - ir);
        for(int p = 0; p < k; p++) {
            int i = 0;
            for(; i < irb; i++) {
                *x_packed++ = transpose ? alpha(ir + i, p) :
                    alpha(p, ir + i);
            }

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
    /* 1st outer loop to slice block MCxKC and `z` into MRxKC micro-panel. */
    for(int i = 0; i < m; i += MR) {
        int ib = min(MR, m - i);

        if(HPGEMM_LIKELY(ib == MR && n == NR))
            /* Dispatch to microkernel. */
            hpgemm_ukernel_6x16(k, beta, &Z(i, 0), ld_z, alpha,
                &x_packed[i * k], y_packed);
        else {
            float dest_shell[MR][NR] HPGEMM_ALIGNAS(64);

            /* Copy original C. */
            if(HPGEMM_UNLIKELY(beta != 0.0f)) {
                for(int ir = 0; ir < ib; ir++) {
                    for(int jr = 0; jr < n; jr++) 
                        dest_shell[ir][jr] = Z(i + ir, jr);
                }
            }

            hpgemm_ukernel_6x16(k, beta, (float *) dest_shell, NR, alpha,
                &x_packed[i * k], y_packed);

            /* Copy shell data back to `z`. */
            for(int ir = 0; ir < ib; ir++) {
                for(int jr = 0; jr < n; jr++) 
                    Z(i + ir, jr) = dest_shell[ir][jr];
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
    /* 2nd outer loop to slice panel KCxNC and `z` into KCxNR micro-panel. */
    for(int j = 0; j < n; j += NR) {
        int jb = min(NR, n - j);
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
    /* 3rd outer loop to slice `x` and `z` by MC over m-dimension. */
    for(int i = 0; i < m; i += MC) {
        int ib = min(MC, m - i);
        /* Pack `MCxKC` block. */
        pack(k, ib, MR, &X(i, 0), ld_x, x_packed, !transp_x);

        /* Dispatch to 2nd outer loop. */
        hpgemm_loop2(ib, n, k, beta, &Z(i, 0), ld_z, beta, x_packed, y_packed);
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
    /* 4th outer loop to slice both `x` and `y` by KC over k-dimension. */
    for(int p = 0; p < k; p += KC) {
        int pb = min(KC, k - p);
        /* Pack the panel KCxNC. */
        pack(pb, n, NR, &Y(p, 0), ld_y, y_packed, transp_y);

        /* Dispatch to loop 3. */
        hpgemm_loop3(transp_x, m, n, pb, beta, z, ld_z, alpha, &X(0, p),
            ld_x, y_packed, x_packed);
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
    int ld_y 
) {
    /* Memory allocation for `MCxKC` block of `x` and `KCxNC` panel of `y`. */
    float *x_packed = _mm_malloc(MC * KC * sizeof(float), 64);
    float *y_packed = _mm_malloc(KC * NC * sizeof(float), 64);

    /* 5th outer loop to slice `z` and `y` by NC over n-dimension. */
    for(int j = 0; j < n; j += NC) {
        int jb = min(NC, n - j);

        /* Dispatch to 4th outer loop. */
        hpgemm_loop4(transp_x, transp_y, m, jb, k, beta, &Z(0, j), ld_z,
            alpha, x, ld_x, &Y(0, j), ld_y, x_packed, y_packed);
    }

    _mm_free(x_packed);
    _mm_free(y_packed);
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
    /* Dispatch to loop 5. */
    hpgemm_loop5(transp_x, transp_y, m, n, k, beta, z, ld_z, alpha, x,
        ld_x, y, ld_y);
}