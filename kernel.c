#include "hpgemm.h"

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
#include "asm.h"
#endif

/* Get an element from `z` in row-major order. */
#define Z(i, j)    z[(i) * ld_z + (j)]


hpgemm_ukernel_t hpgemm_ukernel = NULL;
int hpgemm_mr, hpgemm_nr;
const char *hpgemm_kernel_name;


/* ================ GENERIC ================ */

/* Portable 6x16 microkernel, only a fallback. */
HPGEMM_HOT static void hpgemm_ukernel_generic(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    float c[6][16] = { { 0.0f } };

    for(int p = 0; p < k; p++) {
        for(int i = 0; i < 6; i++) {
            for(int j = 0; j < 16; j++)
                c[i][j] += x_packed[i] * y_packed[j];
        }

        x_packed += 6;
        y_packed += 16;
    }

    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 16; j++)
            Z(i, j) = alpha * c[i][j] + (beta == 0.0f ? 0.0f : beta * Z(i, j));
    }
}

/* ================ GENERIC END ================ */


#if defined(__GNUC__) && defined(__x86_64__)

/* Everything the assembly touches, the compiler must not rely on. */
#define HPGEMM_X86_CLOBBERS                                                  \
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",      \
    "r12", "r13", "r14", "r15",                                              \
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",          \
    "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",    \
    "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23",  \
    "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",  \
    "cc", "memory"

/* ================ AVX2 ================ */

/* Outer product of a column of `x` and a row of `y`, accumulated into the
   6x16 tile. Offsets are in bytes, `y_0` and `y_1` point to the two halves of
   the row of `y` and `x_i` to the elements of the column of `x`. */
#define HPGEMM_AVX2_OUTER(y_0, y_1, x_0, x_1, x_2, x_3, x_4, x_5) \
    _vmovaps(ymm12, _mem(rbx, y_0))        /* `y` */ \
    _vmovaps(ymm13, _mem(rbx, y_1))        /* Next row */ \
 \
    _vbroadcastss(ymm14, _mem(rax, x_0))   /* `x + 0` */ \
    _vfmadd231ps(ymm0, ymm12, ymm14) \
    _vfmadd231ps(ymm1, ymm13, ymm14) \
 \
    _vbroadcastss(ymm14, _mem(rax, x_1))   /* `x + 1` */ \
    _vfmadd231ps(ymm2, ymm12, ymm14) \
    _vfmadd231ps(ymm3, ymm13, ymm14) \
 \
    _vbroadcastss(ymm14, _mem(rax, x_2))   /* `x + 2` */ \
    _vfmadd231ps(ymm4, ymm12, ymm14) \
    _vfmadd231ps(ymm5, ymm13, ymm14) \
 \
    _vbroadcastss(ymm14, _mem(rax, x_3))   /* `x + 3` */ \
    _vfmadd231ps(ymm6, ymm12, ymm14) \
    _vfmadd231ps(ymm7, ymm13, ymm14) \
 \
    _vbroadcastss(ymm14, _mem(rax, x_4))   /* `x + 4` */ \
    _vfmadd231ps(ymm8, ymm12, ymm14) \
    _vfmadd231ps(ymm9, ymm13, ymm14) \
 \
    _vbroadcastss(ymm14, _mem(rax, x_5))   /* `x + 5` */ \
    _vfmadd231ps(ymm10, ymm12, ymm14) \
    _vfmadd231ps(ymm11, ymm13, ymm14)

/* Apply `OP` on every row of the tile, `m_0` and `m_1` are the registers
   holding the address of the row in `z`. */
#define HPGEMM_AVX2_ROWS(OP) \
    OP(ymm0, ymm1, rax, rbx) \
    OP(ymm2, ymm3, rdx, rdi) \
    OP(ymm4, ymm5, r8, r9) \
    OP(ymm6, ymm7, r10, r11) \
    OP(ymm8, ymm9, r12, r13) \
    OP(ymm10, ymm11, r14, r15)

#define HPGEMM_AVX2_SCALE(c_0, c_1, m_0, m_1) \
    _vmulps(c_0, c_0, ymm12) \
    _vmulps(c_1, c_1, ymm12)

#define HPGEMM_AVX2_STORE(c_0, c_1, m_0, m_1) \
    _vmovups(_mem(m_0), c_0) \
    _vmovups(_mem(m_1), c_1)

#define HPGEMM_AVX2_ACC(c_0, c_1, m_0, m_1) \
    _vmovups(ymm12, _mem(m_0)) \
    _vaddps(c_0, c_0, ymm12) \
    _vmovups(ymm12, _mem(m_1)) \
    _vaddps(c_1, c_1, ymm12)

#define HPGEMM_AVX2_AXPBY(c_0, c_1, m_0, m_1) \
    _vmovups(ymm13, _mem(m_0)) \
    _vfmadd231ps(c_0, ymm12, ymm13) \
    _vmovups(ymm13, _mem(m_1)) \
    _vfmadd231ps(c_1, ymm12, ymm13)

/* 6x16 microkernel for Haswell and above. */
HPGEMM_HOT static void hpgemm_ukernel_avx2_6x16(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    const float one = 1.0f;

    /* Unrolled by 8, the rest is handled separately. */
    int k_iter = k / 8;
    int k_left = k % 8;

    _hpgemm_begin_asm()

    /* Zero out the accumulators. */
    _vzeroall()

    _movq(rax, _v(x_packed))    // `x_packed`
    _movq(rbx, _v(y_packed))    // `y_packed`

    _xor(rcx, rcx)              // Zero out
    _movl(ecx, _v(k_iter))      // `k_iter`
    _test(rcx, rcx)
    _jz(.hpgemm_avx2_6x16_kleft)

    _label(.hpgemm_avx2_6x16_kiter)
        HPGEMM_AVX2_OUTER(0, 32, 0, 4, 8, 12, 16, 20)
        HPGEMM_AVX2_OUTER(64, 96, 24, 28, 32, 36, 40, 44)
        HPGEMM_AVX2_OUTER(128, 160, 48, 52, 56, 60, 64, 68)
        HPGEMM_AVX2_OUTER(192, 224, 72, 76, 80, 84, 88, 92)
        HPGEMM_AVX2_OUTER(256, 288, 96, 100, 104, 108, 112, 116)
        HPGEMM_AVX2_OUTER(320, 352, 120, 124, 128, 132, 136, 140)
        HPGEMM_AVX2_OUTER(384, 416, 144, 148, 152, 156, 160, 164)
        HPGEMM_AVX2_OUTER(448, 480, 168, 172, 176, 180, 184, 188)

        _add(rax, _c(192))  // `x_packed += 8 * 6`, MR = 6
        _add(rbx, _c(512))  // `y_packed += 8 * 16`, NR = 16

        _dec(rcx)
        _jnz(.hpgemm_avx2_6x16_kiter)

    _label(.hpgemm_avx2_6x16_kleft)
    _xor(rcx, rcx)
    _movl(ecx, _v(k_left))      // `k_left`
    _test(rcx, rcx)
    _jz(.hpgemm_avx2_6x16_alpha)

    _label(.hpgemm_avx2_6x16_kleft_iter)
        HPGEMM_AVX2_OUTER(0, 32, 0, 4, 8, 12, 16, 20)

        /* 6x16 microkernel. */
        _add(rax, _c(24))  // `x_packed += 6`, MR = 6
        _add(rbx, _c(64))  // `y_packed += 16`, NR = 16

        _dec(rcx)
        _jnz(.hpgemm_avx2_6x16_kleft_iter)

    _label(.hpgemm_avx2_6x16_alpha)
    /* Store `one` to xmm15. */
    _vmovss(xmm15, _v(one))

    /* Apply `alpha`. */
    _vmovss(xmm12, _v(alpha))
    _vucomiss(xmm12, xmm15)  // Compare `alpha` against `one`
    /* No need to apply `alpha` if equal. */
    _je(.hpgemm_avx2_6x16_lea)

    _vbroadcastss(ymm12, xmm12)
    HPGEMM_AVX2_ROWS(HPGEMM_AVX2_SCALE)

    _label(.hpgemm_avx2_6x16_lea)
    /* Calculate addresses of `z` beforehand. */
    _xor(rcx, rcx)
    _movl(ecx, _v(ld_z))            // `ld_z`
    _shl(rcx, _c(2))                // `ld_z * 4`, float32 is 4-bytes

    _movq(rax, _v(z))               // `z`
    _lea(rbx, _mem(rax, 0x20))      // `z + 8`
    _lea(rdx, _mem(rax, rcx, 1))    // `z + 1 * ld_z`
    _lea(rdi, _mem(rdx, 0x20))      // `z + 1 * ld_z + 8`
    _lea(r8, _mem(rax, rcx, 2))     // `z + 2 * ld_z`
    _lea(r9, _mem(r8, 0x20))        // `z + 2 * ld_z + 8`
    _lea(r10, _mem(r8, rcx, 1))     // `z + 3 * ld_z`
    _lea(r11, _mem(r10, 0x20))      // `z + 3 * ld_z + 8`
    _lea(r12, _mem(rax, rcx, 4))    // `z + 4 * ld_z`
    _lea(r13, _mem(r12, 0x20))      // `z + 4 * ld_z + 8`
    _lea(r14, _mem(r12, rcx, 1))    // `z + 5 * ld_z`
    _lea(r15, _mem(r14, 0x20))      // `z + 5 * ld_z + 8`

    /* Apply `beta`. */
    _vmovss(xmm12, _v(beta))
    _vxorps(xmm13, xmm13, xmm13)

    /* If `beta` is 0.0, no need to load `z`. */
    _vucomiss(xmm12, xmm13)
    _je(.hpgemm_avx2_6x16_done)

    /* If `beta` is 1.0, load and add `z`, no need to multiply `beta`. */
    _vucomiss(xmm12, xmm15)
    _je(.hpgemm_avx2_6x16_accm)

    _vbroadcastss(ymm12, xmm12)
    HPGEMM_AVX2_ROWS(HPGEMM_AVX2_AXPBY)

    /* Done applying `beta`. */
    _jmp(.hpgemm_avx2_6x16_done)

    _label(.hpgemm_avx2_6x16_accm)
    HPGEMM_AVX2_ROWS(HPGEMM_AVX2_ACC)

    _label(.hpgemm_avx2_6x16_done)
    /* Store the results. */
    HPGEMM_AVX2_ROWS(HPGEMM_AVX2_STORE)

    _hpgemm_end_asm(
        :  // No output
        : _hpgemm_inop(k_iter, m),
          _hpgemm_inop(k_left, m),
          _hpgemm_inop(beta, m),
          _hpgemm_inop(z, m),
          _hpgemm_inop(ld_z, m),
          _hpgemm_inop(alpha, m),
          _hpgemm_inop(x_packed, m),
          _hpgemm_inop(y_packed, m),
          _hpgemm_inop(one, m)
        : HPGEMM_X86_CLOBBERS
    );
}

/* ================ AVX2 END ================ */


/* ================ AVX-512 ================ */

#if defined(__AVX512__)

/* Outer product of a column of `x` and a row of `y`, accumulated into the
   14x32 tile. Offsets are in bytes, `y_0` and `y_1` point to the two halves of
   the row of `y` and `x_i` to the elements of the column of `x`. */
#define HPGEMM_AVX512_14x32_OUTER(y_0, y_1, \
    x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, \
    x_8, x_9, x_10, x_11, x_12, x_13) \
    _vmovaps(zmm28, _mem(rbx, y_0)) \
    _vmovaps(zmm29, _mem(rbx, y_1)) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_0)) \
    _vfmadd231ps(zmm0, zmm28, zmm30) \
    _vfmadd231ps(zmm1, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_1)) \
    _vfmadd231ps(zmm2, zmm28, zmm30) \
    _vfmadd231ps(zmm3, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_2)) \
    _vfmadd231ps(zmm4, zmm28, zmm30) \
    _vfmadd231ps(zmm5, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_3)) \
    _vfmadd231ps(zmm6, zmm28, zmm30) \
    _vfmadd231ps(zmm7, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_4)) \
    _vfmadd231ps(zmm8, zmm28, zmm30) \
    _vfmadd231ps(zmm9, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_5)) \
    _vfmadd231ps(zmm10, zmm28, zmm30) \
    _vfmadd231ps(zmm11, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_6)) \
    _vfmadd231ps(zmm12, zmm28, zmm30) \
    _vfmadd231ps(zmm13, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_7)) \
    _vfmadd231ps(zmm14, zmm28, zmm30) \
    _vfmadd231ps(zmm15, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_8)) \
    _vfmadd231ps(zmm16, zmm28, zmm30) \
    _vfmadd231ps(zmm17, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_9)) \
    _vfmadd231ps(zmm18, zmm28, zmm30) \
    _vfmadd231ps(zmm19, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_10)) \
    _vfmadd231ps(zmm20, zmm28, zmm30) \
    _vfmadd231ps(zmm21, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_11)) \
    _vfmadd231ps(zmm22, zmm28, zmm30) \
    _vfmadd231ps(zmm23, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_12)) \
    _vfmadd231ps(zmm24, zmm28, zmm30) \
    _vfmadd231ps(zmm25, zmm29, zmm30) \
 \
    _vbroadcastss(zmm30, _mem(rax, x_13)) \
    _vfmadd231ps(zmm26, zmm28, zmm30) \
    _vfmadd231ps(zmm27, zmm29, zmm30)


/* Apply `OP` on every row of the tile, `m` is the register holding the
   address of the row in `z`. */
#define HPGEMM_AVX512_14x32_ROWS(OP) \
    OP(zmm0, zmm1, rax) \
    OP(zmm2, zmm3, rbx) \
    OP(zmm4, zmm5, rdx) \
    OP(zmm6, zmm7, rdi) \
    OP(zmm8, zmm9, r8) \
    OP(zmm10, zmm11, r9) \
    OP(zmm12, zmm13, r10) \
    OP(zmm14, zmm15, r11) \
    OP(zmm16, zmm17, r12) \
    OP(zmm18, zmm19, r13) \
    OP(zmm20, zmm21, r14) \
    OP(zmm22, zmm23, r15) \
    OP(zmm24, zmm25, rsi) \
    OP(zmm26, zmm27, rcx)

#define HPGEMM_AVX512_14x32_SCALE(c_0, c_1, m) \
    _vmulps(c_0, c_0, zmm28) \
    _vmulps(c_1, c_1, zmm28)

#define HPGEMM_AVX512_14x32_STORE(c_0, c_1, m) \
    _vmovups(_mem(m), c_0) \
    _vmovups(_mem(m, 0x40), c_1)

#define HPGEMM_AVX512_14x32_ACC(c_0, c_1, m) \
    _vmovups(zmm29, _mem(m)) \
    _vaddps(c_0, c_0, zmm29) \
    _vmovups(zmm29, _mem(m, 0x40)) \
    _vaddps(c_1, c_1, zmm29)

#define HPGEMM_AVX512_14x32_AXPBY(c_0, c_1, m) \
    _vmovups(zmm29, _mem(m)) \
    _vfmadd231ps(c_0, zmm28, zmm29) \
    _vmovups(zmm29, _mem(m, 0x40)) \
    _vfmadd231ps(c_1, zmm28, zmm29)


/* 14x32 microkernel for AVX-512. 28 accumulators + 2 `y` + 1 `x`
   uses 31 out of 32 vector registers. */
HPGEMM_HOT static void hpgemm_ukernel_avx512_14x32(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    const float one = 1.0f;

    /* Unrolled by 8, the rest is handled separately. */
    int k_iter = k / 8;
    int k_left = k % 8;

    _hpgemm_begin_asm()

    /* Zero out the accumulators. vzeroall only clears upto zmm15. Rest
       we have to clear ourselves. */
    _vzeroall()
    _vxorps(zmm16, zmm16, zmm16)
    _vxorps(zmm17, zmm17, zmm17)
    _vxorps(zmm18, zmm18, zmm18)
    _vxorps(zmm19, zmm19, zmm19)
    _vxorps(zmm20, zmm20, zmm20)
    _vxorps(zmm21, zmm21, zmm21)
    _vxorps(zmm22, zmm22, zmm22)
    _vxorps(zmm23, zmm23, zmm23)
    _vxorps(zmm24, zmm24, zmm24)
    _vxorps(zmm25, zmm25, zmm25)
    _vxorps(zmm26, zmm26, zmm26)
    _vxorps(zmm27, zmm27, zmm27)

    _movq(rax, _v(x_packed))    // `x_packed`
    _movq(rbx, _v(y_packed))    // `y_packed`

    _xor(rcx, rcx)              // Zero out
    _movl(ecx, _v(k_iter))      // `k_iter`
    _test(rcx, rcx)
    _jz(.hpgemm_avx512_14x32_kleft)

    _label(.hpgemm_avx512_14x32_kiter)
        HPGEMM_AVX512_14x32_OUTER(0, 64,
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52)
        HPGEMM_AVX512_14x32_OUTER(128, 192,
            56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108)
        HPGEMM_AVX512_14x32_OUTER(256, 320,
            112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164)
        HPGEMM_AVX512_14x32_OUTER(384, 448,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220)
        HPGEMM_AVX512_14x32_OUTER(512, 576,
            224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276)
        HPGEMM_AVX512_14x32_OUTER(640, 704,
            280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332)
        HPGEMM_AVX512_14x32_OUTER(768, 832,
            336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388)
        HPGEMM_AVX512_14x32_OUTER(896, 960,
            392, 396, 400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444)

        _add(rax, _c(448))  // `x_packed += 8 * 14`, MR = 14
        _add(rbx, _c(1024)) // `y_packed += 8 * 32`, NR = 32

        _dec(rcx)
        _jnz(.hpgemm_avx512_14x32_kiter)

    _label(.hpgemm_avx512_14x32_kleft)
    _xor(rcx, rcx)
    _movl(ecx, _v(k_left))      // `k_left`
    _test(rcx, rcx)
    _jz(.hpgemm_avx512_14x32_alpha)

    _label(.hpgemm_avx512_14x32_kleft_iter)
        HPGEMM_AVX512_14x32_OUTER(0, 64,
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52)

        /* 14x32 microkernel. */
        _add(rax, _c(56))  // `x_packed += 14`, MR = 14
        _add(rbx, _c(128)) // `y_packed += 32`, NR = 32

        _dec(rcx)
        _jnz(.hpgemm_avx512_14x32_kleft_iter)

    _label(.hpgemm_avx512_14x32_alpha)

    /* Store `one` to xmm31. */
    _vmovss(xmm31, _v(one))

    /* Apply `alpha`. */
    _vmovss(xmm28, _v(alpha))
    _vucomiss(xmm28, xmm31)  // Compare `alpha` against `one`
    /* No need to apply `alpha` if equal. */
    _je(.hpgemm_avx512_14x32_lea)

    _vbroadcastss(zmm28, xmm28)
    HPGEMM_AVX512_14x32_ROWS(HPGEMM_AVX512_14x32_SCALE)

    _label(.hpgemm_avx512_14x32_lea)
    /* Calculate addresses of `z` beforehand. */
    _xor(rcx, rcx)
    _movl(ecx, _v(ld_z))            // `ld_z`
    _shl(rcx, _c(2))                // `ld_z * 4`, float32 is 4-bytes

    _movq(rax, _v(z))               // `z`
    _lea(rbx, _mem(rax, rcx, 1))    // `z + 1 * ld_z`
    _lea(rdx, _mem(rax, rcx, 2))    // `z + 2 * ld_z`
    _lea(rdi, _mem(rdx, rcx, 1))    // `z + 3 * ld_z`
    _lea(r8, _mem(rax, rcx, 4))     // `z + 4 * ld_z`
    _lea(r9, _mem(r8, rcx, 1))      // `z + 5 * ld_z`
    _lea(r10, _mem(r8, rcx, 2))     // `z + 6 * ld_z`
    _lea(r11, _mem(r10, rcx, 1))   // `z + 7 * ld_z`
    _lea(r12, _mem(rax, rcx, 8))   // `z + 8 * ld_z`
    _lea(r13, _mem(r12, rcx, 1))   // `z + 9 * ld_z`
    _lea(r14, _mem(r12, rcx, 2))   // `z + 10 * ld_z`
    _lea(r15, _mem(r14, rcx, 1))   // `z + 11 * ld_z`
    _lea(rsi, _mem(r12, rcx, 4))   // `z + 12 * ld_z`
    _lea(rcx, _mem(rsi, rcx, 1))   // `z + 13 * ld_z`

    /* Apply `beta`. */
    _vmovss(xmm28, _v(beta))
    _vxorps(xmm29, xmm29, xmm29)

    /* If `beta` is 0.0, no need to load `z`. */
    _vucomiss(xmm28, xmm29)
    _je(.hpgemm_avx512_14x32_done)

    /* If `beta` is 1.0, load and add `z`, no need to multiply `beta`. */
    _vucomiss(xmm28, xmm31)
    _je(.hpgemm_avx512_14x32_accm)

    _vbroadcastss(zmm28, xmm28)
    HPGEMM_AVX512_14x32_ROWS(HPGEMM_AVX512_14x32_AXPBY)

    /* Done applying `beta`. */
    _jmp(.hpgemm_avx512_14x32_done)

    _label(.hpgemm_avx512_14x32_accm)
    HPGEMM_AVX512_14x32_ROWS(HPGEMM_AVX512_14x32_ACC)

    _label(.hpgemm_avx512_14x32_done)
    /* Store the results. */
    HPGEMM_AVX512_14x32_ROWS(HPGEMM_AVX512_14x32_STORE)

    _hpgemm_end_asm(
        :  // No output
        : _hpgemm_inop(k_iter, m),
          _hpgemm_inop(k_left, m),
          _hpgemm_inop(beta, m),
          _hpgemm_inop(z, m),
          _hpgemm_inop(ld_z, m),
          _hpgemm_inop(alpha, m),
          _hpgemm_inop(x_packed, m),
          _hpgemm_inop(y_packed, m),
          _hpgemm_inop(one, m)
        : HPGEMM_X86_CLOBBERS
    );
}

#endif  // AVX512

/* ================ AVX-512 END ================ */

#endif  // __x86_64__


#if defined(__GNUC__) && defined(__aarch64__)

#define HPGEMM_NEON_CLOBBERS                                                 \
    "x0", "x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "v0",       \
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",      \
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",    \
    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",    \
    "cc", "memory"


/* ================ NEON ================ */

/* Outer product of a column of `x` and a row of `y`, accumulated into the
   6x16 tile. `y_i` are offsets (bytes) of `y_packed` (rows of `y` are loaded
   to v24-v27), `x_0` and `x_1` are of `x_packed` (column of `x` is loaded to
   v28 and v29). */
/* A row of the tile is 4 vectors, multiplied by lane `lane` of `a`. */
#define HPGEMM_NEON_FMA(c_0, c_1, c_2, c_3, a, lane) \
    _fmla_lane(c_0, 24, a, lane) \
    _fmla_lane(c_1, 25, a, lane) \
    _fmla_lane(c_2, 26, a, lane) \
    _fmla_lane(c_3, 27, a, lane)

#define HPGEMM_NEON_OUTER(y_0, y_1, x_0, x_1) \
    _ldp_q(24, 25, 1, y_0)        /* `y` */ \
    _ldp_q(26, 27, 1, y_1)        /* Next row */ \
 \
    _ldur_q(28, 0, x_0)           /* `x` to `x + 3` */ \
    _ldur_d(29, 0, x_1)           /* `x + 4` and `x + 5` */ \
 \
    HPGEMM_NEON_FMA(0, 1, 2, 3, 28, 0) \
    HPGEMM_NEON_FMA(4, 5, 6, 7, 28, 1) \
    HPGEMM_NEON_FMA(8, 9, 10, 11, 28, 2) \
    HPGEMM_NEON_FMA(12, 13, 14, 15, 28, 3) \
    HPGEMM_NEON_FMA(16, 17, 18, 19, 29, 0) \
    HPGEMM_NEON_FMA(20, 21, 22, 23, 29, 1)

/* Apply `OP` on every row of the tile, `n` is the register holding the
   address of the row in `z`. */
#define HPGEMM_NEON_ROWS(OP) \
    OP(0, 1, 2, 3, 5) \
    OP(4, 5, 6, 7, 6) \
    OP(8, 9, 10, 11, 7) \
    OP(12, 13, 14, 15, 8) \
    OP(16, 17, 18, 19, 9) \
    OP(20, 21, 22, 23, 10)

#define HPGEMM_NEON_SCALE(c_0, c_1, c_2, c_3, n) \
    _fmul_lane(c_0, c_0, 30, 0) \
    _fmul_lane(c_1, c_1, 30, 0) \
    _fmul_lane(c_2, c_2, 30, 0) \
    _fmul_lane(c_3, c_3, 30, 0)

#define HPGEMM_NEON_STORE(c_0, c_1, c_2, c_3, n) \
    _stp_q(c_0, c_1, n, 0) \
    _stp_q(c_2, c_3, n, 32)

#define HPGEMM_NEON_ACC(c_0, c_1, c_2, c_3, n) \
    _ldp_q(24, 25, n, 0) \
    _ldp_q(26, 27, n, 32) \
    _fadd(c_0, c_0, 24) \
    _fadd(c_1, c_1, 25) \
    _fadd(c_2, c_2, 26) \
    _fadd(c_3, c_3, 27)

#define HPGEMM_NEON_AXPBY(c_0, c_1, c_2, c_3, n) \
    _ldp_q(24, 25, n, 0) \
    _ldp_q(26, 27, n, 32) \
    _fmla_lane(c_0, 24, 30, 0) \
    _fmla_lane(c_1, 25, 30, 0) \
    _fmla_lane(c_2, 26, 30, 0) \
    _fmla_lane(c_3, 27, 30, 0)

/* 6x16 microkernel for AArch64 NEON. 24 accumulators (v0-v23) + 4 `y` (v24-v27)
   + 2 `x` (v28-v29) uses 30 out of 32 vector registers. */
HPGEMM_HOT static void hpgemm_ukernel_neon_6x16(
    int k,
    float beta,
    float *restrict z,
    int ld_z,
    float alpha,
    float *restrict x_packed,
    float *restrict y_packed
) {
    /* Unrolled by 8, the rest is handled separately. */
    int k_iter = k / 8;
    int k_left = k % 8;

    _hpgemm_begin_asm()

    /* Zero out the accumulators. */
    _zero(0) _zero(1) _zero(2) _zero(3)
    _zero(4) _zero(5) _zero(6) _zero(7)
    _zero(8) _zero(9) _zero(10) _zero(11)
    _zero(12) _zero(13) _zero(14) _zero(15)
    _zero(16) _zero(17) _zero(18) _zero(19)
    _zero(20) _zero(21) _zero(22) _zero(23)

    _ldr_x(0, _v(x_packed))     // `x_packed`
    _ldr_x(1, _v(y_packed))     // `y_packed`

    _ldr_w(2, _v(k_iter))       // `k_iter`
    _cbz(2, .hpgemm_neon_6x16_kleft)

    _label(.hpgemm_neon_6x16_kiter)
        HPGEMM_NEON_OUTER(0, 32, 0, 16)
        HPGEMM_NEON_OUTER(64, 96, 24, 40)
        HPGEMM_NEON_OUTER(128, 160, 48, 64)
        HPGEMM_NEON_OUTER(192, 224, 72, 88)
        HPGEMM_NEON_OUTER(256, 288, 96, 112)
        HPGEMM_NEON_OUTER(320, 352, 120, 136)
        HPGEMM_NEON_OUTER(384, 416, 144, 160)
        HPGEMM_NEON_OUTER(448, 480, 168, 184)

        _add_imm(0, 0, _c(192))  // `x_packed += 8 * 6`, MR = 6
        _add_imm(1, 1, _c(512))  // `y_packed += 8 * 16`, NR = 16

        _subs(2, 2, _c(1))
        _bne(.hpgemm_neon_6x16_kiter)

    _label(.hpgemm_neon_6x16_kleft)
    _ldr_w(2, _v(k_left))       // `k_left`
    _cbz(2, .hpgemm_neon_6x16_alpha)

    _label(.hpgemm_neon_6x16_kleft_iter)
        HPGEMM_NEON_OUTER(0, 32, 0, 16)

        /* 6x16 microkernel. */
        _add_imm(0, 0, _c(24))  // `x_packed += 6`, MR = 6
        _add_imm(1, 1, _c(64))  // `y_packed += 16`, NR = 16

        _subs(2, 2, _c(1))
        _bne(.hpgemm_neon_6x16_kleft_iter)

    _label(.hpgemm_neon_6x16_alpha)
    /* Store `one` to s31. */
    _fmov_one(31)

    /* Apply `alpha`. */
    _ldr_s(30, _v(alpha))
    _fcmp(30, 31)  // Compare `alpha` against `one`
    /* No need to apply `alpha` if equal. */
    _beq(.hpgemm_neon_6x16_lea)

    HPGEMM_NEON_ROWS(HPGEMM_NEON_SCALE)

    _label(.hpgemm_neon_6x16_lea)
    /* Calculate addresses of `z` beforehand. */
    _ldrsw(4, _v(ld_z))             // `ld_z`
    _lsl(4, 4, _c(2))               // `ld_z * 4`, float32 is 4-bytes

    _ldr_x(5, _v(z))                // `z`
    _add_reg(6, 5, 4)               // `z + 1 * ld_z`
    _add_reg(7, 6, 4)               // `z + 2 * ld_z`
    _add_reg(8, 7, 4)               // `z + 3 * ld_z`
    _add_reg(9, 8, 4)               // `z + 4 * ld_z`
    _add_reg(10, 9, 4)              // `z + 5 * ld_z`

    /* Apply `beta`. */
    _ldr_s(30, _v(beta))

    /* If `beta` is 0.0, no need to load `z`. */
    _fcmp_zero(30)
    _beq(.hpgemm_neon_6x16_done)

    /* If `beta` is 1.0, load and add `z`, no need to multiply `beta`. */
    _fcmp(30, 31)
    _beq(.hpgemm_neon_6x16_accm)

    HPGEMM_NEON_ROWS(HPGEMM_NEON_AXPBY)

    /* Done applying `beta`. */
    _b(.hpgemm_neon_6x16_done)

    _label(.hpgemm_neon_6x16_accm)
    HPGEMM_NEON_ROWS(HPGEMM_NEON_ACC)

    _label(.hpgemm_neon_6x16_done)
    /* Store the results. */
    HPGEMM_NEON_ROWS(HPGEMM_NEON_STORE)

    _hpgemm_end_asm(
        :  // No output
        : _hpgemm_inop(k_iter, m),
          _hpgemm_inop(k_left, m),
          _hpgemm_inop(beta, m),
          _hpgemm_inop(z, m),
          _hpgemm_inop(ld_z, m),
          _hpgemm_inop(alpha, m),
          _hpgemm_inop(x_packed, m),
          _hpgemm_inop(y_packed, m)
        : HPGEMM_NEON_CLOBBERS
    );
}

/* ================ NEON END ================ */

#endif  // __aarch64__


void kernel_init(void) {
    /* Fallback. */
    hpgemm_ukernel = hpgemm_ukernel_generic;
    hpgemm_mr = 6;
    hpgemm_nr = 16;
    hpgemm_kernel_name = "generic";

#if defined(__GNUC__) && defined(__x86_64__)
    /* Ask the CPU itself. The kernels are assembly, hence they do not depend on
       the compiler flags. */
    __builtin_cpu_init();

#if defined(__AVX512__)
    if(__builtin_cpu_supports("avx512f")) {
        hpgemm_ukernel = hpgemm_ukernel_avx512_14x32;
        hpgemm_mr = 14;
        hpgemm_nr = 32;
        hpgemm_kernel_name = "avx512";
    } else
#endif  // AVX512

    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        hpgemm_ukernel = hpgemm_ukernel_avx2_6x16;
        hpgemm_kernel_name = "avx2";
    }
#elif defined(__GNUC__) && defined(__aarch64__)
    /* NEON is mandatory on AArch64. */
    hpgemm_ukernel = hpgemm_ukernel_neon_6x16;
    hpgemm_kernel_name = "neon";
#endif
}
