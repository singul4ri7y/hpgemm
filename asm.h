#ifndef HPGEMM_ASM_H
#define HPGEMM_ASM_H

/* Detect compiler. Clang is mostly compatible with GCC, hence no need to handle
   separately. */
#if defined(__GNUC__) || defined(__clang__)


/* ================ ASSEMBLY CONTROL ================ */

/* Begin the extended inline assembly. */
#define _hpgemm_begin_asm()         __asm__ volatile(

/* End extended inline assembly. */
#define _hpgemm_end_asm(...)        __VA_ARGS__)

/* Use this to pass an input operand. */
#define _hpgemm_inop(var, cons)     [var] #cons (var)

/* Use this to pass an output operand. */
#define _hpgemm_outop(var, cons)    _hpgemm_inop(var, cons)

/* Access a variable/parameter outside of inline assembly region. */
#define _v(var)                     %[var]

/* Convert instructions to string, makes writing assembly much more natural. */
#define _istring(...)               #__VA_ARGS__ ";\n\t"

/* Define a label. */
#define _label(label)               _istring(label:)

/* ================ ASSEMBLY CONTROL END ================ */


#if defined(__x86_64__)

/* NOTE: Even though GCC and Clang both use AT&T syntax by default, the macros
 *   try to follow the Intel Assembly flavor by default as it's more
 *   intuitive. Hence, all the `dest` operand will be to the left. */


/* ================ HELPER MACROS ================ */

/* Helps with generalized macros combining multiple macros. */
#define _get_macro(_1, _2, _3, _4, NAME, ...)    \
NAME

/* Constant. */
#define _c(x)                                $##x

/* ================ HELPER MACROS END ================ */


/* ================ REGISTERS ================ */

/* General purpose registers. */
#define rax      %%rax
#define rbx      %%rbx
#define rcx      %%rcx
#define rdx      %%rdx
#define rsi      %%rsi
#define rdi      %%rdi
#define r8       %%r8
#define r9       %%r9
#define r10      %%r10
#define r11      %%r11
#define r12      %%r12
#define r13      %%r13
#define r14      %%r14
#define r15      %%r15

/* Lower 32-bit of general purpose registers. */
#define eax       %%eax
#define ebx       %%ebx
#define ecx       %%ecx
#define edx       %%edx

/* SSE registers. */
#define xmm0      %%xmm0
#define xmm1      %%xmm1
#define xmm2      %%xmm2
#define xmm3      %%xmm3
#define xmm4      %%xmm4
#define xmm5      %%xmm5
#define xmm6      %%xmm6
#define xmm7      %%xmm7
#define xmm8      %%xmm8
#define xmm9      %%xmm9
#define xmm10     %%xmm10
#define xmm11     %%xmm11
#define xmm12     %%xmm12
#define xmm13     %%xmm13
#define xmm14     %%xmm14
#define xmm15     %%xmm15
#define xmm16    %%xmm16
#define xmm17    %%xmm17
#define xmm18    %%xmm18
#define xmm19    %%xmm19
#define xmm20    %%xmm20
#define xmm21    %%xmm21
#define xmm22    %%xmm22
#define xmm23    %%xmm23
#define xmm24    %%xmm24
#define xmm25    %%xmm25
#define xmm26    %%xmm26
#define xmm27    %%xmm27
#define xmm28    %%xmm28
#define xmm29    %%xmm29
#define xmm30    %%xmm30
#define xmm31    %%xmm31

/* AVX2 Vector registers. */
#define ymm0     %%ymm0
#define ymm1     %%ymm1
#define ymm2     %%ymm2
#define ymm3     %%ymm3
#define ymm4     %%ymm4
#define ymm5     %%ymm5
#define ymm6     %%ymm6
#define ymm7     %%ymm7
#define ymm8     %%ymm8
#define ymm9     %%ymm9
#define ymm10    %%ymm10
#define ymm11    %%ymm11
#define ymm12    %%ymm12
#define ymm13    %%ymm13
#define ymm14    %%ymm14
#define ymm15    %%ymm15
#define ymm16    %%ymm16
#define ymm17    %%ymm17
#define ymm18    %%ymm18
#define ymm19    %%ymm19
#define ymm20    %%ymm20
#define ymm21    %%ymm21
#define ymm22    %%ymm22
#define ymm23    %%ymm23
#define ymm24    %%ymm24
#define ymm25    %%ymm25
#define ymm26    %%ymm26
#define ymm27    %%ymm27
#define ymm28    %%ymm28
#define ymm29    %%ymm29
#define ymm30    %%ymm30
#define ymm31    %%ymm31

/* All 32 AVX-512 Vector registers. */
#define zmm0     %%zmm0
#define zmm1     %%zmm1
#define zmm2     %%zmm2
#define zmm3     %%zmm3
#define zmm4     %%zmm4
#define zmm5     %%zmm5
#define zmm6     %%zmm6
#define zmm7     %%zmm7
#define zmm8     %%zmm8
#define zmm9     %%zmm9
#define zmm10    %%zmm10
#define zmm11    %%zmm11
#define zmm12    %%zmm12
#define zmm13    %%zmm13
#define zmm14    %%zmm14
#define zmm15    %%zmm15
#define zmm16    %%zmm16
#define zmm17    %%zmm17
#define zmm18    %%zmm18
#define zmm19    %%zmm19
#define zmm20    %%zmm20
#define zmm21    %%zmm21
#define zmm22    %%zmm22
#define zmm23    %%zmm23
#define zmm24    %%zmm24
#define zmm25    %%zmm25
#define zmm26    %%zmm26
#define zmm27    %%zmm27
#define zmm28    %%zmm28
#define zmm29    %%zmm29
#define zmm30    %%zmm30
#define zmm31    %%zmm31

/* ================ REGISTERS END ================ */


/* ================ INSTRUCTIONS ================ */

/* Zero out all the vector registers. */
#define _vzeroall()                          _istring(vzeroall)

/* Bitwise operations. */
#define _xor(dest, src)                      _istring(xor src, dest)
#define _shl(dest, n)                        _istring(shl n, dest)
#define _test(dest, src)                     _istring(test src, dest)

/* Move a quad-word (64-bit). */
#define _movq(dest, src)                     _istring(movq src, dest)
/* Move a long-word (32-bit). */
#define _movl(dest, src)                     _istring(movl src, dest)

/* Effective memory manipulation. */
#define _mem1(base)                          (base)
#define _mem2(base, offset)                  offset (base)
#define _mem3(base, index, scale)            (base, index, scale)
#define _mem4(base, index, scale, offset)    offset (base, index, scale)

/* More general purpose memory access macro. */
#define _mem(...)    \
_get_macro(__VA_ARGS__, _mem4, _mem3, _mem2, _mem1)(__VA_ARGS__)

/* Load effective address. */
#define _lea(dest, mem)                      _istring(lea mem, dest)

/* Add and decrement instructions. */
#define _add(dest, src)                      _istring(add src, dest)
#define _dec(src)                            _istring(dec src)

/* Jump instructions. */
#define _jmp(label)                          _istring(jmp label)
#define _je(label)                           _istring(je label)
#define _jz(label)                           _istring(jz label)
#define _jnz(label)                          _istring(jnz label)

/* AVX, AVX2 and AVX-512 instructions. The vector length is decided by the
   registers. */
#define _vmovaps(dest, src)                  _istring(vmovaps src, dest)
#define _vmovups(dest, src)                  _istring(vmovups src, dest)
#define _vbroadcastss(dest, src)             _istring(vbroadcastss src, dest)
#define _vaddps(dest, src1, src2)            _istring(vaddps src1, src2, dest)
#define _vmulps(dest, src1, src2)            _istring(vmulps src1, src2, dest)
#define _vfmadd231ps(dest, src1, src2)       _istring(vfmadd231ps src1, src2, dest)
#define _vmovss(dest, src)                   _istring(vmovss src, dest)
#define _vucomiss(dest, src)                 _istring(vucomiss src, dest)
#define _vxorps(dest, src1, src2)            _istring(vxorps src1, src2, dest)

/* ================ INSTRUCTIONS END ================ */


#elif defined(__aarch64__)

/* NOTE: AArch64 assembly already has `dest` operand to the left. Registers are
 *   not named macros here, rather they are given by their number and the macros
 *   build the name, e.g. `_fmla_lane(0, 24, 28, 1)` is `fmla v0.4s, v24.4s,
 *   v28.s[1]`. `n` is the number of a general purpose register.
 */


/* ================ HELPER MACROS ================ */

/* Constant, AArch64 assembler does not need the `#`. */
#define _c(x)                                x

/* ================ HELPER MACROS END ================ */


/* ================ INSTRUCTIONS ================ */

/* Load/store of 128-bit (q) and 64-bit (d) registers from `[x<n>, off]`. */
#define _ldp_q(a, b, n, off)                 _istring(ldp q##a, q##b, [x##n, off])
#define _stp_q(a, b, n, off)                 _istring(stp q##a, q##b, [x##n, off])
/* Unscaled offset, `off` doesn't need to be multiple of register size. */
#define _ldur_q(a, n, off)                   _istring(ldur q##a, [x##n, off])
#define _ldur_d(a, n, off)                   _istring(ldur d##a, [x##n, off])

/* Load from a memory operand (see `_v()`). */
#define _ldr_x(n, mem)                       _istring(ldr x##n, mem)
#define _ldr_w(n, mem)                       _istring(ldr w##n, mem)
#define _ldrsw(n, mem)                       _istring(ldrsw x##n, mem)
#define _ldr_s(a, mem)                       _istring(ldr s##a, mem)

/* General purpose operations. */
#define _add_imm(d, n, imm)                  _istring(add x##d, x##n, imm)
#define _add_reg(d, n, m)                    _istring(add x##d, x##n, x##m)
#define _lsl(d, n, imm)                      _istring(lsl x##d, x##n, imm)
#define _subs(d, n, imm)                     _istring(subs x##d, x##n, imm)

/* Branch instructions. */
#define _b(label)                            _istring(b label)
#define _beq(label)                          _istring(b.eq label)
#define _bne(label)                          _istring(b.ne label)
#define _cbz(n, label)                       _istring(cbz x##n, label)

/* Vector instructions, 4 x float32 (`.4s`). */
#define _zero(a)                             _istring(eor v##a.16b, v##a.16b, v##a.16b)
#define _fadd(d, a, b)                       _istring(fadd v##d.4s, v##a.4s, v##b.4s)
#define _fmla_lane(d, a, b, lane)            _istring(fmla v##d.4s, v##a.4s, v##b.s[lane])
#define _fmul_lane(d, a, b, lane)            _istring(fmul v##d.4s, v##a.4s, v##b.s[lane])

/* Scalar instructions on float32 (`s`). */
#define _fmov_one(a)                         _istring(fmov s##a, 1.0)
#define _fcmp(a, b)                          _istring(fcmp s##a, s##b)
#define _fcmp_zero(a)                        _istring(fcmp s##a, 0.0)

/* ================ INSTRUCTIONS END ================ */


#endif  // Architecture detection

#elif defined(_MSC_VER)  // MSVC

// TODO: Enable MSVC inline assembly support

#else
#error Compiler inline assembly not supported by hp-gemm.
#endif  // Compiler detection

#endif  // HPGEMM_ASM_H
