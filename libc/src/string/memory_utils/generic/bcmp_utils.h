//===-- Shared bcmp building blocks -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/endian_internal.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_INLINE_VAR
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_LIKELY, LIBC_UNLIKELY, LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/os.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT64
#include "src/__support/macros/sanitizer.h" // LIBC_HAS_MEMORY_ACCESS_SANITIZER
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType, load_unaligned

#include <stddef.h> // size_t

// Whether short comparisons (1..8 bytes) may read 8 bytes forward within a
// 4 KiB page and discard trailing bytes (little-endian only). Disabled under
// sanitizers, Arm MTE (16-byte granules), and baremetal/unknown OSes where
// MPU/PMP regions may be smaller than 4 KiB.
#if !defined(LIBC_HAS_MEMORY_ACCESS_SANITIZER) &&                              \
    !defined(__ARM_FEATURE_MEMORY_TAGGING) &&                                  \
    !LIBC_HAS_FEATURE(memtag_stack) &&                                         \
    !defined(LIBC_TARGET_OS_IS_BAREMETAL) &&                                   \
    (defined(LIBC_TARGET_OS_IS_LINUX) || defined(LIBC_TARGET_OS_IS_ANDROID) || \
     defined(LIBC_TARGET_OS_IS_FREEBSD) ||                                     \
     defined(LIBC_TARGET_OS_IS_WINDOWS) ||                                     \
     defined(LIBC_TARGET_OS_IS_FUCHSIA) || defined(__APPLE__)) &&              \
    defined(LIBC_TYPES_HAS_INT64) && defined(__BYTE_ORDER__) &&                \
    (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) &&                             \
    (defined(LIBC_TARGET_ARCH_IS_X86_64) ||                                    \
     (defined(LIBC_TARGET_ARCH_IS_AARCH64) &&                                  \
      (defined(__ARM_FEATURE_UNALIGNED) || defined(_M_ARM64))) ||              \
     (defined(LIBC_TARGET_ARCH_IS_RISCV64) &&                                  \
      defined(__riscv_misaligned_fast)))
#define LIBC_BCMP_USE_OVERREAD 1
#else
#define LIBC_BCMP_USE_OVERREAD 0
#endif

#if LIBC_BCMP_USE_OVERREAD && defined(__BMI2__) &&                             \
    defined(LIBC_TARGET_ARCH_IS_X86_64)
#include <immintrin.h>
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

template <typename T> LIBC_INLINE T bcmp_xor(CPtr p1, CPtr p2) {
  return load_unaligned<T>(p1) ^ load_unaligned<T>(p2);
}

// Compares count bytes for count in 1..3 without reading outside [0, count).
LIBC_INLINE BcmpReturnType bcmp_exact_1_to_3(CPtr p1, CPtr p2, size_t count) {
  return cpp::to_integer<uint32_t>((p1[0] ^ p2[0]) |
                                   (p1[count - 1] ^ p2[count - 1]) |
                                   (p1[count >> 1] ^ p2[count >> 1]));
}

#if LIBC_BCMP_USE_OVERREAD

// Rejects page offsets 4088..4095 so an 8-byte forward read stays within its
// 4 KiB page (never rejects offset 0).
LIBC_INLINE bool read8_stays_in_page(CPtr ptr) {
  return ((reinterpret_cast<uintptr_t>(ptr) + 8) & 0xff8U) != 0;
}

// Single-cache-line little-endian masks for 1..8 bytes.
#if defined(LIBC_TARGET_ARCH_IS_X86_64) && !defined(__BMI2__)
// Indexed by `count - 1`: x86-64 folds `-8` into `testq -8(base, count, 8)`.
alignas(64) LIBC_INLINE_VAR constexpr uint64_t SIZE_MASKS_8[8] = {
    0x00000000000000ffULL, 0x000000000000ffffULL, 0x0000000000ffffffULL,
    0x00000000ffffffffULL, 0x000000ffffffffffULL, 0x0000ffffffffffffULL,
    0x00ffffffffffffffULL, 0xffffffffffffffffULL,
};
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
// Indexed by `count & 7` (index 0 holds the 8-byte mask since `count == 0`
// exits earlier): lets AArch64 compute the index in parallel with address
// materialization and load via `[base, idx, lsl #3]`.
alignas(64) LIBC_INLINE_VAR constexpr uint64_t SIZE_MASKS_8[8] = {
    0xffffffffffffffffULL, 0x00000000000000ffULL, 0x000000000000ffffULL,
    0x0000000000ffffffULL, 0x00000000ffffffffULL, 0x000000ffffffffffULL,
    0x0000ffffffffffffULL, 0x00ffffffffffffffULL,
};
#endif

// Compares count in 1..8 without size branches by XORing 8-byte forward loads
// and discarding bytes past count.
LIBC_INLINE BcmpReturnType bcmp_overread_1_to_8(CPtr p1, CPtr p2,
                                                size_t count) {
  const uint64_t ne_bits = bcmp_xor<uint64_t>(p1, p2);
#if defined(__BMI2__) && defined(LIBC_TARGET_ARCH_IS_X86_64)
  return static_cast<uint32_t>(
      _bzhi_u64(ne_bits, static_cast<unsigned int>(8 * count)) != 0);
#elif defined(LIBC_TARGET_ARCH_IS_X86_64)
  return static_cast<uint32_t>((ne_bits & SIZE_MASKS_8[count - 1]) != 0);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return static_cast<uint32_t>((ne_bits & SIZE_MASKS_8[count & 7]) != 0);
#else
  // On RISC-V (no scaled-register load), shifting avoids table materialization.
  return static_cast<uint32_t>((ne_bits << ((8 - count) * 8)) != 0);
#endif
}

#endif // LIBC_BCMP_USE_OVERREAD

// Compares count bytes for count in 0..8.
LIBC_INLINE BcmpReturnType bcmp_0_to_8(CPtr p1, CPtr p2, size_t count) {
  if (count == 0)
    return BcmpReturnType::zero();
#if LIBC_BCMP_USE_OVERREAD
  if (LIBC_LIKELY(read8_stays_in_page(p1) && read8_stays_in_page(p2)))
    return bcmp_overread_1_to_8(p1, p2, count);
#endif
  if (count >= 4) {
    const size_t tail = count - sizeof(uint32_t);
    return bcmp_xor<uint32_t>(p1, p2) |
           bcmp_xor<uint32_t>(p1 + tail, p2 + tail);
  }
  return bcmp_exact_1_to_3(p1, p2, count);
}

// Compares count in 9..16 using non-adjacent 32-bit loads so LLVM keeps all
// loads in GPRs (avoiding NEON SLP `vmov` and unaligned `ldrd`/`ldm` on Arm).
LIBC_INLINE BcmpReturnType bcmp_9_to_16_32bit(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY((bcmp_xor<uint32_t>(p1 + 4, p2 + 4) |
                     bcmp_xor<uint32_t>(p1 + count - 8, p2 + count - 8)) != 0))
    return BcmpReturnType::nonzero();
  return bcmp_xor<uint32_t>(p1, p2) |
         bcmp_xor<uint32_t>(p1 + count - 4, p2 + count - 4);
}

#ifdef LIBC_TYPES_HAS_INT64

// Compares count in 9..16: predicted early exit on `[4, count)` and direct
// 32-bit XOR return on `[0, 4)` (no `setne`/`cset` or `count`-indexed AGU).
LIBC_INLINE BcmpReturnType bcmp_9_to_16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY((bcmp_xor<uint32_t>(p1 + 4, p2 + 4) |
                     bcmp_xor<uint64_t>(p1 + count - 8, p2 + count - 8)) != 0))
    return BcmpReturnType::nonzero();
  return bcmp_xor<uint32_t>(p1, p2);
}

// Branchless 17..32 comparison using 64-bit loads. Interleaving head and tail
// pairs prevents LLVM from SLP-vectorizing adjacent loads on RVV targets.
LIBC_INLINE BcmpReturnType bcmp_17_to_32_64bit(CPtr p1, CPtr p2, size_t count) {
  const uint64_t d0 = bcmp_xor<uint64_t>(p1, p2) |
                      bcmp_xor<uint64_t>(p1 + count - 16, p2 + count - 16);
  const uint64_t d1 = bcmp_xor<uint64_t>(p1 + 8, p2 + 8) |
                      bcmp_xor<uint64_t>(p1 + count - 8, p2 + count - 8);
  return static_cast<uint32_t>((d0 | d1) != 0);
}

LIBC_INLINE bool bcmp_32_64bit_ne(CPtr p1, CPtr p2) {
  return (bcmp_xor<uint64_t>(p1, p2) | bcmp_xor<uint64_t>(p1 + 8, p2 + 8) |
          bcmp_xor<uint64_t>(p1 + 16, p2 + 16) |
          bcmp_xor<uint64_t>(p1 + 24, p2 + 24)) != 0;
}

#endif // LIBC_TYPES_HAS_INT64

} // namespace internal

// Generic unaligned 32-bit scalar `bcmp` (32-bit Arm with unaligned support,
// x86-32 without SSE2, RV32 with `__riscv_misaligned_fast`). Stride-8 pairing
// keeps all loads in GPRs (no NEON SLP `vmov` or unaligned `ldrd`/`ldm`).
[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_unaligned_32bit(CPtr p1, CPtr p2, size_t count) {
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16_32bit(p1, p2, count);
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < count - 16; i += 16) {
    if (LIBC_UNLIKELY((internal::bcmp_xor<uint32_t>(p1 + i, p2 + i) |
                       internal::bcmp_xor<uint32_t>(p1 + i + 8, p2 + i + 8)) !=
                      0))
      return BcmpReturnType::nonzero();
    if (LIBC_UNLIKELY(
            (internal::bcmp_xor<uint32_t>(p1 + i + 4, p2 + i + 4) |
             internal::bcmp_xor<uint32_t>(p1 + i + 12, p2 + i + 12)) != 0))
      return BcmpReturnType::nonzero();
  }
  const CPtr tail1 = p1 + count - 16;
  const CPtr tail2 = p2 + count - 16;
  if (LIBC_UNLIKELY((internal::bcmp_xor<uint32_t>(tail1 + 4, tail2 + 4) |
                     internal::bcmp_xor<uint32_t>(tail1 + 12, tail2 + 12)) !=
                    0))
    return BcmpReturnType::nonzero();
  return internal::bcmp_xor<uint32_t>(tail1, tail2) |
         internal::bcmp_xor<uint32_t>(tail1 + 8, tail2 + 8);
}

#ifdef LIBC_TYPES_HAS_INT64

// Generic unaligned 64-bit scalar `bcmp` (x86-64 without SSE2, AArch64
// `-mgeneral-regs-only`, RV64 with `__riscv_misaligned_fast`).
[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_unaligned_64bit(CPtr p1, CPtr p2, size_t count) {
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_17_to_32_64bit(p1, p2, count);
  if (LIBC_UNLIKELY(internal::bcmp_32_64bit_ne(p1, p2)))
    return BcmpReturnType::nonzero();
  if (count > 64) {
    const size_t last = count - 32;
    LIBC_LOOP_NOUNROLL
    for (size_t i = 32 - static_cast<size_t>(distance_to_align_down<8>(p1));
         i < last; i += 32) {
      if (LIBC_UNLIKELY(internal::bcmp_32_64bit_ne(p1 + i, p2 + i)))
        return BcmpReturnType::nonzero();
    }
  }
  return static_cast<uint32_t>(
      internal::bcmp_32_64bit_ne(p1 + count - 32, p2 + count - 32));
}

#endif // LIBC_TYPES_HAS_INT64

} // namespace LIBC_NAMESPACE_DECL

#undef LIBC_BCMP_USE_OVERREAD

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H
