//===-- Bcmp implementation for aarch64 -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/config.h"       // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/generic/bcmp_utils.h"
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType, load_unaligned

#include <stddef.h> // size_t

#ifdef __ARM_NEON
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128
#include <arm_neon.h>
#endif

namespace LIBC_NAMESPACE_DECL {

#ifdef __ARM_NEON
namespace internal {

// Comparing `__uint128_t` halves with `|` keeps 17..32 in GPRs (`ldp` + `cmp` +
// `ccmp` + `cset`) without Neon-to-GPR transfer latency or branch mispredicts;
// LLVM would otherwise vectorize 64-bit comparisons into Neon.
LIBC_INLINE BcmpReturnType bcmp_aarch64_17_to_32(CPtr p1, CPtr p2,
                                                 size_t count) {
#ifdef LIBC_TYPES_HAS_INT128
  return static_cast<uint32_t>(
      (bcmp_xor<__uint128_t>(p1, p2) |
       bcmp_xor<__uint128_t>(p1 + count - 16, p2 + count - 16)) != 0);
#else
  return bcmp_17_to_32_64bit(p1, p2, count);
#endif
}

LIBC_INLINE uint8x16_t aarch64_xor16(CPtr p1, CPtr p2) {
  return veorq_u8(load_unaligned<uint8x16_t>(p1),
                  load_unaligned<uint8x16_t>(p2));
}

LIBC_INLINE uint8x16_t aarch64_xor32(CPtr p1, CPtr p2) {
  return vorrq_u8(aarch64_xor16(p1, p2), aarch64_xor16(p1 + 16, p2 + 16));
}

LIBC_INLINE bool aarch64_neon_nonzero(uint8x16_t diff) {
  return vmaxvq_u32(vreinterpretq_u32_u8(diff)) != 0;
}

LIBC_INLINE bool aarch64_neon_64_ne(CPtr p1, CPtr p2) {
  return aarch64_neon_nonzero(
      vorrq_u8(aarch64_xor32(p1, p2), aarch64_xor32(p1 + 32, p2 + 32)));
}

} // namespace internal
#endif // __ARM_NEON

// Sizes > 32 pair Neon early-exit checks with a GPR `[0, 4)` final return on
// `33..128` (hiding Neon-to-GPR latency on equal inputs), while `> 128` checks
// `[0, 4)` before looping so early mismatches exit immediately.
[[maybe_unused]] LIBC_INLINE BcmpReturnType inline_bcmp_aarch64(CPtr p1,
                                                                CPtr p2,
                                                                size_t count) {
#ifdef __ARM_NEON
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_aarch64_17_to_32(p1, p2, count);
  const uint8x16_t head = vorrq_u8(internal::aarch64_xor16(p1 + 4, p2 + 4),
                                   internal::aarch64_xor16(p1 + 16, p2 + 16));
  if (count <= 64) {
    if (LIBC_UNLIKELY(internal::aarch64_neon_nonzero(vorrq_u8(
            head, internal::aarch64_xor32(p1 + count - 32, p2 + count - 32)))))
      return BcmpReturnType::nonzero();
    return internal::bcmp_xor<uint32_t>(p1, p2);
  }
  if (LIBC_UNLIKELY(internal::aarch64_neon_nonzero(
          vorrq_u8(head, internal::aarch64_xor32(p1 + 32, p2 + 32)))))
    return BcmpReturnType::nonzero();
  if (count > 128) {
    if (LIBC_UNLIKELY(internal::bcmp_xor<uint32_t>(p1, p2) != 0))
      return BcmpReturnType::nonzero();
    const size_t last = count - 64;
    for (size_t i = 64 - static_cast<size_t>(distance_to_align_down<16>(p1));
         i < last; i += 64) {
      if (LIBC_UNLIKELY(internal::aarch64_neon_64_ne(p1 + i, p2 + i)))
        return BcmpReturnType::nonzero();
    }
  }
  if (LIBC_UNLIKELY(
          internal::aarch64_neon_64_ne(p1 + count - 64, p2 + count - 64)))
    return BcmpReturnType::nonzero();
  return internal::bcmp_xor<uint32_t>(p1, p2);
#else
  return inline_bcmp_unaligned_64bit(p1, p2, count);
#endif // __ARM_NEON
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H
