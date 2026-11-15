// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-out-of-line-traps -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -o - %s | FileCheck %s --check-prefix=INLINE --implicit-check-not='"kcfi-out-of-line-traps"'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=NOKCFI --implicit-check-not='"kcfi-out-of-line-traps"'

// CHECK: ![[#]] = !{i32 8, !"kcfi-out-of-line-traps", i32 1}
// INLINE: ![[#]] = !{i32 4, !"kcfi", i32 1}
// NOKCFI: !llvm.ident
