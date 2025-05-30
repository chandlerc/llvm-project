; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=cmov | FileCheck %s --check-prefixes=X86,X86-NOSSE
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefixes=X86,X86-SSE2

; This tests codegen time inlining/optimization of memcmp
; rdar://6480398

@.str = private constant [65 x i8] c"0123456789012345678901234567890123456789012345678901234567890123\00", align 1

declare dso_local i32 @memcmp(ptr, ptr, i32)
declare dso_local i32 @bcmp(ptr, ptr, i32)

define i32 @length2(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length2:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movzwl (%ecx), %ecx
; X86-NEXT:    movzwl (%eax), %edx
; X86-NEXT:    rolw $8, %cx
; X86-NEXT:    rolw $8, %dx
; X86-NEXT:    movzwl %cx, %eax
; X86-NEXT:    movzwl %dx, %ecx
; X86-NEXT:    subl %ecx, %eax
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 2) nounwind
  ret i32 %m
}

define i1 @length2_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length2_eq:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movzwl (%ecx), %ecx
; X86-NEXT:    cmpw (%eax), %cx
; X86-NEXT:    sete %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 2) nounwind
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

define i1 @length2_eq_const(ptr %X) nounwind !prof !14 {
; X86-LABEL: length2_eq_const:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    cmpw $12849, (%eax) # imm = 0x3231
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr getelementptr inbounds ([65 x i8], ptr @.str, i32 0, i32 1), i32 2) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i1 @length2_eq_nobuiltin_attr(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length2_eq_nobuiltin_attr:
; X86:       # %bb.0:
; X86-NEXT:    pushl $2
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    sete %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 2) nounwind nobuiltin
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

define i32 @length3(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length3:
; X86:       # %bb.0:
; X86-NEXT:    pushl %esi
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movzwl (%eax), %edx
; X86-NEXT:    movzwl (%ecx), %esi
; X86-NEXT:    rolw $8, %dx
; X86-NEXT:    rolw $8, %si
; X86-NEXT:    cmpw %si, %dx
; X86-NEXT:    jne .LBB4_3
; X86-NEXT:  # %bb.1: # %loadbb1
; X86-NEXT:    movzbl 2(%eax), %eax
; X86-NEXT:    movzbl 2(%ecx), %ecx
; X86-NEXT:    subl %ecx, %eax
; X86-NEXT:    jmp .LBB4_2
; X86-NEXT:  .LBB4_3: # %res_block
; X86-NEXT:    setae %al
; X86-NEXT:    movzbl %al, %eax
; X86-NEXT:    leal -1(%eax,%eax), %eax
; X86-NEXT:  .LBB4_2: # %endblock
; X86-NEXT:    popl %esi
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 3) nounwind
  ret i32 %m
}

define i1 @length3_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length3_eq:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movzwl (%ecx), %edx
; X86-NEXT:    xorw (%eax), %dx
; X86-NEXT:    movb 2(%ecx), %cl
; X86-NEXT:    xorb 2(%eax), %cl
; X86-NEXT:    movzbl %cl, %eax
; X86-NEXT:    orw %dx, %ax
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 3) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i32 @length4(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length4:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl (%ecx), %ecx
; X86-NEXT:    movl (%eax), %eax
; X86-NEXT:    bswapl %ecx
; X86-NEXT:    bswapl %eax
; X86-NEXT:    cmpl %eax, %ecx
; X86-NEXT:    seta %al
; X86-NEXT:    sbbb $0, %al
; X86-NEXT:    movsbl %al, %eax
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 4) nounwind
  ret i32 %m
}

define i1 @length4_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length4_eq:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl (%ecx), %ecx
; X86-NEXT:    cmpl (%eax), %ecx
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 4) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i1 @length4_eq_const(ptr %X) nounwind !prof !14 {
; X86-LABEL: length4_eq_const:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    cmpl $875770417, (%eax) # imm = 0x34333231
; X86-NEXT:    sete %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr getelementptr inbounds ([65 x i8], ptr @.str, i32 0, i32 1), i32 4) nounwind
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

define i32 @length5(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length5:
; X86:       # %bb.0:
; X86-NEXT:    pushl %esi
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl (%eax), %edx
; X86-NEXT:    movl (%ecx), %esi
; X86-NEXT:    bswapl %edx
; X86-NEXT:    bswapl %esi
; X86-NEXT:    cmpl %esi, %edx
; X86-NEXT:    jne .LBB9_3
; X86-NEXT:  # %bb.1: # %loadbb1
; X86-NEXT:    movzbl 4(%eax), %eax
; X86-NEXT:    movzbl 4(%ecx), %ecx
; X86-NEXT:    subl %ecx, %eax
; X86-NEXT:    jmp .LBB9_2
; X86-NEXT:  .LBB9_3: # %res_block
; X86-NEXT:    setae %al
; X86-NEXT:    movzbl %al, %eax
; X86-NEXT:    leal -1(%eax,%eax), %eax
; X86-NEXT:  .LBB9_2: # %endblock
; X86-NEXT:    popl %esi
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 5) nounwind
  ret i32 %m
}

define i1 @length5_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length5_eq:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl (%ecx), %edx
; X86-NEXT:    xorl (%eax), %edx
; X86-NEXT:    movb 4(%ecx), %cl
; X86-NEXT:    xorb 4(%eax), %cl
; X86-NEXT:    movzbl %cl, %eax
; X86-NEXT:    orl %edx, %eax
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 5) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i32 @length8(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length8:
; X86:       # %bb.0:
; X86-NEXT:    pushl %esi
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-NEXT:    movl (%esi), %ecx
; X86-NEXT:    movl (%eax), %edx
; X86-NEXT:    bswapl %ecx
; X86-NEXT:    bswapl %edx
; X86-NEXT:    cmpl %edx, %ecx
; X86-NEXT:    jne .LBB11_2
; X86-NEXT:  # %bb.1: # %loadbb1
; X86-NEXT:    movl 4(%esi), %ecx
; X86-NEXT:    movl 4(%eax), %edx
; X86-NEXT:    bswapl %ecx
; X86-NEXT:    bswapl %edx
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    cmpl %edx, %ecx
; X86-NEXT:    je .LBB11_3
; X86-NEXT:  .LBB11_2: # %res_block
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    cmpl %edx, %ecx
; X86-NEXT:    setae %al
; X86-NEXT:    leal -1(%eax,%eax), %eax
; X86-NEXT:  .LBB11_3: # %endblock
; X86-NEXT:    popl %esi
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 8) nounwind
  ret i32 %m
}

define i1 @length8_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length8_eq:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl (%ecx), %edx
; X86-NEXT:    movl 4(%ecx), %ecx
; X86-NEXT:    xorl (%eax), %edx
; X86-NEXT:    xorl 4(%eax), %ecx
; X86-NEXT:    orl %edx, %ecx
; X86-NEXT:    sete %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 8) nounwind
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

define i1 @length8_eq_const(ptr %X) nounwind !prof !14 {
; X86-LABEL: length8_eq_const:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl $858927408, %ecx # imm = 0x33323130
; X86-NEXT:    xorl (%eax), %ecx
; X86-NEXT:    movl $926299444, %edx # imm = 0x37363534
; X86-NEXT:    xorl 4(%eax), %edx
; X86-NEXT:    orl %ecx, %edx
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr @.str, i32 8) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i1 @length12_eq(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length12_eq:
; X86:       # %bb.0:
; X86-NEXT:    pushl $12
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 12) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i32 @length12(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length12:
; X86:       # %bb.0:
; X86-NEXT:    pushl $12
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 12) nounwind
  ret i32 %m
}

; PR33329 - https://bugs.llvm.org/show_bug.cgi?id=33329

define i32 @length16(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length16:
; X86:       # %bb.0:
; X86-NEXT:    pushl $16
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 16) nounwind
  ret i32 %m
}

define i1 @length16_eq(ptr %x, ptr %y) nounwind !prof !14 {
; X86-NOSSE-LABEL: length16_eq:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $16
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    setne %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length16_eq:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE2-NEXT:    movdqu (%ecx), %xmm0
; X86-SSE2-NEXT:    movdqu (%eax), %xmm1
; X86-SSE2-NEXT:    pcmpeqb %xmm0, %xmm1
; X86-SSE2-NEXT:    pmovmskb %xmm1, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    setne %al
; X86-SSE2-NEXT:    retl
  %call = tail call i32 @memcmp(ptr %x, ptr %y, i32 16) nounwind
  %cmp = icmp ne i32 %call, 0
  ret i1 %cmp
}

define i1 @length16_eq_const(ptr %X) nounwind !prof !14 {
; X86-NOSSE-LABEL: length16_eq_const:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $16
; X86-NOSSE-NEXT:    pushl $.L.str
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    sete %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length16_eq_const:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movdqu (%eax), %xmm0
; X86-SSE2-NEXT:    pcmpeqb {{\.?LCPI[0-9]+_[0-9]+}}, %xmm0
; X86-SSE2-NEXT:    pmovmskb %xmm0, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    sete %al
; X86-SSE2-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr @.str, i32 16) nounwind
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

; PR33914 - https://bugs.llvm.org/show_bug.cgi?id=33914

define i32 @length24(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length24:
; X86:       # %bb.0:
; X86-NEXT:    pushl $24
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 24) nounwind
  ret i32 %m
}

define i1 @length24_eq(ptr %x, ptr %y) nounwind !prof !14 {
; X86-NOSSE-LABEL: length24_eq:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $24
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    sete %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length24_eq:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE2-NEXT:    movdqu (%ecx), %xmm0
; X86-SSE2-NEXT:    movdqu 8(%ecx), %xmm1
; X86-SSE2-NEXT:    movdqu (%eax), %xmm2
; X86-SSE2-NEXT:    pcmpeqb %xmm0, %xmm2
; X86-SSE2-NEXT:    movdqu 8(%eax), %xmm0
; X86-SSE2-NEXT:    pcmpeqb %xmm1, %xmm0
; X86-SSE2-NEXT:    pand %xmm2, %xmm0
; X86-SSE2-NEXT:    pmovmskb %xmm0, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    sete %al
; X86-SSE2-NEXT:    retl
  %call = tail call i32 @memcmp(ptr %x, ptr %y, i32 24) nounwind
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}

define i1 @length24_eq_const(ptr %X) nounwind !prof !14 {
; X86-NOSSE-LABEL: length24_eq_const:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $24
; X86-NOSSE-NEXT:    pushl $.L.str
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    setne %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length24_eq_const:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movdqu (%eax), %xmm0
; X86-SSE2-NEXT:    movdqu 8(%eax), %xmm1
; X86-SSE2-NEXT:    pcmpeqb {{\.?LCPI[0-9]+_[0-9]+}}, %xmm1
; X86-SSE2-NEXT:    pcmpeqb {{\.?LCPI[0-9]+_[0-9]+}}, %xmm0
; X86-SSE2-NEXT:    pand %xmm1, %xmm0
; X86-SSE2-NEXT:    pmovmskb %xmm0, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    setne %al
; X86-SSE2-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr @.str, i32 24) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i32 @length32(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length32:
; X86:       # %bb.0:
; X86-NEXT:    pushl $32
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 32) nounwind
  ret i32 %m
}

; PR33325 - https://bugs.llvm.org/show_bug.cgi?id=33325

define i1 @length32_eq(ptr %x, ptr %y) nounwind !prof !14 {
; X86-NOSSE-LABEL: length32_eq:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $32
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    sete %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length32_eq:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE2-NEXT:    movdqu (%ecx), %xmm0
; X86-SSE2-NEXT:    movdqu 16(%ecx), %xmm1
; X86-SSE2-NEXT:    movdqu (%eax), %xmm2
; X86-SSE2-NEXT:    pcmpeqb %xmm0, %xmm2
; X86-SSE2-NEXT:    movdqu 16(%eax), %xmm0
; X86-SSE2-NEXT:    pcmpeqb %xmm1, %xmm0
; X86-SSE2-NEXT:    pand %xmm2, %xmm0
; X86-SSE2-NEXT:    pmovmskb %xmm0, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    sete %al
; X86-SSE2-NEXT:    retl
  %call = tail call i32 @memcmp(ptr %x, ptr %y, i32 32) nounwind
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}

define i1 @length32_eq_const(ptr %X) nounwind !prof !14 {
; X86-NOSSE-LABEL: length32_eq_const:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl $32
; X86-NOSSE-NEXT:    pushl $.L.str
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    calll memcmp
; X86-NOSSE-NEXT:    addl $12, %esp
; X86-NOSSE-NEXT:    testl %eax, %eax
; X86-NOSSE-NEXT:    setne %al
; X86-NOSSE-NEXT:    retl
;
; X86-SSE2-LABEL: length32_eq_const:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movdqu (%eax), %xmm0
; X86-SSE2-NEXT:    movdqu 16(%eax), %xmm1
; X86-SSE2-NEXT:    pcmpeqb {{\.?LCPI[0-9]+_[0-9]+}}, %xmm1
; X86-SSE2-NEXT:    pcmpeqb {{\.?LCPI[0-9]+_[0-9]+}}, %xmm0
; X86-SSE2-NEXT:    pand %xmm1, %xmm0
; X86-SSE2-NEXT:    pmovmskb %xmm0, %eax
; X86-SSE2-NEXT:    cmpl $65535, %eax # imm = 0xFFFF
; X86-SSE2-NEXT:    setne %al
; X86-SSE2-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr @.str, i32 32) nounwind
  %c = icmp ne i32 %m, 0
  ret i1 %c
}

define i32 @length64(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: length64:
; X86:       # %bb.0:
; X86-NEXT:    pushl $64
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr %Y, i32 64) nounwind
  ret i32 %m
}

define i1 @length64_eq(ptr %x, ptr %y) nounwind !prof !14 {
; X86-LABEL: length64_eq:
; X86:       # %bb.0:
; X86-NEXT:    pushl $64
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %call = tail call i32 @memcmp(ptr %x, ptr %y, i32 64) nounwind
  %cmp = icmp ne i32 %call, 0
  ret i1 %cmp
}

define i1 @length64_eq_const(ptr %X) nounwind !prof !14 {
; X86-LABEL: length64_eq_const:
; X86:       # %bb.0:
; X86-NEXT:    pushl $64
; X86-NEXT:    pushl $.L.str
; X86-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NEXT:    calll memcmp
; X86-NEXT:    addl $12, %esp
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    sete %al
; X86-NEXT:    retl
  %m = tail call i32 @memcmp(ptr %X, ptr @.str, i32 64) nounwind
  %c = icmp eq i32 %m, 0
  ret i1 %c
}

define i32 @bcmp_length2(ptr %X, ptr %Y) nounwind !prof !14 {
; X86-LABEL: bcmp_length2:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movzwl (%eax), %edx
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    cmpw (%ecx), %dx
; X86-NEXT:    setne %al
; X86-NEXT:    retl
  %m = tail call i32 @bcmp(ptr %X, ptr %Y, i32 2) nounwind
  ret i32 %m
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i32 10000}
!4 = !{!"MaxCount", i32 10}
!5 = !{!"MaxInternalCount", i32 1}
!6 = !{!"MaxFunctionCount", i32 1000}
!7 = !{!"NumCounts", i32 3}
!8 = !{!"NumFunctions", i32 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i32 100, i32 1}
!12 = !{i32 999000, i32 100, i32 1}
!13 = !{i32 999999, i32 1, i32 2}
!14 = !{!"function_entry_count", i32 0}
