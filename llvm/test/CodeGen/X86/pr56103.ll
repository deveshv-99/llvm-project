; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-linux-generic | FileCheck %s

@e = global i16 0, align 2
@a = global i32 0, align 4
@c = global i32 0, align 4
@b = global i64 0, align 8

; Check the test instruction won't be optimizated by peephole opt.

define dso_local i32 @main() nounwind {
; CHECK-LABEL: main:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    movq e@GOTPCREL(%rip), %rax
; CHECK-NEXT:    movw $1, (%rax)
; CHECK-NEXT:    movq b@GOTPCREL(%rip), %rax
; CHECK-NEXT:    movq $1, (%rax)
; CHECK-NEXT:    movq a@GOTPCREL(%rip), %rax
; CHECK-NEXT:    movl (%rax), %ecx
; CHECK-NEXT:    movl $-2, %eax
; CHECK-NEXT:    sarl %cl, %eax
; CHECK-NEXT:    movq c@GOTPCREL(%rip), %rdx
; CHECK-NEXT:    movl (%rdx), %edx
; CHECK-NEXT:    decl %edx
; CHECK-NEXT:    movzwl %ax, %eax
; CHECK-NEXT:    decl %eax
; CHECK-NEXT:    xorl %edx, %eax
; CHECK-NEXT:    notl %ecx
; CHECK-NEXT:    andl %eax, %ecx
; CHECK-NEXT:    testq %rcx, %rcx
; CHECK-NEXT:    jle .LBB0_2
; CHECK-NEXT:  # %bb.1: # %if.end
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  .LBB0_2: # %if.then
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    callq abort@PLT
entry:
  store i16 1, ptr @e, align 2
  store i64 1, ptr @b, align 8
  %0 = load i32, ptr @a, align 4
  %shr = ashr i32 -2, %0
  %1 = load i32, ptr @c, align 4
  %sub = add i32 %1, -1
  %conv2 = zext i32 %sub to i64
  %2 = and i32 %shr, 65535
  %conv3 = zext i32 %2 to i64
  %sub4 = add nsw i64 %conv3, -1
  %xor = xor i64 %sub4, %conv2
  %neg5 = xor i32 %0, -1
  %conv6 = sext i32 %neg5 to i64
  %and = and i64 %xor, %conv6
  %cmp = icmp slt i64 %and, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @abort() #2
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

declare void @abort()