// RUN: mlir-opt %s -convert-gpu-to-nvvm='has-redux=1' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='has-redux=1 allowed-dialects=func,arith,cf' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='has-redux=1 use-bare-ptr-memref-call-conv=1' -split-input-file | FileCheck %s --check-prefix=CHECK-BARE
// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

gpu.module @test_module_0 {
  // CHECK-LABEL: func @gpu_index_ops()
  func.func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index,
          index) {

    // CHECK: = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = gpu.thread_id x
    // CHECK: = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = gpu.thread_id y
    // CHECK: = nvvm.read.ptx.sreg.tid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = gpu.thread_id z

    // CHECK: = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = gpu.block_dim x
    // CHECK: = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = gpu.block_dim y
    // CHECK: = nvvm.read.ptx.sreg.ntid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = gpu.block_dim z

    // CHECK: = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = gpu.block_id x
    // CHECK: = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = gpu.block_id y
    // CHECK: = nvvm.read.ptx.sreg.ctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = gpu.block_id z

    // CHECK: = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = gpu.grid_dim x
    // CHECK: = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = gpu.grid_dim y
    // CHECK: = nvvm.read.ptx.sreg.nctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimZ = gpu.grid_dim z


    // CHECK: = nvvm.read.ptx.sreg.laneid range <i32, 0, 32> : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %laneId = gpu.lane_id

    func.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ,
               %laneId
        : index, index, index, index, index, index,
          index, index, index, index, index, index,
          index
  }
}



gpu.module @test_module_1 {
  // CHECK-LABEL: func @gpu_index_comp
  func.func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    %0 = arith.addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    func.return %0 : index
  }
}



gpu.module @test_module_2 {
  // CHECK-LABEL: func @gpu_all_reduce_op()
  gpu.func @gpu_all_reduce_op() {
    %arg0 = arith.constant 1.0 : f32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync bfly {{.*}}
    // CHECK: nvvm.barrier0
    // CHECK: llvm.fadd
    %result = gpu.all_reduce add %arg0 uniform {} : (f32) -> (f32)

    gpu.return
  }
}



gpu.module @test_module_3 {
  // CHECK-LABEL: func @gpu_all_reduce_region()
  gpu.func @gpu_all_reduce_region() {
    %arg0 = arith.constant 1 : i32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync bfly {{.*}}
    // CHECK: nvvm.barrier0
    %result = gpu.all_reduce %arg0 uniform {
    ^bb(%lhs : i32, %rhs : i32):
      %xor = arith.xori %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    } : (i32) -> (i32)
    gpu.return
  }
}



gpu.module @test_module_4 {
  // CHECK-LABEL: func @gpu_shuffle()
  func.func @gpu_shuffle() -> (f32, f32, f32, f32, i1, i1, i1, i1) {
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %arg0 = arith.constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : i32
    %arg1 = arith.constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : i32
    %arg2 = arith.constant 23 : i32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#CLAMP:]] = llvm.sub %[[#WIDTH]], %[[#ONE]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync bfly %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#CLAMP]] {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1] : !llvm.struct<(f32, i1)>
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync up %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#NUM_LANES]] {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1] : !llvm.struct<(f32, i1)>
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync down {{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync idx {{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli, %pred, %predu, %predd, %predi
      : f32, f32,f32, f32, i1, i1, i1, i1
  }

  // CHECK-LABEL: func @gpu_shuffle_unused_pred()
  func.func @gpu_shuffle_unused_pred() -> (f32, f32, f32, f32) {
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %arg0 = arith.constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : i32
    %arg1 = arith.constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : i32
    %arg2 = arith.constant 23 : i32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#CLAMP:]] = llvm.sub %[[#WIDTH]], %[[#ONE]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync bfly %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#CLAMP]] : f32 -> f32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync up %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#NUM_LANES]] : f32 -> f32
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync down {{.*}} : f32 -> f32
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync idx {{.*}} : f32 -> f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli : f32, f32,f32, f32
  }
}

gpu.module @test_module_5 {
  // CHECK-LABEL: func @gpu_sync()
  func.func @gpu_sync() {
    // CHECK: nvvm.barrier0
    gpu.barrier
    func.return
  }
}



gpu.module @test_module_6 {
  // CHECK: llvm.func @__nv_fabsf(f32) -> f32
  // CHECK: llvm.func @__nv_fabs(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  func.func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.absf %arg_f32 : f32
    // CHECK: llvm.call @__nv_fabsf(%{{.*}}) : (f32) -> f32
    %result64 = math.absf %arg_f64 : f64
    // CHECK: llvm.call @__nv_fabs(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_7 {
  // CHECK: llvm.func @__nv_cbrtf(f32) -> f32
  // CHECK: llvm.func @__nv_cbrt(f64) -> f64
  // CHECK-LABEL: func @gpu_cbrt
  func.func @gpu_cbrt(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cbrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_cbrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.cbrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_cbrt(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_8 {
  // CHECK: llvm.func @__nv_ceilf(f32) -> f32
  // CHECK: llvm.func @__nv_ceil(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  func.func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.ceil %arg_f32 : f32
    // CHECK: llvm.call @__nv_ceilf(%{{.*}}) : (f32) -> f32
    %result64 = math.ceil %arg_f64 : f64
    // CHECK: llvm.call @__nv_ceil(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_9 {
  // CHECK: llvm.func @__nv_floorf(f32) -> f32
  // CHECK: llvm.func @__nv_floor(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  func.func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.floor %arg_f32 : f32
    // CHECK: llvm.call @__nv_floorf(%{{.*}}) : (f32) -> f32
    %result64 = math.floor %arg_f64 : f64
    // CHECK: llvm.call @__nv_floor(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_10 {
  // CHECK: llvm.func @__nv_cosf(f32) -> f32
  // CHECK: llvm.func @__nv_cos(f64) -> f64
  // CHECK: llvm.func @__nv_fast_cosf(f32) -> f32
  // CHECK-LABEL: func @gpu_cos
  func.func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__nv_cosf(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__nv_cos(%{{.*}}) : (f64) -> f64
    %result32Fast = math.cos %arg_f32 fastmath<afn> : f32
    // CHECK: llvm.call @__nv_fast_cosf(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}


gpu.module @test_module_11 {
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK: llvm.func @__nv_fast_expf(f32) -> f32
  // CHECK-LABEL: func @gpu_exp
  func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
    %result32Fast = math.exp %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_fast_expf(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}


gpu.module @test_module_12 {
  // CHECK: llvm.func @__nv_exp2f(f32) -> f32
  // CHECK: llvm.func @__nv_exp2(f64) -> f64
  // CHECK-LABEL: func @gpu_exp2
  func.func @gpu_exp2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_exp2f(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp2(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_13 {
  // CHECK: llvm.func @__nv_logf(f32) -> f32
  // CHECK: llvm.func @__nv_log(f64) -> f64
  // CHECK: llvm.func @__nv_fast_logf(f32) -> f32
  // CHECK-LABEL: func @gpu_log
  func.func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__nv_logf(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__nv_log(%{{.*}}) : (f64) -> f64
    %result32Fast = math.log %arg_f32 fastmath<afn> : f32
    // CHECK: llvm.call @__nv_fast_logf(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}



gpu.module @test_module_14 {
  // CHECK: llvm.func @__nv_log10f(f32) -> f32
  // CHECK: llvm.func @__nv_log10(f64) -> f64
  // CHECK: llvm.func @__nv_fast_log10f(f32) -> f32
  // CHECK-LABEL: func @gpu_log10
  func.func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log10f(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log10(%{{.*}}) : (f64) -> f64
    %result32Fast = math.log10 %arg_f32 fastmath<afn> : f32
    // CHECK: llvm.call @__nv_fast_log10f(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}



gpu.module @test_module_15 {
  // CHECK: llvm.func @__nv_log1pf(f32) -> f32
  // CHECK: llvm.func @__nv_log1p(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  func.func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__nv_log1pf(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__nv_log1p(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_16 {
  // CHECK: llvm.func @__nv_log2f(f32) -> f32
  // CHECK: llvm.func @__nv_log2(f64) -> f64
  // CHECK: llvm.func @__nv_fast_log2f(f32) -> f32
  // CHECK-LABEL: func @gpu_log2
  func.func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log2f(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log2(%{{.*}}) : (f64) -> f64
    %result32Fast = math.log2 %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_fast_log2f(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}



gpu.module @test_module_17 {
  // CHECK: llvm.func @__nv_sinf(f32) -> f32
  // CHECK: llvm.func @__nv_sin(f64) -> f64
  // CHECK: llvm.func @__nv_fast_sinf(f32) -> f32
  // CHECK-LABEL: func @gpu_sin
  func.func @gpu_sin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.sin %arg_f32 : f32
    // CHECK: llvm.call @__nv_sinf(%{{.*}}) : (f32) -> f32
    %result64 = math.sin %arg_f64 : f64
    // CHECK: llvm.call @__nv_sin(%{{.*}}) : (f64) -> f64
    %result32Fast = math.sin %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_fast_sinf(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}



gpu.module @test_module_18 {
  // CHECK: llvm.func @__nv_tanf(f32) -> f32
  // CHECK: llvm.func @__nv_tan(f64) -> f64
  // CHECK: llvm.func @__nv_fast_tanf(f32) -> f32
  // CHECK-LABEL: func @gpu_tan
  func.func @gpu_tan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64, f32) {
    %result16 = math.tan %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_tanf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.tan %arg_f32 : f32
    // CHECK: llvm.call @__nv_tanf(%{{.*}}) : (f32) -> f32
    %result64 = math.tan %arg_f64 : f64
    // CHECK: llvm.call @__nv_tan(%{{.*}}) : (f64) -> f64
    %result32Fast = math.tan %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_fast_tanf(%{{.*}}) : (f32) -> f32
    func.return %result16, %result32, %result64, %result32Fast : f16, f32, f64, f32
  }
}



gpu.module @test_module_19 {
  // CHECK: llvm.func @__nv_tanhf(f32) -> f32
  // CHECK: llvm.func @__nv_tanh(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  func.func @gpu_tanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.tanh %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__nv_tanh(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}



gpu.module @test_module_20 {
  // CHECK: llvm.func @__nv_rsqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_rsqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  func.func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_rsqrt(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}



gpu.module @test_module_21 {
  // CHECK: llvm.func @__nv_sqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_sqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  func.func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_sqrt(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}



gpu.module @test_module_22 {
  // CHECK: llvm.func @__nv_atanf(f32) -> f32
  // CHECK: llvm.func @__nv_atan(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  func.func @gpu_atan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atan %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__nv_atan(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}



gpu.module @test_module_23 {
  // CHECK: llvm.func @__nv_atan2f(f32, f32) -> f32
  // CHECK: llvm.func @__nv_atan2(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  func.func @gpu_atan2(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atan2 %arg_f16, %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atan2f(%{{.*}}) : (f32, f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_atan2f(%{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_atan2(%{{.*}}) : (f64, f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}



// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module_24 {
  "test.symbol_scope"() ({
  // CHECK: test.symbol_scope
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK: llvm.func @__nv_fast_expf(f32) -> f32
  // CHECK-LABEL: func @gpu_exp
    func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
      %result32 = math.exp %arg_f32 : f32
      // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
      %result64 = math.exp %arg_f64 : f64
      // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
      %result32Fast = math.exp %arg_f32 fastmath<afn> : f32
      // CHECK: llvm.call @__nv_fast_expf(%{{.*}}) : (f32) -> f32
      func.return %result32, %result64, %result32Fast : f32, f64, f32
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}



gpu.module @test_module_25 {
  // CHECK: llvm.func @__nv_expm1f(f32) -> f32
  // CHECK: llvm.func @__nv_expm1(f64) -> f64
  // CHECK-LABEL: func @gpu_expm1
  func.func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__nv_expm1f(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__nv_expm1(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}



gpu.module @test_module_26 {
  // CHECK: llvm.func @__nv_powf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_pow(f64, f64) -> f64
  // CHECK: llvm.func @__nv_fast_powf(f32, f32) -> f32
  // CHECK-LABEL: func @gpu_pow
  func.func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_powf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_pow(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    %result32Fast = math.powf %arg_f32, %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_fast_powf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}



gpu.module @test_module_27 {
  // CHECK-LABEL: func @gpu_unroll
  func.func @gpu_unroll(%arg0 : vector<4xf32>) -> vector<4xf32> {
    %result = math.exp %arg0 : vector<4xf32>
    // CHECK: %[[V0:.+]] = llvm.mlir.poison : vector<4xf32>
    // CHECK: %[[CL:.+]] = llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V1:.+]] = llvm.insertelement %[[CL]], %[[V0]]
    // CHECK: %[[CL:.+]] = llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V2:.+]] = llvm.insertelement %[[CL]], %[[V1]]
    // CHECK: %[[CL:.+]] = llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V3:.+]] = llvm.insertelement %[[CL]], %[[V2]]
    // CHECK: %[[CL:.+]] = llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V4:.+]] = llvm.insertelement %[[CL]], %[[V3]]
    // CHECK: return %[[V4]]
    func.return %result : vector<4xf32>
  }
}



gpu.module @test_module_28 {
  // CHECK-LABEL: @kernel_func
  // CHECK: attributes
  // CHECK: gpu.kernel
  // CHECK: nvvm.kernel
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}



gpu.module @test_module_29 {
  // CHECK-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL0:[A-Za-z0-9_]+]]("Hello, world\0A\00")
  // CHECK-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL1:[A-Za-z0-9_]+]]("Hello: %d\0A\00")
  // CHECK-DAG: llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32

  // CHECK-LABEL: func @test_const_printf
  gpu.func @test_const_printf() {
    // CHECK-NEXT: %[[FORMATSTR:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL0]] : !llvm.ptr
    // CHECK-NEXT: %[[FORMATSTART:.*]] = llvm.getelementptr %[[FORMATSTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    // CHECK-NEXT: %[[O:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT: %[[ALLOC:.*]] = llvm.alloca %[[O]] x !llvm.struct<()> : (i64) -> !llvm.ptr
    // CHECK-NEXT: llvm.call @vprintf(%[[FORMATSTART]], %[[ALLOC]]) : (!llvm.ptr, !llvm.ptr) -> i32
    gpu.printf "Hello, world\n"

    // Make sure that the same global is reused.
    // CHECK: %[[FORMATSTR2:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL0]] : !llvm.ptr
    // CHECK: %[[FORMATSTART2:.*]] = llvm.getelementptr %[[FORMATSTR2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    // CHECK: llvm.call @vprintf(%[[FORMATSTART2]], %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
    gpu.printf "Hello, world\n"

    gpu.return
  }

  // CHECK-LABEL: func @test_printf
  // CHECK: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: f32)
  gpu.func @test_printf(%arg0: i32, %arg1: f32) {
    // CHECK-NEXT: %[[FORMATSTR:.*]] = llvm.mlir.addressof @[[$PRINT_GLOBAL1]] : !llvm.ptr
    // CHECK-NEXT: %[[FORMATSTART:.*]] = llvm.getelementptr %[[FORMATSTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
    // CHECK-NEXT: %[[EXT:.+]] = llvm.fpext %[[ARG1]] : f32 to f64
    // CHECK-NEXT: %[[O:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT: %[[ALLOC:.*]] = llvm.alloca %[[O]] x !llvm.struct<(i32, f64)> : (i64) -> !llvm.ptr
    // CHECK-NEXT: %[[EL0:.*]] = llvm.getelementptr %[[ALLOC]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    // CHECK-NEXT: llvm.store %[[ARG0]], %[[EL0]] : i32, !llvm.ptr
    // CHECK-NEXT: %[[EL1:.*]] = llvm.getelementptr %[[ALLOC]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    // CHECK-NEXT: llvm.store %[[EXT]], %[[EL1]] : f64, !llvm.ptr
    // CHECK-NEXT: llvm.call @vprintf(%[[FORMATSTART]], %[[ALLOC]]) : (!llvm.ptr, !llvm.ptr) -> i32
    gpu.printf "Hello: %d\n", %arg0, %arg1 : i32, f32
    gpu.return
  }
}



gpu.module @test_module_30 {
  // CHECK-LABEL: func @subgroup_reduce_add
  gpu.func @subgroup_reduce_add(%arg0 : i32) {
    // CHECK: nvvm.redux.sync add {{.*}}
    %result = gpu.subgroup_reduce add %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
  // CHECK-LABEL: @subgroup_reduce_minsi
  gpu.func @subgroup_reduce_minsi(%arg0 : i32) {
    // CHECK: nvvm.redux.sync min {{.*}}
    %result = gpu.subgroup_reduce minsi %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
  // CHECK-LABEL:  @subgroup_reduce_maxsi
  gpu.func @subgroup_reduce_maxsi(%arg0 : i32) {
    // CHECK: nvvm.redux.sync max {{.*}}
    %result = gpu.subgroup_reduce maxsi %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
  // CHECK-LABEL: func @subgroup_reduce_and
  gpu.func @subgroup_reduce_and(%arg0 : i32) {
    // CHECK: nvvm.redux.sync and {{.*}}
    %result = gpu.subgroup_reduce and %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
  // CHECK-LABEL:  @subgroup_reduce_or
  gpu.func @subgroup_reduce_or(%arg0 : i32) {
    // CHECK: nvvm.redux.sync or {{.*}}
    %result = gpu.subgroup_reduce or %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
  // CHECK-LABEL: @subgroup_reduce_xor
  gpu.func @subgroup_reduce_xor(%arg0 : i32) {
    // CHECK: nvvm.redux.sync xor {{.*}}
    %result = gpu.subgroup_reduce xor %arg0 uniform {} : (i32) -> (i32)
    gpu.return
  }
}

gpu.module @test_module_31 {
  // CHECK: llvm.func @__nv_fmodf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_fmod(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_fmod
  func.func @gpu_fmod(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = arith.remf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_fmodf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = arith.remf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_fmod(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_32 {
  // CHECK: llvm.func @__nv_erff(f32) -> f32
  // CHECK: llvm.func @__nv_erf(f64) -> f64
  // CHECK-LABEL: func @gpu_erf
  func.func @gpu_erf(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.erf %arg_f32 : f32
    // CHECK: llvm.call @__nv_erff(%{{.*}}) : (f32) -> f32
    %result64 = math.erf %arg_f64 : f64
    // CHECK: llvm.call @__nv_erf(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_33 {
// CHECK-LABEL: func @kernel_with_block_size(
// CHECK: attributes {gpu.kernel, gpu.known_block_size = array<i32: 32, 4, 2>, nvvm.kernel, nvvm.maxntid = array<i32: 32, 4, 2>}
  gpu.func @kernel_with_block_size(%arg0: !llvm.ptr) kernel attributes {known_block_size = array<i32: 32, 4, 2>} {
    // CHECK: = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
    %0 = gpu.thread_id x
    // CHECK: = nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = gpu.thread_id y
    // CHECK: = nvvm.read.ptx.sreg.tid.z range <i32, 0, 2> : i32
    %2 = gpu.thread_id z

    // Fake usage to prevent dead code elimination
    %3 = arith.addi %0, %1 : index
    %4 = arith.addi %3, %2 : index
    %5 = arith.index_cast %4 : index to i64
    llvm.store %5, %arg0 : i64, !llvm.ptr
    gpu.return
  }
}


gpu.module @test_module_34 {
  // CHECK-LABEL: llvm.func @memref_signature(
  //  CHECK-SAME:     %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: f32) -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, f32)>
  //       CHECK:   llvm.mlir.poison
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.mlir.poison
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.insertvalue
  //       CHECK:   llvm.return

  // CHECK-BARE-LABEL: llvm.func @memref_signature(
  //  CHECK-BARE-SAME:     %{{.*}}: !llvm.ptr, %{{.*}}: f32) -> !llvm.struct<(ptr, f32)>
  gpu.func @memref_signature(%m: memref<2xf32>, %f: f32) -> (memref<2xf32>, f32) {
    gpu.return %m, %f : memref<2xf32>, f32
  }
}

gpu.module @test_module_35 {
  // CHECK: llvm.func @__nv_acosf(f32) -> f32
  // CHECK: llvm.func @__nv_acos(f64) -> f64
  // CHECK-LABEL: func @gpu_acos
  func.func @gpu_acos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.acos %arg_f32 : f32
    // CHECK: llvm.call @__nv_acosf(%{{.*}}) : (f32) -> f32
    %result64 = math.acos %arg_f64 : f64
    // CHECK: llvm.call @__nv_acos(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_36 {
  // CHECK: llvm.func @__nv_acoshf(f32) -> f32
  // CHECK: llvm.func @__nv_acosh(f64) -> f64
  // CHECK-LABEL: func @gpu_acosh
  func.func @gpu_acosh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.acosh %arg_f32 : f32
    // CHECK: llvm.call @__nv_acoshf(%{{.*}}) : (f32) -> f32
    %result64 = math.acosh %arg_f64 : f64
    // CHECK: llvm.call @__nv_acosh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_37 {
  // CHECK: llvm.func @__nv_asinf(f32) -> f32
  // CHECK: llvm.func @__nv_asin(f64) -> f64
  // CHECK-LABEL: func @gpu_asin
  func.func @gpu_asin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.asin %arg_f32 : f32
    // CHECK: llvm.call @__nv_asinf(%{{.*}}) : (f32) -> f32
    %result64 = math.asin %arg_f64 : f64
    // CHECK: llvm.call @__nv_asin(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_38 {
  // CHECK: llvm.func @__nv_asinhf(f32) -> f32
  // CHECK: llvm.func @__nv_asinh(f64) -> f64
  // CHECK-LABEL: func @gpu_asinh
  func.func @gpu_asinh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.asinh %arg_f32 : f32
    // CHECK: llvm.call @__nv_asinhf(%{{.*}}) : (f32) -> f32
    %result64 = math.asinh %arg_f64 : f64
    // CHECK: llvm.call @__nv_asinh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_39 {
  // CHECK: llvm.func @__nv_atanhf(f32) -> f32
  // CHECK: llvm.func @__nv_atanh(f64) -> f64
  // CHECK-LABEL: func @gpu_atanh
  func.func @gpu_atanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atanh %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atanh %arg_f32 : f32
    // CHECK: llvm.call @__nv_atanhf(%{{.*}}) : (f32) -> f32
    %result64 = math.atanh %arg_f64 : f64
    // CHECK: llvm.call @__nv_atanh(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

gpu.module @test_module_40 {
  // CHECK: llvm.func @__nv_copysignf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_copysign(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_copysign
  func.func @gpu_copysign(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.copysign %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_copysignf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.copysign %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_copysign(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_41 {
  // CHECK: llvm.func @__nv_coshf(f32) -> f32
  // CHECK: llvm.func @__nv_cosh(f64) -> f64
  // CHECK-LABEL: func @gpu_cosh
  func.func @gpu_cosh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cosh %arg_f32 : f32
    // CHECK: llvm.call @__nv_coshf(%{{.*}}) : (f32) -> f32
    %result64 = math.cosh %arg_f64 : f64
    // CHECK: llvm.call @__nv_cosh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_42 {
  // CHECK: llvm.func @__nv_fmaf(f32, f32, f32) -> f32
  // CHECK: llvm.func @__nv_fma(f64, f64, f64) -> f64
  // CHECK-LABEL: func @gpu_fma
  func.func @gpu_fma(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.fma %arg_f32, %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_fmaf(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
    %result64 = math.fma %arg_f64, %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_fma(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_43 {
  // CHECK: llvm.func @__nv_roundf(f32) -> f32
  // CHECK: llvm.func @__nv_round(f64) -> f64
  // CHECK-LABEL: func @gpu_round
  func.func @gpu_round(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.round %arg_f32 : f32
    // CHECK: llvm.call @__nv_roundf(%{{.*}}) : (f32) -> f32
    %result64 = math.round %arg_f64 : f64
    // CHECK: llvm.call @__nv_round(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_44 {
  // CHECK: llvm.func @__nv_rintf(f32) -> f32
  // CHECK: llvm.func @__nv_rint(f64) -> f64
  // CHECK-LABEL: func @gpu_roundeven
  func.func @gpu_roundeven(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.roundeven %arg_f32 : f32
    // CHECK: llvm.call @__nv_rintf(%{{.*}}) : (f32) -> f32
    %result64 = math.roundeven %arg_f64 : f64
    // CHECK: llvm.call @__nv_rint(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_45 {
  // CHECK: llvm.func @__nv_sinhf(f32) -> f32
  // CHECK: llvm.func @__nv_sinh(f64) -> f64
  // CHECK-LABEL: func @gpu_sinh
  func.func @gpu_sinh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sinh %arg_f32 : f32
    // CHECK: llvm.call @__nv_sinhf(%{{.*}}) : (f32) -> f32
    %result64 = math.sinh %arg_f64 : f64
    // CHECK: llvm.call @__nv_sinh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_46 {
  // CHECK: llvm.func @__nv_coshf(f32) -> f32
  // CHECK: llvm.func @__nv_cosh(f64) -> f64
  // CHECK-LABEL: func @gpu_cosh_with_fastmath
  func.func @gpu_cosh_with_fastmath(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cosh %arg_f32 fastmath<fast> : f32
    // CHECK: llvm.call @__nv_coshf(%{{.*}}) : (f32) -> f32
    %result64 = math.cosh %arg_f64 fastmath<afn> : f64
    // CHECK: llvm.call @__nv_cosh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_47 {
  // CHECK: llvm.func @__nv_sinhf(f32) -> f32
  // CHECK: llvm.func @__nv_sinh(f64) -> f64
  // CHECK-LABEL: func @gpu_sinh_with_fastmath
  func.func @gpu_sinh_with_fastmath(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sinh %arg_f32 fastmath<contract> : f32
    // CHECK: llvm.call @__nv_sinhf(%{{.*}}) : (f32) -> f32
    %result64 = math.sinh %arg_f64 fastmath<none> : f64
    // CHECK: llvm.call @__nv_sinh(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_48 {
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK-LABEL: func @gpu_exp_with_fastmath
  func.func @gpu_exp_with_fastmath(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64, f32) {
    %result32 = math.exp %arg_f32 fastmath<reassoc>: f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 fastmath<contract>: f64
    // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
    %result32Fast = math.exp %arg_f32 fastmath<ninf> : f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    func.return %result32, %result64, %result32Fast : f32, f64, f32
  }
}

gpu.module @test_module_49 {
// CHECK-LABEL: func @explicit_id_bounds()
  func.func @explicit_id_bounds() -> (index, index, index) {
    // CHECK: = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
    %0 = gpu.thread_id x upper_bound 32
    // CHECK: = nvvm.read.ptx.sreg.ntid.x range <i32, 1, 33> : i32
    %1 = gpu.block_dim x upper_bound 32
    // CHECK: = nvvm.read.ptx.sreg.laneid range <i32, 0, 16> : i32
    %2 = gpu.lane_id upper_bound 16

    return %0, %1, %2 : index, index, index
  }
}

gpu.module @test_module_50 {
// CHECK-LABEL: func @kernel_with_grid_size(
  gpu.func @kernel_with_grid_size(%arg0: !llvm.ptr) kernel attributes {known_grid_size = array<i32: 32, 4, 2>} {
    // CHECK: = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 32> : i32
    %0 = gpu.block_id x
    // CHECK: = nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %1 = gpu.block_id y
    // CHECK: = nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 2> : i32
    %2 = gpu.block_id z

    // Fake usage to prevent dead code elimination
    %3 = arith.addi %0, %1 : index
    %4 = arith.addi %3, %2 : index
    %5 = arith.index_cast %4 : index to i64
    llvm.store %5, %arg0 : i64, !llvm.ptr
    gpu.return
  }
}

// CHECK-LABEL: gpu.module @test_module_51
//       CHECK:   llvm.mlir.global internal constant @[[func_name:.*]]("(unknown)\00") {addr_space = 0 : i32}
//       CHECK:   llvm.mlir.global internal constant @[[file_name:.*]]("{{.*}}gpu-to-nvvm.mlir{{.*}}") {addr_space = 0 : i32}
//       CHECK:   llvm.mlir.global internal constant @[[message:.*]]("assert message\00") {addr_space = 0 : i32}
//       CHECK:   llvm.func @__assertfail(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i64) attributes {passthrough = ["noreturn"]}
//       CHECK:   llvm.func @test_assert(%[[cond:.*]]: i1) attributes {gpu.kernel, nvvm.kernel} {
//       CHECK:     llvm.cond_br %[[cond]], ^[[after_block:.*]], ^[[assert_block:.*]]
//       CHECK:   ^[[assert_block]]:
//       CHECK:     %[[message_ptr:.*]] = llvm.mlir.addressof @[[message]] : !llvm.ptr
//       CHECK:     %[[message_start:.*]] = llvm.getelementptr %[[message_ptr]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
//       CHECK:     %[[file_ptr:.*]] = llvm.mlir.addressof @[[file_name]] : !llvm.ptr
//       CHECK:     %[[file_start:.*]] = llvm.getelementptr %[[file_ptr]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<{{.*}} x i8>
//       CHECK:     %[[func_ptr:.*]] = llvm.mlir.addressof @[[func_name]] : !llvm.ptr
//       CHECK:     %[[func_start:.*]] = llvm.getelementptr %[[func_ptr]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<{{.*}} x i8>
//       CHECK:     %[[line_num:.*]] = llvm.mlir.constant({{.*}} : i32) : i32
//       CHECK:     %[[ptr:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:     llvm.call @__assertfail(%[[message_start]], %[[file_start]], %[[line_num]], %[[func_start]], %[[ptr]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i64) -> ()
//       CHECK:     llvm.br ^[[after_block]]
//       CHECK:   ^[[after_block]]:
//       CHECK:     llvm.return
//       CHECK:   }

gpu.module @test_module_51 {
  gpu.func @test_assert(%arg0: i1) kernel {
    cf.assert %arg0, "assert message"
    gpu.return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%toplevel_module: !transform.any_op {transform.readonly}) {
    %gpu_module = transform.structured.match ops{["gpu.module"]} in %toplevel_module
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %gpu_module {
      transform.apply_patterns.gpu.gpu_rewrite_patterns
    } : !transform.any_op

    transform.apply_conversion_patterns to %gpu_module {
      transform.apply_conversion_patterns.dialect_to_llvm "arith"
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
      transform.apply_conversion_patterns.vector.vector_to_llvm
      transform.apply_conversion_patterns.func.func_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "memref"
      transform.apply_conversion_patterns.gpu.gpu_to_nvvm {benefit = 10 : i16}
      transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm
      transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm
      transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        {index_bitwidth = 64,
        use_bare_ptr_call_conv = false}
    } {
      legal_dialects = ["llvm", "memref", "nvvm", "test"],
      legal_ops = ["func.func", "gpu.module", "gpu.yield"],
      illegal_dialects = ["gpu"],
      illegal_ops = ["llvm.copysign", "llvm.cos", "llvm.exp", "llvm.exp2", "llvm.fabs", "llvm.fceil",
                    "llvm.ffloor", "llvm.frem", "llvm.log", "llvm.log10", "llvm.log2", "llvm.pow",
                    "llvm.roundeven", "llvm.round", "llvm.sin", "llvm.sqrt"],
      partial_conversion
    } : !transform.any_op
    transform.yield
  }
}


gpu.module @test_module_52 {
  // CHECK: llvm.func @__nv_abs(i32) -> i32
  // CHECK-LABEL: func @gpu_abs
  func.func @gpu_abs(%arg_i32 : i32) -> (i32) {
    %result32 = math.absi %arg_i32 : i32
    // CHECK: llvm.call @__nv_abs(%{{.*}}) : (i32) -> i32
    func.return %result32 : i32
  }
}

gpu.module @test_module_53 {
  // CHECK: llvm.func @__nv_powif(f32, i32) -> f32
  // CHECK: llvm.func @__nv_powi(f64, i32) -> f64
  // CHECK-LABEL: func @gpu_powi
  func.func @gpu_powi(%arg_f32 : f32, %arg_f64 : f64, %arg_i32 : i32) -> (f32, f64) {
    %result32 = math.fpowi %arg_f32, %arg_i32 : f32, i32
    // CHECK: llvm.call @__nv_powif(%{{.*}}, %{{.*}}) : (f32, i32) -> f32
    %result64 = math.fpowi %arg_f64, %arg_i32 : f64, i32
    // CHECK: llvm.call @__nv_powi(%{{.*}}, %{{.*}}) : (f64, i32) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

gpu.module @test_module_54 {
  // CHECK: llvm.func @__nv_isinff(f32) -> i32
  // CHECK: llvm.func @__nv_isinfd(f64) -> i32
  // CHECK: llvm.func @__nv_isnanf(f32) -> i32
  // CHECK: llvm.func @__nv_isnand(f64) -> i32
  // CHECK: llvm.func @__nv_finitef(f32) -> i32
  // CHECK: llvm.func @__nv_isfinited(f64) -> i32
  // CHECK-LABEL: @fpclassify
  func.func @fpclassify(%f32: f32, %f64: f64) -> (i1, i1, i1, i1, i1, i1) {
    // CHECK: %[[INFF:.+]] = llvm.call @__nv_isinff(%{{.*}}) : (f32) -> i32
    // CHECK: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[R0:.+]] = llvm.icmp "ne" %[[INFF]], %[[ZERO]]
    %0 = math.isinf %f32 : f32
    // CHECK: llvm.call @__nv_isinfd(%{{.*}}) : (f64) -> i32
    // CHECK: llvm.mlir.constant(0
    // CHECK: llvm.icmp "ne"
    %1 = math.isinf %f64 : f64
    // CHECK: llvm.call @__nv_isnanf(%{{.*}}) : (f32) -> i32
    // CHECK: llvm.mlir.constant(0
    // CHECK: llvm.icmp "ne"
    %2 = math.isnan %f32 : f32
    // CHECK: llvm.call @__nv_isnand(%{{.*}}) : (f64) -> i32
    // CHECK: llvm.mlir.constant(0
    // CHECK: llvm.icmp "ne"
    %3 = math.isnan %f64 : f64
    // CHECK: llvm.call @__nv_finitef(%{{.*}}) : (f32) -> i32
    // CHECK: llvm.mlir.constant(0
    // CHECK: llvm.icmp "ne"
    %4 = math.isfinite %f32 : f32
    // CHECK: llvm.call @__nv_isfinited(%{{.*}}) : (f64) -> i32
    // CHECK: llvm.mlir.constant(0
    // CHECK: llvm.icmp "ne"
    %5 = math.isfinite %f64 : f64
    // CHECK: llvm.return %[[R0]]
    return %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
  }
}

gpu.module @test_module_55 {
  // CHECK: llvm.func @__nv_erfcf(f32) -> f32
  // CHECK: llvm.func @__nv_erfc(f64) -> f64
  // CHECK-LABEL: func @gpu_erf
  func.func @gpu_erfc(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.erfc %arg_f32 : f32
    // CHECK: llvm.call @__nv_erfcf(%{{.*}}) : (f32) -> f32
    %result64 = math.erfc %arg_f64 : f64
    // CHECK: llvm.call @__nv_erfc(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}
