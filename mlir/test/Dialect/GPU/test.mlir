module {
  gpu.module @kernels {
    gpu.func @kernel(%arg0: f32) workgroup(%arg1 : memref<32xf32, #gpu.address_space<workgroup>>) kernel {
      %c31_i32 = arith.constant 31 : i32
      %c0_i32 = arith.constant 0 : i32
      %c32_i32 = arith.constant 32 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c16_i32 = arith.constant 16 : i32
      %0 = gpu.block_dim  x
      %1 = arith.index_cast %0 : index to i32
      %2 = gpu.block_dim  y
      %3 = arith.index_cast %2 : index to i32
      %4 = gpu.block_dim  z
      %5 = arith.index_cast %4 : index to i32
      %6 = gpu.thread_id  x
      %7 = arith.index_cast %6 : index to i32
      %8 = gpu.thread_id  y
      %9 = arith.index_cast %8 : index to i32
      %10 = gpu.thread_id  z
      %11 = arith.index_cast %10 : index to i32
      %12 = arith.muli %11, %3 : i32
      %13 = arith.addi %12, %9 : i32
      %14 = arith.muli %13, %1 : i32
      %15 = arith.muli %1, %3 : i32
      %16 = arith.addi %14, %7 : i32
      %17 = arith.muli %15, %5 : i32
      %18 = arith.andi %16, %c31_i32 : i32
      %19 = arith.cmpi eq, %18, %c0_i32 : i32
      %20 = arith.subi %16, %18 : i32
      %21 = arith.subi %17, %20 : i32
      %22 = arith.cmpi slt, %21, %c32_i32 : i32
      cf.cond_br %22, ^bb1, ^bb17
    ^bb1:  // pred: ^bb0
      %shuffleResult, %valid = gpu.shuffle  xor %arg0, %c1_i32, %21 : f32
      cf.cond_br %valid, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %23 = arith.cmpf ugt, %arg0, %shuffleResult : f32
      %24 = arith.select %23, %arg0, %shuffleResult : f32
      cf.br ^bb4(%24 : f32)
    ^bb3:  // pred: ^bb1
      cf.br ^bb4(%arg0 : f32)
    ^bb4(%25: f32):  // 2 preds: ^bb2, ^bb3
      %shuffleResult_0, %valid_1 = gpu.shuffle  xor %25, %c2_i32, %21 : f32
      cf.cond_br %valid_1, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
      %26 = arith.cmpf ugt, %25, %shuffleResult_0 : f32
      %27 = arith.select %26, %25, %shuffleResult_0 : f32
      cf.br ^bb7(%27 : f32)
    ^bb6:  // pred: ^bb4
      cf.br ^bb7(%25 : f32)
    ^bb7(%28: f32):  // 2 preds: ^bb5, ^bb6
      %shuffleResult_2, %valid_3 = gpu.shuffle  xor %28, %c4_i32, %21 : f32
      cf.cond_br %valid_3, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %29 = arith.cmpf ugt, %28, %shuffleResult_2 : f32
      %30 = arith.select %29, %28, %shuffleResult_2 : f32
      cf.br ^bb10(%30 : f32)
    ^bb9:  // pred: ^bb7
      cf.br ^bb10(%28 : f32)
    ^bb10(%31: f32):  // 2 preds: ^bb8, ^bb9
      %shuffleResult_4, %valid_5 = gpu.shuffle  xor %31, %c8_i32, %21 : f32
      cf.cond_br %valid_5, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %32 = arith.cmpf ugt, %31, %shuffleResult_4 : f32
      %33 = arith.select %32, %31, %shuffleResult_4 : f32
      cf.br ^bb13(%33 : f32)
    ^bb12:  // pred: ^bb10
      cf.br ^bb13(%31 : f32)
    ^bb13(%34: f32):  // 2 preds: ^bb11, ^bb12
      %shuffleResult_6, %valid_7 = gpu.shuffle  xor %34, %c16_i32, %21 : f32
      cf.cond_br %valid_7, ^bb14, ^bb15
    ^bb14:  // pred: ^bb13
      %35 = arith.cmpf ugt, %34, %shuffleResult_6 : f32
      %36 = arith.select %35, %34, %shuffleResult_6 : f32
      cf.br ^bb16(%36 : f32)
    ^bb15:  // pred: ^bb13
      cf.br ^bb16(%34 : f32)
    ^bb16(%37: f32):  // 2 preds: ^bb14, ^bb15
      cf.br ^bb18(%37 : f32)
    ^bb17:  // pred: ^bb0
      %shuffleResult_8, %valid_9 = gpu.shuffle  xor %arg0, %c1_i32, %c32_i32 : f32
      %38 = arith.cmpf ugt, %arg0, %shuffleResult_8 : f32
      %39 = arith.select %38, %arg0, %shuffleResult_8 : f32
      %shuffleResult_10, %valid_11 = gpu.shuffle  xor %39, %c2_i32, %c32_i32 : f32
      %40 = arith.cmpf ugt, %39, %shuffleResult_10 : f32
      %41 = arith.select %40, %39, %shuffleResult_10 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle  xor %41, %c4_i32, %c32_i32 : f32
      %42 = arith.cmpf ugt, %41, %shuffleResult_12 : f32
      %43 = arith.select %42, %41, %shuffleResult_12 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle  xor %43, %c8_i32, %c32_i32 : f32
      %44 = arith.cmpf ugt, %43, %shuffleResult_14 : f32
      %45 = arith.select %44, %43, %shuffleResult_14 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle  xor %45, %c16_i32, %c32_i32 : f32
      %46 = arith.cmpf ugt, %45, %shuffleResult_16 : f32
      %47 = arith.select %46, %45, %shuffleResult_16 : f32
      cf.br ^bb18(%47 : f32)
    ^bb18(%48: f32):  // 2 preds: ^bb16, ^bb17
      cf.cond_br %19, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      %49 = arith.divsi %16, %c32_i32 : i32
      %50 = arith.index_cast %49 : i32 to index
      memref.store %48, %arg1[%50] : memref<32xf32, #gpu.address_space<workgroup>>
      cf.br ^bb21
    ^bb20:  // pred: ^bb18
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      gpu.barrier
      %51 = arith.addi %17, %c31_i32 : i32
      %52 = arith.divsi %51, %c32_i32 : i32
      %53 = arith.cmpi slt, %16, %52 : i32
      cf.cond_br %53, ^bb22, ^bb41
    ^bb22:  // pred: ^bb21
      %54 = arith.index_cast %16 : i32 to index
      %55 = memref.load %arg1[%54] : memref<32xf32, #gpu.address_space<workgroup>>
      %56 = arith.cmpi slt, %52, %c32_i32 : i32
      cf.cond_br %56, ^bb23, ^bb39
    ^bb23:  // pred: ^bb22
      %shuffleResult_18, %valid_19 = gpu.shuffle  xor %55, %c1_i32, %52 : f32
      cf.cond_br %valid_19, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %57 = arith.cmpf ugt, %55, %shuffleResult_18 : f32
      %58 = arith.select %57, %55, %shuffleResult_18 : f32
      cf.br ^bb26(%58 : f32)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26(%55 : f32)
    ^bb26(%59: f32):  // 2 preds: ^bb24, ^bb25
      %shuffleResult_20, %valid_21 = gpu.shuffle  xor %59, %c2_i32, %52 : f32
      cf.cond_br %valid_21, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %60 = arith.cmpf ugt, %59, %shuffleResult_20 : f32
      %61 = arith.select %60, %59, %shuffleResult_20 : f32
      cf.br ^bb29(%61 : f32)
    ^bb28:  // pred: ^bb26
      cf.br ^bb29(%59 : f32)
    ^bb29(%62: f32):  // 2 preds: ^bb27, ^bb28
      %shuffleResult_22, %valid_23 = gpu.shuffle  xor %62, %c4_i32, %52 : f32
      cf.cond_br %valid_23, ^bb30, ^bb31
    ^bb30:  // pred: ^bb29
      %63 = arith.cmpf ugt, %62, %shuffleResult_22 : f32
      %64 = arith.select %63, %62, %shuffleResult_22 : f32
      cf.br ^bb32(%64 : f32)
    ^bb31:  // pred: ^bb29
      cf.br ^bb32(%62 : f32)
    ^bb32(%65: f32):  // 2 preds: ^bb30, ^bb31
      %shuffleResult_24, %valid_25 = gpu.shuffle  xor %65, %c8_i32, %52 : f32
      cf.cond_br %valid_25, ^bb33, ^bb34
    ^bb33:  // pred: ^bb32
      %66 = arith.cmpf ugt, %65, %shuffleResult_24 : f32
      %67 = arith.select %66, %65, %shuffleResult_24 : f32
      cf.br ^bb35(%67 : f32)
    ^bb34:  // pred: ^bb32
      cf.br ^bb35(%65 : f32)
    ^bb35(%68: f32):  // 2 preds: ^bb33, ^bb34
      %shuffleResult_26, %valid_27 = gpu.shuffle  xor %68, %c16_i32, %52 : f32
      cf.cond_br %valid_27, ^bb36, ^bb37
    ^bb36:  // pred: ^bb35
      %69 = arith.cmpf ugt, %68, %shuffleResult_26 : f32
      %70 = arith.select %69, %68, %shuffleResult_26 : f32
      cf.br ^bb38(%70 : f32)
    ^bb37:  // pred: ^bb35
      cf.br ^bb38(%68 : f32)
    ^bb38(%71: f32):  // 2 preds: ^bb36, ^bb37
      cf.br ^bb40(%71 : f32)
    ^bb39:  // pred: ^bb22
      %shuffleResult_28, %valid_29 = gpu.shuffle  xor %55, %c1_i32, %c32_i32 : f32
      %72 = arith.cmpf ugt, %55, %shuffleResult_28 : f32
      %73 = arith.select %72, %55, %shuffleResult_28 : f32
      %shuffleResult_30, %valid_31 = gpu.shuffle  xor %73, %c2_i32, %c32_i32 : f32
      %74 = arith.cmpf ugt, %73, %shuffleResult_30 : f32
      %75 = arith.select %74, %73, %shuffleResult_30 : f32
      %shuffleResult_32, %valid_33 = gpu.shuffle  xor %75, %c4_i32, %c32_i32 : f32
      %76 = arith.cmpf ugt, %75, %shuffleResult_32 : f32
      %77 = arith.select %76, %75, %shuffleResult_32 : f32
      %shuffleResult_34, %valid_35 = gpu.shuffle  xor %77, %c8_i32, %c32_i32 : f32
      %78 = arith.cmpf ugt, %77, %shuffleResult_34 : f32
      %79 = arith.select %78, %77, %shuffleResult_34 : f32
      %shuffleResult_36, %valid_37 = gpu.shuffle  xor %79, %c16_i32, %c32_i32 : f32
      %80 = arith.cmpf ugt, %79, %shuffleResult_36 : f32
      %81 = arith.select %80, %79, %shuffleResult_36 : f32
      cf.br ^bb40(%81 : f32)
    ^bb40(%82: f32):  // 2 preds: ^bb38, ^bb39
      memref.store %82, %arg1[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
      cf.br ^bb42
    ^bb41:  // pred: ^bb21
      cf.br ^bb42
    ^bb42:  // 2 preds: ^bb40, ^bb41
      gpu.barrier
      gpu.return
    }
  }
}

