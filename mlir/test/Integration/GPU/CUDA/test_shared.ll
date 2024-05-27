module attributes {gpu.container_module, llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @debugI32(%arg0: i32) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.null : !llvm.ptr
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %3 = llvm.ptrtoint %2 : !llvm.ptr to i64
    %4 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.insertvalue %8, %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.alloca %10 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %9, %11 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(i64, ptr)> 
    %15 = llvm.insertvalue %11, %14[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefI32(%12, %11) : (i64, !llvm.ptr) -> ()
    llvm.call @free(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @debugI64(%arg0: i64) {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.null : !llvm.ptr
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %3 = llvm.ptrtoint %2 : !llvm.ptr to i64
    %4 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.insertvalue %8, %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.alloca %10 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %9, %11 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(i64, ptr)> 
    %15 = llvm.insertvalue %11, %14[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefI64(%12, %11) : (i64, !llvm.ptr) -> ()
    llvm.call @free(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.null : !llvm.ptr
    %5 = llvm.getelementptr %4[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %2, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %3, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.store %0, %7 : i32, !llvm.ptr
    %15 = llvm.call @check(%7, %7, %11, %2, %3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    llvm.call @debugI32(%15) : (i32) -> ()
    llvm.return
  }
  llvm.func @printMemrefI32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefI64(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @check(!llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32 attributes {sym_visibility = "private"}
  llvm.func @init_relation(!llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
}

