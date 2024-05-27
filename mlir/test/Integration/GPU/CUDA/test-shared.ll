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
    %0 = llvm.mlir.constant(5 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.null : !llvm.ptr
    %3 = llvm.getelementptr %2[5] : (!llvm.ptr) -> !llvm.ptr, i32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.insertvalue %9, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %0, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %1, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @init_relation_index(%5, %5, %9, %0, %1) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.null : !llvm.ptr
    %15 = llvm.getelementptr %14[5] : (!llvm.ptr) -> !llvm.ptr, i32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %0, %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %13, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @init_relation_index(%17, %17, %21, %0, %13) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    %25 = llvm.call @check(%5, %5, %9, %0, %1, %17, %17, %21, %0, %13, %5, %5, %9, %0, %1, %17, %17, %21, %0, %13) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    llvm.call @debugI32(%25) : (i32) -> ()
    llvm.return
  }
  llvm.func @printMemrefI32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefI64(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @check(!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32 attributes {sym_visibility = "private"}
  llvm.func @init_relation(!llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @init_relation_index(!llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
}

