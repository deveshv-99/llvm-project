// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    func.func @debugI32(%val: i32) {
        %A = memref.alloc() : memref<i32>
        memref.store %val, %A[]: memref<i32>
        %U = memref.cast %A :  memref<i32> to memref<*xi32>

        func.call @printMemrefI32(%U): (memref<*xi32>) -> ()

        memref.dealloc %A : memref<i32>
        return
    }
    
    func.func @main() {

        // Constants

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32
        %ci32_32 = arith.constant 32 : i32
        %ci32_1024 = arith.constant 1024 : i32

        %val = llvm.mlir.constant (43 : i32) : i32
        %val1 = llvm.mlir.constant (44 : i32) : i32
        %c0 = llvm.mlir.constant (0 : i32) : i32
        %c1 = llvm.mlir.constant (1 : i32) : i32


        // use alloca to create array of 2 elements
        %a2 = llvm.alloca %ci32_2 x i32 : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr1 = llvm.getelementptr %a2[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)
        %ptr2 = llvm.getelementptr %a2[%c1] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)

        //insert value at this pointer
        llvm.store %val, %ptr1 : !llvm.ptr< i32>
        llvm.store %val1, %ptr2 : !llvm.ptr< i32>

        %value1 = llvm.load %ptr1 : !llvm.ptr<i32>
        %value2 = llvm.load %ptr2 : !llvm.ptr<i32>

        call @debugI32(%value1) : (i32) -> ()
        call @debugI32(%value2) : (i32) -> ()

        // Create a 2D array
        %x = llvm.alloca %ci32_1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr = llvm.getelementptr %x[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<array<2 x i32>>)

        //insert value at this pointer
        %ptptr = llvm.getelementptr %x[%c0, %c0] : (!llvm.ptr<array<2 x i32>> ,i32, i32) -> (!llvm.ptr<i32>)
        llvm.store %val, %ptptr : !llvm.ptr< i32>

        %valuee = llvm.load %ptptr : !llvm.ptr<i32>
        call @debugI32(%valuee) : (i32) -> ()


        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
}