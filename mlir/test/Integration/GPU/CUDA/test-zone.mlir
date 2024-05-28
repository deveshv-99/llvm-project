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
    
    func.func @debugI64(%val: i64) {
        %A = memref.alloc() : memref<i64>
        memref.store %val, %A[]: memref<i64>
        %U = memref.cast %A :  memref<i64> to memref<*xi64>

        func.call @printMemrefI64(%U): (memref<*xi64>) -> ()

        memref.dealloc %A : memref<i64>
        return
    }

    func.func @main() {

        %c0 = arith.constant 4 : i32
        %c1 = arith.constant 1 : i32
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_2 = arith.constant 2 : index

        // Table and table sizes have to be passed as arguments later on
        %relation1_rows = arith.constant 5 : index
        %relation2_rows = arith.constant 5 : index

        %relation1 = memref.alloc(%relation1_rows) : memref<?xindex>
        memref.store %cidx_2, %relation1[%cidx_0] : memref<?xindex>

        %x_ = arith.constant 0 : i32
        %y_ = arith.constant 10 : i32

        // %x_ = memref.atomic_rmw "assign" %cidx_2, %relation1[%cidx_0] : (index, memref<?xindex>) -> index
        // %y_ = memref.load %relation1[%cidx_0] : memref<?xindex>
        // print success
        // cast to integer
        %x = arith.index_cast %x_ : i32 to index        
        %y = arith.index_cast %y_ : i32 to index

        %z_ = memref.load %relation1[%cidx_0] : memref<?xindex>
        %z = arith.index_cast %z_ : index to i32
        func.call @debugI32(%z) : (i32) -> ()
        //func.call @debugI32(%x) : (i32) -> ()
        //func.call @debugI32(%y) : (i32) -> ()
        // %dst = memref.cast %d : memref<?xi32> to memref<*xi32>
        // call @printMemrefI32(%dst) : (memref<*xi32>) -> ()
        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefI64(memref<*xi64>)

}