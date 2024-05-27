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
        %cidx_0 = arith.constant 0 : index

        // Table and table sizes have to be passed as arguments later on
        %relation1_rows = arith.constant 5 : index
        %relation2_rows = arith.constant 5 : index

        // Allocate and initialize the memrefs for keys of both relations
        %relation1 = memref.alloc(%relation1_rows) : memref<?xi32>
        call @init_relation_index(%relation1) : (memref<?xi32>) -> ()

        %relation2 = memref.alloc(%relation2_rows) : memref<?xi32>
        call @init_relation_index(%relation2) : (memref<?xi32>) -> ()
        
        
        %success = call @check(%relation1, %relation2, %relation1,%relation2)
        : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32
        // print success
        func.call @debugI32(%success) : (i32) -> ()

        // %dst = memref.cast %d : memref<?xi32> to memref<*xi32>
        // call @printMemrefI32(%dst) : (memref<*xi32>) -> ()
        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefI64(memref<*xi64>)
    func.func private @check(memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32
    func.func private @init_relation(memref<?xi32>)
    func.func private @init_relation_index(memref<?xi32>)
}