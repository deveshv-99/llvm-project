// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    // Function to print the contents of a single memref value
    func.func @debugI32(%val: i32) {
        %A = memref.alloc() : memref<i32>
        memref.store %val, %A[]: memref<i32>
        %U = memref.cast %A :  memref<i32> to memref<*xi32>

        func.call @printMemrefI32(%U): (memref<*xi32>) -> ()

        memref.dealloc %A : memref<i32>
        return
    }

    // To initialize the tables with fixed values
    func.func @init(%relation : memref<?xi32>, %rows: index) -> memref<?xi32> {

        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index

        scf.for %i = %cidx_0 to %rows step %cidx_1 {

                %i_i32 = arith.index_cast %i : index to i32
                memref.store %i_i32, %relation[%i] : memref<?xi32>
        }
        return %relation: memref<?xi32>
    }
    


    func.func @main() {

        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32
        
        %ci32_43 = arith.constant 43 : i32

        // Table and table sizes have to be passed as arguments later on
        %relation1_rows = arith.constant 5 : index
        %relation2_rows = arith.constant 5 : index

        // Allocate and initialize the memrefs for keys of both relations
        %relation1_ = memref.alloc(%relation1_rows) : memref<?xi32>
        %relation1 = call @init(%relation1_, %relation1_rows) : (memref<?xi32>, index) -> memref<?xi32>

        %relation2_ = memref.alloc(%relation2_rows) : memref<?xi32>
        // store 43 to the first element of the second relation
        %relation2 = call @init(%relation2_, %relation2_rows) : (memref<?xi32>, index) -> memref<?xi32>
        memref.store %ci32_43, %relation2_[%cidx_0] : memref<?xi32>


        // print the result
        %dst = memref.cast %relation2 : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
}