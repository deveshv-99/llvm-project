// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    func.func @main() {

        %c0 = arith.constant 4 : i32
        %cidx_0 = arith.constant 0 : index
        %c = memref.alloc() : memref<3xi32>
        memref.store %c0, %c[%cidx_0] : memref<3xi32>

        %d = memref.cast %c : memref<3xi32> to memref<?xi32>

        call @init_relation(%d) : (memref<?xi32>) -> ()

        %dst = memref.cast %d : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()
        return
    }

    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefI64(memref<*xi64>)
    func.func private @init_relation(memref<?xi32>)
}