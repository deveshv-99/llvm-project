// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    func.func @identity(%arg0: index) -> index {
        return %arg0 : index
    }


    gpu.module @kernels {
        gpu.func @add_arrays (%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel {
            %idx = gpu.thread_id x

            %val0 = memref.load %arg0[%idx] : memref<?xf32>
            %val1 = memref.load %arg1[%idx] : memref<?xf32>

            %result = arith.addf %val0, %val1 : f32
            memref.store %result, %arg2[%idx] : memref<?xf32>
            gpu.printf "Thread ID: %lld \t Result: %f\n" %idx, %result  : index, f32

            gpu.return
        }
    }

    
    func.func @main() {

        %lo_size = arith.constant 4 : index
        %p_size = arith.constant 4 : index

        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_2 = arith.constant 2 : index
        %cidx_3 = arith.constant 3 : index
        
        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32

        //Whichever table is smaller, we will use that for comparison (as the outer loop)
        %lo_or_p_as_outer = arith.cmpi "ult", %lo_size, %p_size : index
        %size = arith.constant 0 : index
        scf.if %lo_or_p_as_outer {
            %size = func.call @identity(%p_size) : (index) -> (index)
            // part table is 0, line_order table is 1
            %which_table = arith.constant 0 : index
        } else {
            %size = func.call @identity(%lo_size) : (index) -> (index)
            // part table is 0, line_order table is 1
            %which_table = arith.constant 1 : index
        }

    
        %num_blocks = arith.constant 1 : index
        %num_threads = func.call @identity(%size) : (index) -> (index)















        %arg0 = memref.alloc(%size) : memref<?xf32>
        %arg1 = memref.alloc(%size) : memref<?xf32>
        %arg2 = memref.alloc(%size) : memref<?xf32>

        affine.for %i = 0 to %size {
            memref.store %constant_1, %arg0[%i] : memref<?xf32>
            memref.store %constant_2, %arg1[%i] : memref<?xf32>
        }

        %gpu_arg0 = gpu.alloc(%size) : memref<?xf32>
        %gpu_arg1 = gpu.alloc(%size) : memref<?xf32>
        %gpu_arg2 = gpu.alloc(%size) : memref<?xf32>

        gpu.memcpy %gpu_arg0, %arg0 : memref<?xf32>, memref<?xf32>
        gpu.memcpy %gpu_arg1, %arg1 : memref<?xf32>, memref<?xf32>

        
        
        gpu.launch_func @kernels::@add_arrays
            blocks in (%c1, %c1, %c1) 
            threads in (%size, %c1, %c1)
            args(%gpu_arg0 : memref<?xf32> , %gpu_arg1 : memref<?xf32>, %gpu_arg2 : memref<?xf32>)

        gpu.memcpy %arg2, %gpu_arg2 : memref<?xf32>, memref<?xf32>
        %printval = memref.cast %arg2 : memref<?xf32> to memref<*xf32>

        call @printMemrefF32(%printval) : (memref<*xf32>) -> ()
        //CHECK: [3, 3, 3, 3]
        return
    }
    func.func private @printMemrefF32(memref<*xf32>)
}

