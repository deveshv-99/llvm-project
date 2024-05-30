module attributes {gpu.container_module} {

    gpu.module @kernels {

        gpu.func @add(%arr1: memref<?xi32>, %arr2: memref<?xi32>) 
        kernel
        {
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            %x = memref.load %arr1[%tidx] : memref<?xi32>  
            %y = memref.load %arr2[%tidx] : memref<?xi32>

            %z = arith.addi %x, %y : i32
            memref.store %z, %arr1[%tidx] : memref<?xi32> 

            gpu.return 
        }

        
    }

    gpu.module @kernel1{
        gpu.func @sub(%arr1: memref<?xi32>, %arr2: memref<?xi32>)
            kernel
        {
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            %x = memref.load %arr1[%tidx] : memref<?xi32>  
            %y = memref.load %arr2[%tidx] : memref<?xi32>

            %z = arith.subi %x, %y : i32
            memref.store %z, %arr1[%tidx] : memref<?xi32>

            gpu.return
        }
    }

    func.func @main(){
        
        %c0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %num_blocks = arith.constant 1 : index

        %size = arith.constant 10 : index

        %val1 = arith.constant 1 : i32
        %val2 = arith.constant 2 : i32
        
        %arr1_ = memref.alloc(%size) : memref<?xi32>
        %arr2_ = memref.alloc(%size) : memref<?xi32>

        scf.for %i = %c0 to %size step %cidx_1{
            
            memref.store %val1, %arr1_[%i] : memref<?xi32>
            memref.store %val2, %arr2_[%i] : memref<?xi32>
        }

        // perfom gpu addition of 2 10 element vectors

        %arr1 = gpu.alloc(%size) : memref<?xi32>
        %arr2 = gpu.alloc(%size) : memref<?xi32>

        gpu.memcpy %arr1, %arr1_ : memref<?xi32>, memref<?xi32>
        gpu.memcpy %arr2, %arr2_ : memref<?xi32>, memref<?xi32>

        gpu.launch_func @kernels::@add
        blocks in (%num_blocks, %cidx_1, %cidx_1)
        threads in (%size, %cidx_1, %cidx_1)
        args(%arr1 : memref<?xi32> , %arr2 : memref<?xi32>)  

        gpu.launch_func @kernel1::@sub
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%size, %cidx_1, %cidx_1)
        args(%arr1: memref<?xi32>, %arr2: memref<?xi32>)

        gpu.memcpy %arr1_, %arr1 : memref<?xi32>, memref<?xi32>

        %dst = memref.cast %arr1_ : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()


        
        return
    }
    func.func private @printMemrefI32(memref<*xi32>)
}