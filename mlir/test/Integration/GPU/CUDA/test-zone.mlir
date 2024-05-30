module attributes {gpu.container_module} {

    gpu.module @kernels {

        gpu.func @init_ht(%ht_size : index, %ht_ptr : memref<?xi32>) 
            kernel
        {
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            // Global_thread_index = bdim * bidx + tidx
            %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
            %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

            // Check if the thread is valid
            %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %ht_size : index

            scf.if %is_thread_valid {

                // Set the ht_ptr to -1
                %ci32_neg1 = arith.constant -1 : i32
                memref.store %ci32_neg1, %ht_ptr[%g_thread_idx] : memref<?xi32>
    
            }
            gpu.return 
        }

        gpu.func @build(%relation1 : memref<?xi32>, %relation1_rows : index,
                %ht_ptr : memref<?xi32>, %ll_key : memref<?xi32>, %ll_rowID : memref<?xindex>, 
                %ll_next : memref<?xindex>, %free_index : memref<1xi32>)
            kernel
        {
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            // Global_thread_index = bdim * bidx + tidx
            %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
            %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

            // Check if the thread is valid
            %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %relation1_rows : index

            scf.if %is_thread_valid {
                
                //constants
                %cidx_0 = arith.constant 0 : index
                %cidx_1 = arith.constant 1 : index
                %cidx_2 = arith.constant 2 : index

                %key = memref.load %relation1[%g_thread_idx] : memref<?xi32>
                
                // func.call @Insert_Node_HT(%key, %ht_ptr, %ll_key, %ll_rowID, %ll_next, %free_index, %g_thread_idx) 
                //  : (i32, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index) -> ()


            }
            
            gpu.return
        }
    }

    func.func @main(){

        %cidx_1 = arith.constant 1 : index
        %num_blocks = arith.constant 1 : index
        %num_threads_per_block = arith.constant 1 : index

        %ht_size = arith.constant 10 : index
        %ht_ptr = gpu.alloc(%cidx_1) : memref<?xi32>


        gpu.launch_func @kernels::@init_ht
        blocks in (%num_blocks, %cidx_1, %cidx_1)
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%ht_size : index, %ht_ptr : memref<?xi32>)


        %relation1 = gpu.alloc(%cidx_1) : memref<?xi32>
        %relation1_rows = arith.constant 10 : index

        %ll_key = gpu.alloc(%cidx_1) : memref<?xi32>
        %ll_rowID = gpu.alloc(%cidx_1) : memref<?xindex>
        %ll_next = gpu.alloc(%cidx_1) : memref<?xindex>

        %free_index = gpu.alloc() : memref<1xi32>


        gpu.launch_func @kernels::@build
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%relation1 : memref<?xi32>, %relation1_rows : index, %ht_ptr : memref<?xi32>, %ll_key : memref<?xi32>,
            %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>, %free_index : memref<1xi32>)
        return
    }
}