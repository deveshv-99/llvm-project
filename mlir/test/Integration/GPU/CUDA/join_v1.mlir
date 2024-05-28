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

    // To calculate the number of blocks needed, perform ceil division:
    // num_blocks = (total_threads + num_threads_per_block - 1) / num_threads_per_block
    
    func.func @alloc_hash_table(%num_tuples : index, %ht_size : index) 
        -> ( memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>) {
            
        // Allocate linked_list 
        %ll_key = gpu.alloc(%num_tuples) : memref<?xi32>
        %ll_rowID = gpu.alloc(%num_tuples) : memref<?xindex>
        %ll_next = gpu.alloc(%num_tuples) : memref<?xindex>

        // Allocate hash table
        %ht_ptr = gpu.alloc(%ht_size) : memref<?xi32>
        
        // Return all the allocated memory
        return %ll_key, %ll_rowID, %ll_next, %ht_ptr
            : memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>
    }

    func.func @calc_num_blocks(%total_threads : index, %num_threads_per_block: index) -> index {
        
        // Constants
        %cidx_1 = arith.constant 1 : index

        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index
        return %num_blocks : index
    }

    func.func @init_hash_table(%ht_size : index, %ht_ptr : memref<?xi32>) {
            
        // Constants
        %cidx_1 = arith.constant 1 : index

        // //-------------> Keep threads per block constant at 1024.. Need to change this?
        %num_threads_per_block = arith.constant 1024 : index

        %num_blocks = func.call @calc_num_blocks(%ht_size, %num_threads_per_block) : (index, index) -> index
        
        gpu.launch_func @kernels::@init_ht
        blocks in (%num_blocks, %cidx_1, %cidx_1)
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%ht_size : index, %ht_ptr : memref<?xi32>)

        return
    }

    func.func @build_table(%relation1 : memref<?xi32>, %relation1_rows : index, %ht_ptr : memref<?xi32>,
        %ll_key : memref<?xi32>, %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>) {
        
        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %ci32_0 = arith.constant 0 : i32

        //global variable for next free index in linked_list
        %h_free_index = memref.alloc() : memref<1xi32>
        %free_index = gpu.alloc() : memref<1xi32>
        

        //store 0 initially
        memref.store %ci32_0, %h_free_index[%cidx_0] : memref<1xi32>
        gpu.memcpy %free_index, %h_free_index : memref<1xi32>, memref<1xi32>

        // //-------------> Keep threads per block constant at 1024.. Need to change this?
        %num_threads_per_block = arith.constant 1024 : index

        %num_blocks = func.call @calc_num_blocks(%relation1_rows, %num_threads_per_block) : (index, index) -> index

        gpu.launch_func @kernels::@build
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%relation1 : memref<?xi32>, %relation1_rows : index,%ht_ptr : memref<?xi32>, %ll_key : memref<?xi32>,
            %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>, %free_index : memref<1xi32>)

        return

    }

    func.func @count_rows(%relation2 : memref<?xi32>, %relation2_rows : index, 
                %ht_ptr : memref<?xi32>, %prefix : memref<?xi32>) -> (i32) {
            
            // Constants
            %cidx_0 = arith.constant 0 : index
            %cidx_1 = arith.constant 1 : index
            
            %ci32_0 = arith.constant 0 : i32
            // //-------------> Keep threads per block constant at 1024.. Need to change this?
            %num_threads_per_block = arith.constant 1024 : index

            %num_blocks = func.call @calc_num_blocks(%relation2_rows, %num_threads_per_block) : (index, index) -> index


            gpu.launch_func @kernels::@count 
            blocks in (%num_blocks, %cidx_1, %cidx_1) 
            threads in (%num_threads_per_block, %cidx_1, %cidx_1)
            args( %relation2 : memref<?xi32>, %relation2_rows : index, %ht_ptr : memref<?xi32>, %prefix : memref<?xi32>)

            return %ci32_0 : i32
        }
    
    gpu.module @kernels {
    
        func.func @hash(%key : i32) -> i32{
            // return modulo 100
            %cidx_100 = arith.constant 100 : i32
            %hash_val = arith.remui %key, %cidx_100 : i32
            return %hash_val : i32
        }


        func.func @Insert_Node_HT(%key : i32, %ht_ptr : memref<?xi32>, %ll_key : memref<?xi32>,
            %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>, %free_index : memref<1xi32>, %g_thread_idx : index) {

            // constants
            %cidx_neg1 = arith.constant -1 : index
            %cidx_0 = arith.constant 0 : index 

            %ci32_neg1 = arith.constant -1 : i32
            %ci32_1 = arith.constant 1 : i32

            // The free index at which the node is being modified
            %index_i32 = memref.atomic_rmw addi %ci32_1, %free_index[%cidx_0] : (i32, memref<1xi32>) -> i32

            // cast the i32 to index
            %index = arith.index_cast %index_i32 : i32 to index

            // Insert key and rowID into the new node
            memref.store %key, %ll_key[%index] : memref<?xi32>
            memref.store %g_thread_idx, %ll_rowID[%index] : memref<?xindex>

            // compute the hash value
            %hash_val_ = func.call @hash(%key) : (i32) -> i32

            // cast the hash value to index
            %hash_val = arith.index_cast %hash_val_ : i32 to index

            %cmp_val = memref.load %ht_ptr[%hash_val] : memref<?xi32>
            
            // if the value is -1, then update it to %index
            %cmp = arith.cmpi "eq", %cmp_val, %ci32_neg1 : i32

            
            scf.if %cmp {

                memref.store %index_i32, %ht_ptr[%hash_val] : memref<?xi32>
                memref.store %cidx_neg1, %ll_next[%index] : memref<?xindex>
    
            }
            else{
                // implement memref.rmw to update the hash table
                %index_old_i32 = memref.atomic_rmw assign %index_i32, %ht_ptr[%hash_val] : (i32, memref<?xi32>) -> i32
                // cast the index to i32
                %index_old = arith.index_cast %index_old_i32 : i32 to index
                memref.store %index_old, %ll_next[%index] : memref<?xindex>

            }
            return
        }

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
                // gpu.printf "Thread ID: %lld \n" %tidx : index
                
                // print debugging constants
                %print_thread_id = arith.constant 0: index
                %print_block_id = arith.constant 0: index

                %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
                %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
                %should_print = arith.andi %should_print_thread, %should_print_block : i1

                // scf.if %should_print {
                //     gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
                // }
                
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
                gpu.printf "Thread ID: %lld \n" %tidx : index
                
                // print debugging constants
                %print_thread_id = arith.constant 0: index
                %print_block_id = arith.constant 0: index

                %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
                %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
                %should_print = arith.andi %should_print_thread, %should_print_block : i1

                scf.if %should_print {
                    gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
                }

                //constants
                %cidx_0 = arith.constant 0 : index
                %cidx_1 = arith.constant 1 : index
                %cidx_2 = arith.constant 2 : index

                %key = memref.load %relation1[%g_thread_idx] : memref<?xi32>
                
                func.call @Insert_Node_HT(%key, %ht_ptr, %ll_key, %ll_rowID, %ll_next, %free_index, %g_thread_idx) 
                 : (i32, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index) -> ()


            }
            
            gpu.return
        }

        gpu.func @count (%relation2 : memref<?xi32>, %relation2_rows : index, %ht_size : index,
                %ht_ptr : memref<?xi32>, %prefix : memref<?xi32>)
            kernel
        {
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            // Global_thread_index = bdim * bidx + tidx
            %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
            %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

            // Check if the thread is valid
            %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %relation2_rows : index

            scf.if %is_thread_valid {
                gpu.printf "Thread ID: %lld \n" %tidx : index
                
                // print debugging constants
                %print_thread_id = arith.constant 0: index
                %print_block_id = arith.constant 0: index

                %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
                %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
                %should_print = arith.andi %should_print_thread, %should_print_block : i1

                scf.if %should_print {
                    gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
                }

                //constants
                %cidx_0 = arith.constant 0 : index

                %ci32_0 = arith.constant 0 : i32
                

                // Initialize the prefix sum to 0  
                memref.store %ci32_0, %prefix[%g_thread_idx] : memref<?xi32>

                // Step 1: Compute the prefix sum for each thread, which is used for calculating the start index in the result array 
                // For each thread, compare its key with all keys in the smaller table

                %key1 = memref.load %table_x[%g_thread_idx, %cidx_0] : memref<?x?xi32>
                
                //%thread_sums stores the prefix sum for each thread in the block
                memref.store %cidx_0, %thread_sums[%tidx] : memref<1024xindex, 3>

                // for each key in the smaller table
                scf.for %j = %cidx_0 to %table_y_rows step %cidx_1 {
                    %key2 = memref.load %table_y[%j, %cidx_0] : memref<?x?xi32>
                    // compare the keys
                    %cmp = arith.cmpi "eq", %key1, %key2 : i32
                    // if keys match, increment the prefix sum
                    scf.if %cmp {
                        %cur_count = memref.load %thread_sums[%tidx] : memref<1024xindex, 3>
                        %new_count = arith.addi %cur_count, %cidx_1 : index
                        memref.store %new_count, %thread_sums[%tidx] : memref<1024xindex, 3>
                    }
                }
                gpu.barrier // we need this so that all the warps are done computing the thread local sums


                //Step 2: Compute the global prefix sum
                // Single threaded prefix sum

                %is_t0 = arith.cmpi "eq", %tidx, %cidx_0 : index
                scf.if %is_t0 {
                    scf.if %should_print {
                        gpu.printf "Thread start indices: [0, "
                    }
                    
                    // %temp_idx stores the current value of prefix sum
                    memref.store %cidx_0, %temp_idx[%cidx_0] : memref<1xindex> 

                    //For all threads in thread block
                    scf.for %i = %cidx_0 to %bdim step %cidx_1 {
                        %g_thread_index = arith.addi %g_thread_offset_in_blocks, %i : index

                        %is_valid = arith.cmpi "ult", %g_thread_index, %table_x_rows : index
                        scf.if %is_valid{

                            %cur_count = memref.load %thread_sums[%i] : memref<1024xindex, 3>
                            %cur_idx = memref.load %temp_idx[%cidx_0] : memref<1xindex>
                            %next_index = arith.addi %cur_idx, %cur_count : index

                            //thread_sums[i] stores the starting index to write from for thread i, which is 0 for thread 0
                            memref.store %cur_idx, %thread_sums[%i] : memref<1024xindex, 3>
                            memref.store %next_index, %temp_idx[%cidx_0] : memref<1xindex>

                            scf.if %should_print {
                                gpu.printf "%ld, " %next_index : index
                            }
                        }
                    }

                    scf.if %should_print {
                        gpu.printf "]\n"
                    }

                    //Compute global block offset
                    %total_elements = memref.load %temp_idx[%cidx_0] : memref<1xindex>
                    %total_elements_i32 = arith.index_cast %total_elements : index to i32
                    
                    %cur_block_offset = memref.atomic_rmw addi %total_elements_i32, %gblock_offset[%cidx_0] : (i32, memref<1xi32>) -> i32
                    %cur_block_offset_idx = arith.index_cast %cur_block_offset : i32 to index

                    memref.store %cur_block_offset_idx, %b_block_offset[%cidx_0] : memref<1xindex, 3>

                    scf.if %should_print {
                        gpu.printf "Current block# %ld offset: %ld, total elements: %ld\n" %bidx, %cur_block_offset_idx, %total_elements : index, index, index
                    }
                }

            


            gpu.return
        }

        // Kernel to perform hash join
        // gpu.func @probe (%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
        //     %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>) 
        //     //---------------> Size of shared memory is fixed for now. To be changed later 
        //     workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>) 
        //     private(%temp_idx: memref<1xindex>)
        //     kernel 
        // {
            
            
        // }
    }
    
    func.func @main() {

        // Table and table sizes have to be passed as arguments later on
        %relation1_rows = arith.constant 5 : index
        %relation2_rows = arith.constant 5 : index


        // Allocate and initialize the memrefs for keys of both relations
        %relation1 = memref.alloc(%relation1_rows) : memref<?xi32>
        call @init_relation(%relation1) : (memref<?xi32>) -> ()

        %relation2 = memref.alloc(%relation2_rows) : memref<?xi32>
        call @init_relation(%relation2) : (memref<?xi32>) -> ()


        // Allocate device memory for the relations
        %d_relation1 = gpu.alloc(%relation1_rows) : memref<?xi32>
        gpu.memcpy %d_relation1, %relation1 : memref<?xi32>, memref<?xi32>

        %d_relation2 = gpu.alloc(%relation2_rows) : memref<?xi32>
        gpu.memcpy %d_relation2, %relation2 : memref<?xi32>, memref<?xi32>

        // //-------------> Keep items per thread constant at 1.. Not sure about this
        // %items_per_thread = arith.constant 1 : index

    
        // //-------------> Initialize hash table size as 1000 for now
        %ht_size = arith.constant 1000 : index

        // number of rows in the first table(build) is the num_tuples in the linked list
        %ll_key, %ll_rowID, %ll_next, %ht_ptr = func.call @alloc_hash_table(%relation1_rows, %ht_size) 
        : (index, index) -> (memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>)

        
        func.call @init_hash_table(%ht_size, %ht_ptr) : (index, memref<?xi32>) -> ()


        func.call @build_table(%relation1, %relation1_rows, %ht_ptr, %ll_key, %ll_rowID, %ll_next) 
         : (memref<?xi32>, index, memref<?xi32>, memref<?xi32>, memref<?xindex>, memref<?xindex>) -> ()

        // Allocate memref for prefix sum array
        %prefix = gpu.alloc(%relation2_rows) : memref<?xi32>

        %result_size = func.call @count_rows(%relation2, %relation2_rows, %ht_ptr, %prefix)
         : (memref<?xi32>, index,  memref<?xi32>, memref<?xi32>) -> i32


        // Allocate device memory for the result
        %result_size_idx = arith.index_cast %result_size : i32 to index
        %d_result_r = gpu.alloc(%result_size_idx) : memref<?xi32>
        %d_result_s = gpu.alloc(%result_size_idx) : memref<?xi32>
       

        //func.call @probe_relation(%relation2, %relation2_rows, %)


        // Check the result of join
        %success = call @check(%relation1, %relation2, %relation1,%relation2)
        : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32

        // print success
        func.call @debugI32(%success) : (i32) -> ()

        // print the result
        %dst = memref.cast %relation1 : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    func.func private @init_relation(memref<?xi32>)
    func.func private @init_relation_index(memref<?xi32>)
    func.func private @check(memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> i32
    func.func private @printMemrefI32(memref<*xi32>)
}