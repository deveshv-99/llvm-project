// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    func.func @init(%size: index) -> memref<?x2xi32> {

        %arr = memref.alloc(%size) : memref<?x2xi32>
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %ci32_1 = arith.constant 1 : i32
        scf.for %i = %cidx_0 to %size step %cidx_1 {
            %t = arith.index_cast %i : index to i32
            %val = arith.addi %t, %ci32_1 : i32

            memref.store %t, %arr[%i,%cidx_0] : memref<?x2xi32>
            memref.store %val, %arr[%i,%cidx_1] : memref<?x2xi32>
        }
        return %arr: memref<?x2xi32>
    }

    gpu.module @kernels {
        //func.func @line_order_outer(%)



        gpu.func @nested_join (%d_part : memref<?x2xi32>, %d_line_order : memref<?x2xi32>, %d_result : memref<?x3xi32>, 
            %items_per_thread : index, %p_size : index, %lo_size : index, %which_table : index, %gblock_offset: memref<1xi32>) 
            workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>)
            private(%temp_idx: memref<1xindex>)
            kernel 
        {
            // standard code: result thread id r_tid = bdim * bidx + tid
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            %num_threads = arith.muli %bdim, %bidx : index
            %boffset = arith.muli %num_threads, %items_per_thread : index
            %elem_offset = arith.muli %tidx, %items_per_thread : index
            %thread_start_idx = arith.addi %boffset, %elem_offset : index

            // print debugging constants
            %print_thread_id = arith.constant 0: index
            %print_block_id = arith.constant 1: index
            %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
            %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
            %should_print = arith.andi %should_print_thread, %should_print_block : i1

            scf.if %should_print {
                gpu.printf "Block ID: %ld, Thread ID: %ld, thread_start_idx: %ld\n" %bidx, %tidx, %thread_start_idx : index, index, index
            }

            //constants
            %cidx_0 = arith.constant 0 : index
            %cidx_1 = arith.constant 1 : index
            %cidx_2 = arith.constant 2 : index

            // Step 1: Compute the prefix sum for each thread for calculating the start index in the result array 
            // for each thread, compare its key with all keys in the smaller table

            scf.for %i = %cidx_0 to %items_per_thread step %cidx_1 {

                %idx = arith.addi %thread_start_idx, %i : index
                //-------------------- Assuming Line Order is the outer loop ---------------------
                %key1 = memref.load %d_line_order[%idx, %cidx_0] : memref<?x2xi32>
                // for each key in the smaller table
                scf.for %j = %cidx_0 to %p_size step %cidx_1 {
                    %key2 = memref.load %d_part[%j, %cidx_0] : memref<?x2xi32>
                    // compare the keys
                    %cmp = arith.cmpi "eq", %key1, %key2 : i32
                    // if keys match, increment the prefix sum
                    scf.if %cmp {
                        %cur_count = memref.load %thread_sums[%tidx] : memref<1024xindex, 3>
                        %new_count = arith.addi %cur_count, %cidx_1 : index
                        memref.store %new_count, %thread_sums[%tidx] : memref<1024xindex, 3>
                    }
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
                memref.store %cidx_0, %temp_idx[%cidx_0] : memref<1xindex> //Start with first thread
                //For all threads in thread block
                scf.for %i = %cidx_0 to %bdim step %cidx_1 {

                    %cur_idx = memref.load %temp_idx[%i] : memref<1xindex>
                    %cur_count = memref.load %thread_sums[%i] : memref<1024xindex, 3>
                    %next_index = arith.addi %cur_idx, %cur_count : index
                    memref.store %next_index, %temp_idx[%cidx_0] : memref<1xindex>

                    scf.if %should_print {
                        gpu.printf "%ld, " %next_index : index
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

            gpu.barrier // other threads need to wait until the prefix sum is complete

            // Step 3: Each thread needs to store its value in the result array
            // The start index are loaded from the thread_sums array
            // The current index is stored in the temp_idx array for each thread
            // It is incremented after each store that passes the predicate

            %cur_block_offset = memref.load %b_block_offset[%cidx_0] : memref<1xindex, 3>
            %cur_thread_offset = memref.load %thread_sums[%tidx] : memref<1024xindex, 3>
            %start_index = arith.addi %cur_block_offset, %cur_thread_offset : index
            memref.store %start_index, %temp_idx[%cidx_0] : memref<1xindex>

            scf.if %should_print {
                gpu.printf "Block %ld, thread ID: %ld: start_index: %ld \n" %bidx, %tidx, %start_index: index, index, index
            }

            scf.for %i = %cidx_0 to %items_per_thread step %cidx_1 {

                %idx = arith.addi %thread_start_idx, %i : index
                %key1 = memref.load %d_line_order[%idx, %cidx_0] : memref<?x2xi32>

                // for each key in the smaller table
                scf.for %j = %cidx_0 to %p_size step %cidx_1 {
                    %key2 = memref.load %d_part[%j, %cidx_0] : memref<?x2xi32>

                    // compare the keys
                    %cmp = arith.cmpi "eq", %key1, %key2 : i32
                    // if keys match, increment the prefix sum
                    scf.if %cmp {

                        %cur_idx = memref.load %temp_idx[%cidx_0] : memref<1xindex>

                        %val1 = memref.load %d_line_order[%idx, %cidx_1] : memref<?x2xi32>
                        %val2 = memref.load %d_part[%j, %cidx_1] : memref<?x2xi32>

                        memref.store %key1, %d_result[%cur_idx, %cidx_0] : memref<?x3xi32>
                        memref.store %val1, %d_result[%cur_idx, %cidx_1] : memref<?x3xi32>
                        memref.store %val2, %d_result[%cur_idx, %cidx_2] : memref<?x3xi32>

                    }
                }
            }


            gpu.return
        }
    }

    
    func.func @main() {

        %p_size = arith.constant 4 : index
        %lo_size = arith.constant 4 : index
        
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32

        //Initialize the tables
        %h_part = call @init(%p_size) : (index) -> memref<?x2xi32>
        %h_line_order = call @init(%lo_size) : (index) -> memref<?x2xi32>

        %d_part = gpu.alloc(%p_size) : memref<?x2xi32>
        gpu.memcpy %d_part, %h_part : memref<?x2xi32>, memref<?x2xi32>
        %d_line_order = gpu.alloc(%lo_size) : memref<?x2xi32>
        gpu.memcpy %d_line_order, %h_line_order : memref<?x2xi32>, memref<?x2xi32>

        // Allocate result array to be of maximum size
        %result_size = arith.muli %p_size, %lo_size : index
        %d_result = gpu.alloc(%result_size) : memref<?x3xi32>


        //Whichever table is smaller, we use that for comparison (as the outer loop)
        // i.e. the larger table is allocated to threads
        %lo_or_p_as_outer = arith.cmpi "ult", %lo_size, %p_size : index

        //%total_threads contains the total number of threads to be created
        %total_threads = arith.select %lo_or_p_as_outer, %p_size, %lo_size : index

        // part table is 0, line_order table is 1
        %which_table = arith.select %lo_or_p_as_outer, %cidx_0, %cidx_1 : index

        //Keep threads per block constant at 1024
        %num_threads_per_block = arith.constant 1024 : index

        //Calculate the number of blocks needed
        //perform ceil division
        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index

        %items_per_thread = arith.constant 1 : index


        //launch the kernel
        %gblock_offset = gpu.alloc() : memref<1xi32>
        gpu.launch_func @kernels::@nested_join
            blocks in (%num_blocks, %cidx_1, %cidx_1) 
            threads in (%num_threads_per_block, %cidx_1, %cidx_1)
            args(%d_part : memref<?x2xi32>, %d_line_order : memref<?x2xi32>, %d_result : memref<?x3xi32>, 
            %items_per_thread : index, %p_size : index, %lo_size : index, %which_table : index, %gblock_offset : memref<1xi32>)

        // copy the result column from the device to host
        %h_result = memref.alloc(%result_size) : memref<?x3xi32>
        gpu.memcpy %h_result, %d_result : memref<?x3xi32>, memref<?x3xi32>

        // print the result
        %dst = memref.cast %h_result : memref<?x3xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()


        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
}