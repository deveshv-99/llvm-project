// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

    // To initialize the tables with fixed values.. in column major order
    func.func @init(%cols: index, %rows: index) -> memref<?x?xi32> {

        %arr = memref.alloc(%cols, %rows) : memref<?x?xi32>
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %ci32_1 = arith.constant 1 : i32

        scf.for %i = %cidx_0 to %cols step %cidx_1 {
            scf.for %j = %cidx_0 to %rows step %cidx_1 {

                %i_i32 = arith.index_cast %i : index to i32
                %j_i32 = arith.index_cast %j : index to i32
                %val = arith.addi %i_i32, %j_i32 : i32
                memref.store %val, %arr[%i,%j] : memref<?x?xi32>
            }
        }
        return %arr: memref<?x?xi32>
    }

    gpu.module @kernels {

        func.func @build_hash_table(%table_x : memref<?x?xi32>, %table_x_rows : index) -> memref<?x?xi32> {
            // Constants
            %cidx_0 = arith.constant 0 : index
            %cidx_1 = arith.constant 1 : index

            // Number of different hash values is kept 100 for now
            %ht_size = arith.constant 100 : index

            %hash_table = func.call @Init_hash_table(%ht_size, %table_x_rows) : (index, index) -> memref<?x?xi32>
            
            %key = memref.
            scf.for %i = %cidx_0 to %table_x_rows step %cidx_1 {
                %hash_val = func.call @hash(%key) : (i32) -> i32
                func.call @Insert_Node_HT(%hash_table, %hash_val, %key, %val) : (memref<?x?xi32>, i32, i32,i32) -> ()
            }

            return %hash_table : memref<?x?xi32>
        }

        // Kernel to perform nested join
        gpu.func @hash_join (%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
            %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>) 
            //---------------> Size of shared memory is fixed for now. To be changed later 
            workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>) 
            private(%temp_idx: memref<1xindex>)
            kernel 
        {
            
            %bdim = gpu.block_dim x
            %bidx = gpu.block_id x
            %tidx = gpu.thread_id x

            // Global_thread_index = bdim * bidx + tidx
            %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
            %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

            // Check if the thread is valid
            %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %table_x_rows : index

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


            }
            
            gpu.return
        }
    }
    
    func.func @main() {

        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32

        // Table sizes have to be passed as arguments later on, with the number of columns for each table.
        // For our specific example, part_table is table_1 and line_order is table_2 
        %table_1_rows = arith.constant 20 : index
        %table_2_rows = arith.constant 20 : index
        
        %table_1_cols = arith.constant 2 : index
        %table_2_cols = arith.constant 2 : index

        //Initialize the tables to fixed values for now.. 
        %h_table_1 = call @init(%table_1_cols, %table_1_rows) : (index,index) -> memref<?x?xi32>
        %h_table_2 = call @init(%table_2_cols, %table_2_rows) : (index,index) -> memref<?x?xi32>

        //Separate the keys and the values
        %h_table_1_keys = memref.subview %h_table_1[%cidx_0, %cidx_0], [%table_1_rows, %ci32_1] : memref<?x?xi32>
        %h_table_1_vals = memref.subview %h_table_1[%cidx_0, %ci32_1], [%table_1_rows, %ci32_1] : memref<?x?xi32>

        // Allocate device memory for the tables
        %d_table_1 = gpu.alloc(%table_1_rows, %table_1_cols) : memref<?x?xi32>
        gpu.memcpy %d_table_1, %h_table_1 : memref<?x?xi32>, memref<?x?xi32>
        %d_table_2 = gpu.alloc(%table_2_rows, %table_2_cols) : memref<?x?xi32>
        gpu.memcpy %d_table_2, %h_table_2 : memref<?x?xi32>, memref<?x?xi32>

        // //-------------> Allocate result array to contain number of rows as size of table1*table2... Need something better than this?
        %result_rows = arith.muli %table_1_rows, %table_2_rows : index
        // Result columns will be (sum of columns of both tables - 1 (for the duplicate key))
        %result_cols_ = arith.addi %table_1_cols, %table_2_cols : index 
        %result_cols = arith.subi %result_cols_, %cidx_1 : index 

        %d_result = gpu.alloc(%result_rows, %result_cols) : memref<?x?xi32>

        // //-------------> Keep threads per block constant at 1024.. Need to change this?
        %num_threads_per_block = arith.constant 1024 : index

        // //-------------> Keep items per thread constant at 1.. Not sure about this either
        // %items_per_thread = arith.constant 1 : index

        //global variable for all blocks
        %gblock_offset = gpu.alloc() : memref<1xi32>

        // Whichever table is smaller, we build hash table upon that
        // so the larger table is used for probing
        %table_1_or_2_as_build = arith.cmpi "ult", %table_1_rows, %table_2_rows : index

        //Number of threads would be the number of rows in the larger table
        %total_threads = arith.select %table_1_or_2_as_build, %table_2_rows, %table_1_rows : index

        // To calculate the number of blocks needed, perform ceil division: num_blocks = (total_threads + num_threads_per_block - 1) / num_threads_per_block
        // TODO: arith.ceildivui gives errors which i cant figure out. so using the above thing instead..
        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index


        // defining parameters to be passed to the kernel
        // Table x is the build table, which will be assigned to threads. and Table y is the probe table
        %table_x = arith.select %table_1_or_2_as_build, %d_table_1, %d_table_2 : memref<?x?xi32>
        %table_y = arith.select %table_1_or_2_as_build, %d_table_2, %d_table_1 : memref<?x?xi32>
        
        %table_x_rows = arith.select %table_1_or_2_as_build, %table_2_rows, %table_1_rows : index
        %table_x_cols = arith.select %table_1_or_2_as_build, %table_2_cols, %table_1_cols : index
        %table_y_rows = arith.select %table_1_or_2_as_build, %table_1_rows, %table_2_rows : index
        %table_y_cols = arith.select %table_1_or_2_as_build, %table_1_cols, %table_2_cols : index

        gpu.launch_func @kernels::@nested_join
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
            %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>)

        // copy the result column from the device to host
        %h_result = memref.alloc(%result_rows, %result_cols) : memref<?x?xi32>
        gpu.memcpy %h_result, %d_result : memref<?x?xi32>, memref<?x?xi32>

        // print the result
        %dst = memref.cast %h_result : memref<?x?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
}