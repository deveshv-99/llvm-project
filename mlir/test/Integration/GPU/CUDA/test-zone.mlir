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

    func.func @Init_hash_table (%num_tuples : index, %ht_size : index) -> !llvm.ptr<i8> {
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index

        %linked_list = llvm.alloca %num_tuples x !llvm.struct<(i32, i32, ptr)> :(index) -> !llvm.ptr<struct<(i32, i32, ptr)>>

        %hash_table = llvm.alloca %ht_size x !llvm.struct<(i32,ptr)> : (index) -> !llvm.ptr<struct<(i32,ptr)>>

        %ht = llvm.alloca %cidx_1 x !llvm.struct<(ptr,ptr)> : (index) -> !llvm.ptr<struct<(ptr,ptr)>>

        %ht_ptr = llvm.bitcast %ht: !llvm.ptr<struct<(ptr, ptr)>> to !llvm.ptr<i8> 
        
        return %ht_ptr : !llvm.ptr<i8>
    }

    func.func @hash(%key : i32){
        // return modulo 100
        %cidx_100 = arith.constant 100 : i32
        %hash_val = arith.remui %key, %cidx_100 : i32
        return %hash_val : i32
    }


    gpu.module @kernels {

        //gpu.func @build(%table_x_keys: memref<?xi32>, %table_x_vals: memref<?xi32>, %table_x_rows: index, %table_x_cols: index, %ht : !llvm.ptr<i8>)




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
        // gpu.func @probe (%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
        //     %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>) 
        //     //---------------> Size of shared memory is fixed for now. To be changed later 
        //     workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>) 
        //     private(%temp_idx: memref<1xindex>)
        //     kernel 
        // {
            
        //     %bdim = gpu.block_dim x
        //     %bidx = gpu.block_id x
        //     %tidx = gpu.thread_id x

        //     // Global_thread_index = bdim * bidx + tidx
        //     %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
        //     %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

        //     // Check if the thread is valid
        //     %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %table_x_rows : index

        //     scf.if %is_thread_valid {
        //         gpu.printf "Thread ID: %lld \n" %tidx : index
                
        //         // print debugging constants
        //         %print_thread_id = arith.constant 0: index
        //         %print_block_id = arith.constant 0: index

        //         %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
        //         %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
        //         %should_print = arith.andi %should_print_thread, %should_print_block : i1

        //         scf.if %should_print {
        //             gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
        //         }

        //         //constants
        //         %cidx_0 = arith.constant 0 : index
        //         %cidx_1 = arith.constant 1 : index
        //         %cidx_2 = arith.constant 2 : index


        //     }
            
        //     gpu.return
        // }
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
        %table1_rows = arith.constant 5 : index
        %table2_rows = arith.constant 5 : index
        
        // Number of columns is fixed at 2 for both tables for now
        %table1_cols = arith.constant 2 : index
        %table2_cols = arith.constant 2 : index

        //Initialize the tables to fixed values for now.. 
        %h_table1 = call @init(%table1_cols, %table1_rows) : (index,index) -> memref<?x?xi32>
        %h_table2 = call @init(%table2_cols, %table2_rows) : (index,index) -> memref<?x?xi32>

        %h_table1_keys= memref.alloc(%table1_rows) : memref<?xi32>
        %h_table1_vals= memref.alloc(%table1_rows) : memref<?xi32>
        %h_table2_keys= memref.alloc(%table2_rows) : memref<?xi32>
        %h_table2_vals= memref.alloc(%table2_rows) : memref<?xi32>

        //Separate the keys and the values
        scf.for %i = %cidx_0 to %table1_rows step %cidx_1 {
            %key1 = memref.load %h_table1[%cidx_0, %i] : memref<?x?xi32>
            memref.store %key1, %h_table1_keys[%i] : memref<?xi32>

            %val1 = memref.load %h_table1[%cidx_1, %i] : memref<?x?xi32>
            memref.store %val1, %h_table1_vals[%i] : memref<?xi32>
        }

        scf.for %i = %cidx_0 to %table2_rows step %cidx_1 {
            %key2 = memref.load %h_table2[%cidx_0, %i] : memref<?x?xi32>
            memref.store %key2, %h_table2_keys[%i] : memref<?xi32>

            %val2 = memref.load %h_table2[%cidx_1, %i] : memref<?x?xi32>
            memref.store %val2, %h_table2_vals[%i] : memref<?xi32>
        }


        // Allocate device memory for the tables
        %d_table1_keys = gpu.alloc(%table1_rows) : memref<?xi32>
        gpu.memcpy %d_table1_keys, %h_table1_keys : memref<?xi32>, memref<?xi32>
        %d_table1_vals = gpu.alloc(%table1_rows) : memref<?xi32>
        gpu.memcpy %d_table1_vals, %h_table1_vals : memref<?xi32>, memref<?xi32>

        %d_table2_keys = gpu.alloc(%table2_rows) : memref<?xi32>
        gpu.memcpy %d_table2_keys, %h_table2_keys : memref<?xi32>, memref<?xi32>
        %d_table2_vals = gpu.alloc(%table2_rows) : memref<?xi32>
        gpu.memcpy %d_table2_vals, %h_table2_vals : memref<?xi32>, memref<?xi32>


       


        // //-------------> Allocate result array to contain number of rows as size of table1*table2... Need something better than this?
        %result_rows = arith.muli %table1_rows, %table2_rows : index
        // Result columns will be (sum of columns of both tables - 1 (for the duplicate key))
        %result_cols_ = arith.addi %table1_cols, %table2_cols : index 
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
        %table_1_or_2_as_build = arith.cmpi "ult", %table1_rows, %table2_rows : index

        //Number of threads would be the number of rows in the larger table
        %total_threads = arith.select %table_1_or_2_as_build, %table2_rows, %table1_rows : index

        // To calculate the number of blocks needed, perform ceil division: num_blocks = (total_threads + num_threads_per_block - 1) / num_threads_per_block
        // TODO: arith.ceildivui gives errors which i cant figure out. so using the above thing instead..
        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index


        // defining parameters to be passed to the kernel
        // Table x is the build table and Table y is the probe table
        %table_x_keys = arith.select %table_1_or_2_as_build, %d_table1_keys, %d_table2_keys : memref<?xi32>
        %table_x_vals = arith.select %table_1_or_2_as_build, %d_table1_vals, %d_table2_vals : memref<?xi32>
        %table_y_keys = arith.select %table_1_or_2_as_build, %d_table2_keys, %d_table1_keys : memref<?xi32>
        %table_y_vals = arith.select %table_1_or_2_as_build, %d_table2_vals, %d_table1_vals : memref<?xi32>

        
        %table_x_rows = arith.select %table_1_or_2_as_build, %table1_rows, %table2_rows : index
        %table_x_cols = arith.select %table_1_or_2_as_build, %table1_cols, %table2_cols : index
        %table_y_rows = arith.select %table_1_or_2_as_build, %table2_rows, %table1_rows : index
        %table_y_cols = arith.select %table_1_or_2_as_build, %table2_cols, %table1_cols : index

        // //-------------> Initialize hash table size as 100 for now
        %ht_size = arith.constant 100 : index
        // %table_x_rows is the num_tuples
        %ht = func.call @Init_hash_table(%table_x_rows, %ht_sizes) : (index, index) -> !llvm.ptr<i8>




        gpu.launch_func @kernels::@build
        blocks in (%num_blocks, %cidx_1, %cidx_1) 
        threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        args(%table_x_keys: memref<?xi32>, %table_x_vals: memref<?xi32>, %table_x_rows: index, %table_x_cols: index, %ht : !llvm.ptr<i8>)

        

        // print the result
        %dst = memref.cast %h_table1_vals : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
}