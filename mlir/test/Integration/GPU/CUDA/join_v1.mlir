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

    func.func @alloc_hash_table(%num_tuples : index, %ht_size : index) -> () {
        
        // Allocate linked_list 
        %ll_key = gpu.alloc(%num_tuples) : memref<?xi32>
        %ll_rowID = gpu.alloc(%num_tuples) : memref<?xindex>
        %ll_next = gpu.alloc(%num_tuples) : memref<?xindex>

        // Allocate hash table
        %ht_val = gpu.alloc(%ht_size) : memref<?xi32>
        %ht_ptr = gpu.alloc(%ht_size) : memref<?xindex>
        
        // Return all the allocated memory

    }

    gpu.module @kernels {
    
        func.func @hash(%key : i32) -> i32{
            // return modulo 100
            %cidx_100 = arith.constant 100 : i32
            %hash_val = arith.remui %key, %cidx_100 : i32
            return %hash_val : i32
        }

        // func.func @Insert_Node_HT(%ht : !llvm.ptr<i8>, %hash_val : i32, %key : i32, %val : i32, %free_index : memref<1xi32>) {

        //     %cidx_0 = arith.constant 0 : index 

        //     %ci32_1 = arith.constant 1 : i32

        //     %c0 = llvm.mlir.constant (0 : i32) : i32
        //     %c1 = llvm.mlir.constant (1 : i32) : i32
        //     %c2 = llvm.mlir.constant (2 : i32) : i32

        //     %const_1 = llvm.mlir.constant (1 : i1) : i1

        //     // The free index to which the node is being inserted to
        //     %index = memref.atomic_rmw addi %ci32_1, %free_index[%cidx_0] : (i32, memref<1xi32>) -> i32

        //     //Loading linked list and hash table
        //     %ht_ptr = llvm.bitcast %ht: !llvm.ptr<i8> to !llvm.ptr<struct<(ptr, ptr)>>
        //     %ptr = llvm.getelementptr %ht_ptr[%c0] : (!llvm.ptr<struct<(ptr, ptr)>>, i32) -> (!llvm.ptr<struct<(ptr, ptr)>>)
        //     %linked_list = llvm.getelementptr %ptr[%c0] : (!llvm.ptr<struct<(ptr, ptr)>>, i32) -> (!llvm.ptr<ptr>)
        //     %ll_ptr = llvm.load %linked_list : !llvm.ptr<ptr>
        //     %ll = llvm.bitcast %ll_ptr: !llvm.ptr to !llvm.ptr<struct<(i32, i32, ptr)>>

        //     %hash_table = llvm.getelementptr %ptr[%c1] : (!llvm.ptr<struct<(ptr, ptr)>>, i32) -> (!llvm.ptr<ptr>)  
        //     %table_ptr = llvm.load %hash_table : !llvm.ptr<ptr>
        //     %table = llvm.bitcast %table_ptr: !llvm.ptr to !llvm.ptr<struct<(i32, ptr)>>

        //     // Insert key and value into the new node
        //     %node = llvm.getelementptr %ll[%index] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<struct<(i32, i32, ptr)>>)
        //     %key_ptr = llvm.getelementptr %node[%c0] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<i32>)
        //     llvm.store %key, %key_ptr : !llvm.ptr<i32>

        //     %val_ptr = llvm.getelementptr %node[%c1] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<i32>)
        //     llvm.store %val, %val_ptr : !llvm.ptr<i32>

        //     // Add values in the hash table
        //     %entry = llvm.getelementptr %table[%hash_val] : (!llvm.ptr<struct<(i32, ptr)>>, i32) -> (!llvm.ptr<struct<(i32, ptr)>>)
        //     %entry_key = llvm.getelementptr %entry[%c0] : (!llvm.ptr<struct<(i32, ptr)>>, i32) -> (!llvm.ptr<i32>)
            
        //     // Storing the hash value in the hash table.. need to shift it to when initializing the hash table
        //     llvm.store %key, %entry_key : !llvm.ptr<i32>

        //     // find cmp_ptr
        //     %cmp_ptr = llvm.getelementptr %entry[%c1] : (!llvm.ptr<struct<(i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)
        //     %cmp = llvm.load %cmp_ptr : !llvm.ptr<ptr>

        //     // // 1. without atomics
        //     // //first, add the value of cmp to the next of the new node, then change the value of cmp to the new node
        //     // %next = llvm.getelementptr %node[%c2] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)
        //     // llvm.store %cmp, %next : !llvm.ptr<ptr>
        //     // %casted_node = llvm.bitcast %node: !llvm.ptr<struct<(i32, i32, ptr)>> to !llvm.ptr
        //     // llvm.store %casted_node, %cmp_ptr : !llvm.ptr<ptr>



        //     //2. with atomics

        //     // cast
        //     %casted_node = llvm.bitcast %node: !llvm.ptr<struct<(i32, i32, ptr)>> to !llvm.ptr

        //     %res = scf.while (%arg1 = %cmp) :(!llvm.ptr) -> !llvm.ptr {
        //         // "Before" region.
        //         // In a "do-while" loop, this region contains the loop body.
        //         %cmp_value = llvm.load %cmp_ptr : !llvm.ptr<ptr>
        //         %x = llvm.cmpxchg %cmp_ptr, %cmp_value, %casted_node "monotonic" "monotonic" : !llvm.ptr<ptr>, !llvm.ptr
        //         %value = llvm.extractvalue %x[0] : !llvm.struct<(!llvm.ptr, i1)>

        //         // condition
        //         %success = llvm.extractvalue %x[1] : !llvm.struct<(!llvm.ptr, i1)>

        //         // negate success
        //         %fail = llvm.xor %success, %const_1 : i1

        //         scf.condition(%fail) %value : !llvm.ptr

        //     } do {
        //         ^bb0(%arg: !llvm.ptr):
        //         scf.yield %arg : !llvm.ptr
        //     }

        //     // get the next_node pointer from the current node
        //     %next = llvm.getelementptr %node[%c2] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)

        //     // Store the address of the previous node into the "next_node" ptr of the new node
        //     llvm.store %res, %next : !llvm.ptr<ptr>

        //     // store the address of the new node in the pointer of hash table
        //     llvm.store %casted_node, %cmp_ptr : !llvm.ptr<ptr>
            
        //     return
        // }

        // gpu.func @build(%table_x_keys: memref<?xi32>, %table_x_vals: memref<?xi32>, %table_x_rows: index, %table_x_cols: index, %ht : !llvm.ptr<i8>, %free_index : memref<1xi32>)
        //   kernel
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

        //         %key = memref.load %table_x_keys[%g_thread_idx] : memref<?xi32>
        //         %val = memref.load %table_x_vals[%g_thread_idx] : memref<?xi32>
        //         %hash_val = func.call @hash(%key) : (i32) -> i32
        //         func.call @Insert_Node_HT(%ht, %hash_val, %key, %val, %free_index) : (!llvm.ptr<i8>, i32, i32, i32, memref<1xi32>) -> ()


        //     }
            
        //     gpu.return
        // }

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

        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32


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


        // //-------------> Allocate result array to contain number of rows as size of table1*table2... Need something better than this?
        %result_rows = arith.muli %relation1_rows, %relation2_rows : index

        // No of columns in result will be 2
        %result_cols = arith.constant 2 : index

        // Allocate device memory for the result
        %d_result = gpu.alloc(%result_rows, %result_cols) : memref<?x?xindex>

        // //-------------> Keep threads per block constant at 1024.. Need to change this?
        %num_threads_per_block = arith.constant 1024 : index

        // //-------------> Keep items per thread constant at 1.. Not sure about this either
        // %items_per_thread = arith.constant 1 : index

    

        // Two KERNELS required... One for building the hash table and the other for probing


        // //-------------> Initialize hash table size as 1000 for now
        %ht_size = arith.constant 1000 : index

        // %table_x_rows is the num_tuples
        // %ht = func.call @alloc_hash_table(%relation1_rows, %ht_size) : (index, index) -> ()

        //Number of threads for build would be the number of rows in relation1

        // To calculate the number of blocks needed, perform ceil division: num_blocks = (total_threads + num_threads_per_block - 1) / num_threads_per_block
        // arith.ceildivui gives errors which i cant figure out. so using the above thing instead..
        %for_ceil_div_ = arith.addi %relation1_rows, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks = arith.divui %for_ceil_div, %num_threads_per_block : index


        //global variable for all blocks
        //%free_index = gpu.alloc() : memref<1xi32>

        //store 1 in the global block offset
        %h_free_index= memref.alloc() : memref<1xi32>
        memref.store %ci32_1, %h_free_index[%cidx_0] : memref<1xi32>
        //gpu.memcpy %free_index, %h_free_index : memref<1xi32>, memref<1xi32>


        // gpu.launch_func @kernels::@build
        // blocks in (%num_blocks, %cidx_1, %cidx_1) 
        // threads in (%num_threads_per_block, %cidx_1, %cidx_1)
        // args(%table_x_keys: memref<?xi32>, %table_x_vals: memref<?xi32>, %table_x_rows: index, %table_x_cols: index, %ht : !llvm.ptr<i8>, %free_index : memref<1xi32>)



        //--------------  TODO----------------------------
        //gpu.launch_func @kernels::@probe



        // print the result
        %dst = memref.cast %relation1 : memref<?xi32> to memref<*xi32>
        call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    func.func private @init_relation(memref<?xi32>)
    func.func private @printMemrefI32(memref<*xi32>)
}