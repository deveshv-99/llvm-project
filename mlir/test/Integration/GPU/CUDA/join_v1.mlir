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
    func.func @calc_num_blocks(%total_threads : index, %num_threads_per_block: index) -> index {
        
        %for_ceil_div_ = arith.addi %total_threads, %num_threads_per_block : index
        %for_ceil_div = arith.subi %for_ceil_div_, %cidx_1 : index
        %num_blocks_ = arith.divui %for_ceil_div, %num_threads_per_block : index
        %num_blocks = scf.if %num_blocks_ {
            scf.yield %num_blocks_ : index
        }
        scf.else {
            scf.yield %cidx_1 : index
        }

        return %num_blocks : index
    }

    
    gpu.module @kernels {
    
        func.func @hash(%key : i32) -> i32{
            // return modulo 100
            %cidx_100 = arith.constant 100 : i32
            %hash_val = arith.remui %key, %cidx_100 : i32
            return %hash_val : i32
        }

        func.func @alloc_hash_table(%num_tuples : index, %ht_size : index) 
        -> ( memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>, memref<?xindex>) {
            
            // Allocate linked_list 
            %ll_key = gpu.alloc(%num_tuples) : memref<?xi32>
            %ll_rowID = gpu.alloc(%num_tuples) : memref<?xindex>
            %ll_next = gpu.alloc(%num_tuples) : memref<?xindex>

            // Allocate hash table
            %ht_val = gpu.alloc(%ht_size) : memref<?xi32>
            %ht_ptr = gpu.alloc(%ht_size) : memref<?xindex>
            
            // Return all the allocated memory
            return %ll_key, %ll_rowID, %ll_next, %ht_val, %ht_ptr
             : memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>, memref<?xindex>

        }

        func.func @init_hash_table(%ht_size : index, %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>) {
            
            // Constants
            %cidx_1 = arith.constant 1 : index

            // //-------------> Keep threads per block constant at 1024.. Need to change this?
            %num_threads_per_block = arith.constant 1024 : index

            %num_blocks = func.call @calc_num_blocks(%ht_size, %num_threads_per_block) : (index, index) -> index
            
            gpu.launch_func @kernels::@init_ht
            blocks in (%num_blocks, %cidx_1, %cidx_1)
            threads in (%num_threads_per_block, %cidx_1, %cidx_1)
            args(%ht_size : index, %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>)

            return
        }


        func.func @build_table(%relation1 : memref<?xi32>, %relation1_rows : index, %ht_size : index, %ht_val : memref<?xi32>, 
            %ht_ptr : memref<?xindex>, %ll_key : memref<?xi32>, %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>) {
            
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

            %num_blocks = func.call @calc_num_blocks(%ht_size, %num_threads_per_block) : (index, index) -> index

            gpu.launch_func @kernels::@build
            blocks in (%num_blocks, %cidx_1, %cidx_1) 
            threads in (%num_threads_per_block, %cidx_1, %cidx_1)
            args( %relation1 : memref<?xi32>, %relation1_rows : index, %ht_size : index,
                %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>, %ll_key : memref<?xi32>, %ll_rowID : memref<?xindex>, 
                %ll_next : memref<?xindex>, %free_index : memref<1xi32>)

            return

        }


        func.func @count_rows(%relation2 : memref<?xi32>, %relation2_rows : index, %ht_size : index,
                %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>, %prefix : memref<?xindex>) -> index {
            
            // Constants
            %cidx_0 = arith.constant 0 : index
            %cidx_1 = arith.constant 1 : index
            
            // //-------------> Keep threads per block constant at 1024.. Need to change this?
            %num_threads_per_block = arith.constant 1024 : index

            %num_blocks = func.call @calc_num_blocks(%ht_size, %num_threads_per_block) : (index, index) -> index

            // --------> Initialize the prefix sum array to 0 in kernel
            gpu.launch_func @kernels::@count
            blocks in (%num_blocks, %cidx_1, %cidx_1) 
            threads in (%num_threads_per_block, %cidx_1, %cidx_1)
            args( %relation2 : memref<?xi32>, %relation2_rows : index, %ht_size : index,
                %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>, %prefix : memref<?xi32>)

            return
        }


        func.func @Insert_Node_HT(%key : index, %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>, %ll_key : memref<?xi32>,
            %ll_rowID : memref<?xindex>, %ll_next : memref<?xindex>, %free_index : memref<1xi32>, %g_thread_idx : index) {

            // constants
            %cidx_0 = arith.constant 0 : index 
            %ci32_1 = arith.constant 1 : i32

            // The free index at which the node is being modified
            %index = memref.atomic_rmw addi %ci32_1, %free_index[%cidx_0] : (i32, memref<1xi32>) -> i32

            // Insert key and rowID into the new node
            memref.store %key, %ll_key[%index] : memref<?xi32>
            memref.store %g_thread_idx, %ll_rowID[%index] : memref<?xindex>

            %hash_val = func.call @hash(%key) : (i32) -> i32

            
            // Storing the hash value in the hash table.. need to shift it to when initializing the hash table
            llvm.store %key, %entry_key : !llvm.ptr<i32>

            // find cmp_ptr
            %cmp_ptr = llvm.getelementptr %entry[%c1] : (!llvm.ptr<struct<(i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)
            %cmp = llvm.load %cmp_ptr : !llvm.ptr<ptr>

        //     // // 1. without atomics
        //     // //first, add the value of cmp to the next of the new node, then change the value of cmp to the new node
        //     // %next = llvm.getelementptr %node[%c2] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)
        //     // llvm.store %cmp, %next : !llvm.ptr<ptr>
        //     // %casted_node = llvm.bitcast %node: !llvm.ptr<struct<(i32, i32, ptr)>> to !llvm.ptr
        //     // llvm.store %casted_node, %cmp_ptr : !llvm.ptr<ptr>



        //     //2. with atomics

            // cast
            %casted_node = llvm.bitcast %node: !llvm.ptr<struct<(i32, i32, ptr)>> to !llvm.ptr

            %res = scf.while (%arg1 = %cmp) :(!llvm.ptr) -> !llvm.ptr {
                // "Before" region.
                // In a "do-while" loop, this region contains the loop body.
                %cmp_value = llvm.load %cmp_ptr : !llvm.ptr<ptr>
                %x = llvm.cmpxchg %cmp_ptr, %cmp_value, %casted_node "monotonic" "monotonic" : !llvm.ptr<ptr>, !llvm.ptr
                %value = llvm.extractvalue %x[0] : !llvm.struct<(!llvm.ptr, i1)>

                // condition
                %success = llvm.extractvalue %x[1] : !llvm.struct<(!llvm.ptr, i1)>

                // negate success
                %fail = llvm.xor %success, %const_1 : i1

                scf.condition(%fail) %value : !llvm.ptr

            } do {
                ^bb0(%arg: !llvm.ptr):
                scf.yield %arg : !llvm.ptr
            }

            // get the next_node pointer from the current node
            %next = llvm.getelementptr %node[%c2] : (!llvm.ptr<struct<(i32, i32, ptr)>>, i32) -> (!llvm.ptr<ptr>)

            // Store the address of the previous node into the "next_node" ptr of the new node
            llvm.store %res, %next : !llvm.ptr<ptr>

            // store the address of the new node in the pointer of hash table
            llvm.store %casted_node, %cmp_ptr : !llvm.ptr<ptr>
            
            return
        }

        gpu.func @init_ht(%ht_size : index, %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>) 
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

                // Initialize the hash table
                %cidx_neg1 = arith.constant -1 : index
                %cidx_0 = arith.constant 0 : index
                %cidx_1 = arith.constant 1 : index

                // Since the hash function is a modulo, I assign the index as the value
                // Set the ht_val to index and ht_ptr to -1

                %g_thread_i32 = arith.index_cast %g_thread_idx : index to i32

                memref.store %g_thread_i32, %ht_val[%g_thread_idx] : memref<?xi32>
                memref.store %cidx_neg1, %ht_ptr[%g_thread_idx] : memref<?xindex>
    
            }
            gpu.return 
        }

        gpu.func @build(%relation1 : memref<?xi32>, %relation1_rows : index, %ht_size : index,
                %ht_val : memref<?xi32>, %ht_ptr : memref<?xindex>, %ll_key : memref<?xi32>, %ll_rowID : memref<?xindex>, 
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
                
                func.call @Insert_Node_HT(%key, %ht_val, %ht_ptr, %ll_key, %ll_rowID, %ll_next, %free_index, %g_thread_idx) 
                 : (index, memref<?xi32>, memref<?xindex>, memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<1xi32>, index) -> ()


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

        // number of rows in the first table is the num_tuples in the linked list
        %ll_key, %ll_rowID, %ll_next, %ht_val, %ht_ptr = func.call @alloc_hash_table(%relation1_rows, %ht_size) 
        : (index, index) -> ( memref<?xi32>, memref<?xindex>, memref<?xindex>, memref<?xi32>, memref<?xindex>)

        
        func.call @init_hash_table(%ht_size, %ht_val, %ht_ptr) : (index, memref<?xi32>, memref<?xindex>) -> ()


        func.call @build_table(%relation1, %relation1_rows, %ht_size, %ht_val, %ht_ptr, %ll_key, %ll_rowID, %ll_next) 
         : (memref<?xi32>, index, index, memref<?xi32>, memref<?xindex>, memref<?xi32>, memref<?xindex>, memref<?xindex>) -> ()

        // Allocate memref for prefix sum array
        %prefix = gpu.alloc(%relation2_rows) : memref<?xindex>

        %result_size = func.call @count_rows(%relation2, %relation2_rows, %ht_size, %ht_val, %ht_ptr, %prefix)
         : (memref<?xi32>, index, index, memref<?xi32>, memref<?xindex>, memref<?xindex>) -> index


        // Allocate device memory for the result
        %d_result_r = gpu.alloc(%result_size) : memref<?xindex>
        %d_result_s = gpu.alloc(%result_size) : memref<?xindex>
       

        func.call @probe_relation(%relation2, %relation2_rows, %)


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
    func.func private @printMemrefI32(memref<*xi32>)
}