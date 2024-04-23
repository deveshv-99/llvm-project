// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/nested-loop.mlir

module attributes {gpu.container_module} {

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

    // gpu.module @kernels {

    //     func.func @build_hash_table(%table_x : memref<?x?xi32>, %table_x_rows : index) -> memref<?x?xi32> {
    //         // Constants
    //         %cidx_0 = arith.constant 0 : index
    //         %cidx_1 = arith.constant 1 : index

    //         // Number of different hash values is kept 100 for now
    //         %ht_size = arith.constant 100 : index

    //         %hash_table = func.call @Init_hash_table(%ht_size, %table_x_rows) : (index, index) -> memref<?x?xi32>
            
    //         %key = memref.
    //         scf.for %i = %cidx_0 to %table_x_rows step %cidx_1 {
    //             %hash_val = func.call @hash(%key) : (i32) -> i32
    //             func.call @Insert_Node_HT(%hash_table, %hash_val, %key, %val) : (memref<?x?xi32>, i32, i32,i32) -> ()
    //         }

    //         return %hash_table : memref<?x?xi32>
    //     }

    //     // Kernel to perform nested join
    //     gpu.func @hash_join (%table_x : memref<?x?xi32>, %table_y : memref<?x?xi32>, %d_result : memref<?x?xi32>, %table_x_rows : index, 
    //         %table_x_cols : index, %table_y_rows : index, %table_y_cols : index, %gblock_offset : memref<1xi32>) 
    //         //---------------> Size of shared memory is fixed for now. To be changed later 
    //         workgroup(%thread_sums : memref<1024xindex, 3>, %b_block_offset : memref<1xindex, 3>) 
    //         private(%temp_idx: memref<1xindex>)
    //         kernel 
    //     {
            
    //         %bdim = gpu.block_dim x
    //         %bidx = gpu.block_id x
    //         %tidx = gpu.thread_id x

    //         // Global_thread_index = bdim * bidx + tidx
    //         %g_thread_offset_in_blocks = arith.muli %bdim, %bidx : index
    //         %g_thread_idx = arith.addi %g_thread_offset_in_blocks, %tidx : index

    //         // Check if the thread is valid
    //         %is_thread_valid = arith.cmpi "ult", %g_thread_idx, %table_x_rows : index

    //         scf.if %is_thread_valid {
    //             gpu.printf "Thread ID: %lld \n" %tidx : index
                
    //             // print debugging constants
    //             %print_thread_id = arith.constant 0: index
    //             %print_block_id = arith.constant 0: index

    //             %should_print_thread = arith.cmpi "eq", %tidx, %print_thread_id : index
    //             %should_print_block = arith.cmpi "eq", %bidx, %print_block_id : index
    //             %should_print = arith.andi %should_print_thread, %should_print_block : i1

    //             scf.if %should_print {
    //                 gpu.printf "Block ID: %ld, Thread ID: %ld, bdim: %ld\n" %bidx, %tidx, %bdim : index, index, index
    //             }

    //             //constants
    //             %cidx_0 = arith.constant 0 : index
    //             %cidx_1 = arith.constant 1 : index
    //             %cidx_2 = arith.constant 2 : index


    //         }
            
    //         gpu.return
    //     }
    // }
    func.func @hash(%key : i32) -> i32{
        // return modulo 100
        %cidx_100 = arith.constant 100 : i32
        %hash_val = arith.remui %key, %cidx_100 : i32
        return %hash_val : i32
    }
    
    func.func @main() {

        // Constants
        %cidx_0 = arith.constant 0 : index
        %cidx_1 = arith.constant 1 : index
        %cidx_2 = arith.constant 2 : index
        %cidx_1024 = arith.constant 1024 : index

        %ci32_0 = arith.constant 0 : i32
        %ci32_1 = arith.constant 1 : i32
        %ci32_2 = arith.constant 2 : i32
        %ci32_32 = arith.constant 32 : i32
        %ci32_1024 = arith.constant 1024 : i32

        %val = llvm.mlir.constant (43 : i32) : i32
        %val1 = llvm.mlir.constant (44 : i32) : i32
        %c0 = llvm.mlir.constant (0 : i32) : i32
        %c1 = llvm.mlir.constant (1 : i32) : i32




        // use alloca to create array of 2 elements
        %a2 = llvm.alloca %ci32_2 x i32 : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr1 = llvm.getelementptr %a2[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)
        %ptr2 = llvm.getelementptr %a2[%c1] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<i32>)

        //insert value at this pointer
        llvm.store %val, %ptr1 : !llvm.ptr< i32>
        llvm.store %val1, %ptr2 : !llvm.ptr< i32>

        %value1 = llvm.load %ptr1 : !llvm.ptr<i32>
        %value2 = llvm.load %ptr2 : !llvm.ptr<i32>

        call @debugI32(%value1) : (i32) -> ()
        call @debugI32(%value2) : (i32) -> ()

   
        %x = llvm.alloca %ci32_1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr<array<2 x i32>>
        %ptr = llvm.getelementptr %x[%c0] : (!llvm.ptr<array<2 x i32>> ,i32) -> (!llvm.ptr<array<2 x i32>>)
        %ptptr = llvm.getelementptr %x[%c0, %c0] : (!llvm.ptr<array<2 x i32>> ,i32, i32) -> (!llvm.ptr<i32>)

        %valuee = llvm.load %ptptr : !llvm.ptr<i32>
        call @debugI32(%valuee) : (i32) -> ()

        // %linked_list =llvm.mlir.undef: !llvm.struct<(i32, i32, ptr)>
        // //%linked_list = llvm.alloca %cidx_1024 x !llvm.struct<(i32, i32, ptr)> :(index) -> !llvm.ptr<<struct<(i32, i32, ptr)>>

        // //%hash_table = llvm.alloca %cidx_1024 x !llvm.struct<(i32,ptr)> : (index) -> !llvm.ptr<struct<(i32,ptr)>>

        // %ptr1 = builtin.unrealized_conversion_cast %ci32_32 : i32 to !llvm.ptr

        // %1 = llvm.insertvalue %ci32_1, %linked_list[0] : !llvm.struct<(i32, i32, ptr)>
        // %2 = llvm.insertvalue %ci32_2, %1[1] : !llvm.struct<(i32, i32, ptr)>
        // %3 = llvm.insertvalue %ptr1, %2[2] : !llvm.struct<(i32, i32, ptr)>

        // //convert list to memref
        // %yy = llvm.load %ptr : !llvm.ptr<array<2 x i32>>
        // %yc = builtin.unrealized_conversion_cast %yy : !llvm.array<2 x i32> to memref<2xi32>                                                                                                                                                                                           

        // //%first_memref = builtin.unrealized_conversion_cast %first : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>
        
        // %y = memref.alloc() : memref<1xi32>
        // //memref.store %yy, %y[%cidx_0] : memref<1xi32>

        // %dst = memref.cast %yc: memref<2xi32> to memref<*xi32>
        // call @printMemrefI32(%dst) : (memref<*xi32>) -> ()

        return
    }
    
    func.func private @printMemrefI32(memref<*xi32>)
    func.func private @printMemrefF32(memref<*xf32>)
}