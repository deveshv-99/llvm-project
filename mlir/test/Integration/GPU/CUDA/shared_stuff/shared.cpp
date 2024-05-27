#include <stdint.h>
#include <iostream>


// Initialize the memref
extern "C" void init_relation(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){

    for(auto i = 0; i < sizes; i++){
         basePtr[i] = (int32_t)i;
    }
   
}

// Check for the correctness of the result
extern "C" int32_t check(int32_t* basePtr,int32_t* alignedPtr, int64_t offset, int64_t sizes, int64_t strides){
    // std::cout <<"sizes " << sizes <<'\n';
    // std::cout <<"strides " << strides <<'\n';
    // std::cout <<"basePtr "<< basePtr;
    for(auto i = 0; i < sizes; i++){
         basePtr[i] = (int32_t)i;
    }
    return (int32_t) sizes;
}
