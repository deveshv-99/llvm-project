extern "C" int check(int* basePtr,int* alignedPtr, long offset, long sizes[], long strides[]){
    return basePtr[0];
}
