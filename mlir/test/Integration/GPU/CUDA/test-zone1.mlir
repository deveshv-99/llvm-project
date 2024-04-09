module attributes {gpu.container_module} {
  func.func @main() {
    %one = arith.constant 1 : index    
    %sx = arith.constant 12 : index

    gpu.launch_func  @main_kernel::@empty_kernel blocks in (%one, %one, %one) threads in (%sx, %one, %one) args()
    %token_1 = gpu.wait async

    gpu.launch_func  @main_kernel::@empty_kernel blocks in (%one, %one, %one) threads in (%sx, %one, %one) args()
    %token_2 = gpu.wait async

    return
  }

 gpu.module @main_kernel {
    gpu.func @empty_kernel() kernel {
      gpu.return
    }
  }
}