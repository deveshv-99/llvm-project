#!/bin/bash
# Original Location at:
# /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/run_test.sh
MLIR_OPT=~/llvm-project/build/bin/mlir-opt
MLIR_CPU_RUNNER=~/llvm-project/build/bin/mlir-cpu-runner
SO_DEP1=~/llvm-project/build/lib/libmlir_cuda_runtime.so
SO_DEP2=~/llvm-project/build/lib/libmlir_runner_utils.so
SO_DEP3=~/llvm-project/build/lib/libmlir_async_runtime.so
FileCheck=~/llvm-project/build/bin/FileCheck

#Input file is the first argument to the script
INPUT_FILE=$1

$MLIR_OPT  -convert-scf-to-cf $INPUT_FILE \
    | $MLIR_OPT -gpu-kernel-outlining \
    | $MLIR_OPT -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))'\
    | $MLIR_OPT -gpu-async-region -gpu-to-llvm \
    | $MLIR_CPU_RUNNER --shared-libs=$SO_DEP1 --shared-libs=$SO_DEP2 --shared-libs=$SO_DEP3 --entry-point-result=void -O0 
# | -o ./test.mlir