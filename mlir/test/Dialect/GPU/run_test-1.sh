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

$MLIR_OPT -test-gpu-rewrite $INPUT_FILE -o ./test.mlir