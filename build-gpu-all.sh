#!/usr/bin/env bash
set -e

NUM_THREADS=$(nproc)

##############################################
# 1. 编译 GPU 版 BIN（可执行文件 + benchmark）
##############################################
echo "====== Building BIN (CUDA) ======"
mkdir -p Linux-BIN-CUDA
pushd Linux-BIN-CUDA
cmake -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_BUILD_TYPE=Release \
      -DOCR_OUTPUT=BIN \
      -DOCR_ONNX=CUDA \
      ..
cmake --build . --config Release -j "$NUM_THREADS"
popd

##############################################
# 2. 编译 GPU 版 CLIB（.so 动态库）
##############################################
echo "====== Building CLIB / .so (CUDA) ======"
mkdir -p Linux-CLIB-CUDA
pushd Linux-CLIB-CUDA
cmake -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_BUILD_TYPE=Release \
      -DOCR_OUTPUT=CLIB \
      -DOCR_ONNX=CUDA \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      ..
cmake --build . --config Release -j "$NUM_THREADS"
cmake --build . --config Release --target install
popd

echo ""
echo "====== Build Complete ======"
echo "BIN:  Linux-BIN-CUDA/RapidOcrOnnx"
echo "SO:   Linux-CLIB-CUDA/libRapidOcrOnnx.so"
echo "Headers: Linux-CLIB-CUDA/install/include/"
