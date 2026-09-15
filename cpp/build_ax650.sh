#!/usr/bin/env bash
# Cross-compile the pocket-tts-zh-en C++ runtime for AX650 (aarch64).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_DIR="${ROOT}/cpp"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-/data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out}"
ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR:-$(cd "${ROOT}" && pwd)/.work_tmp/ort/onnxruntime-1.23.0-aarch64}"
BUILD_DIR="${CPP_DIR}/build/ax650"

mkdir -p "${BUILD_DIR}" "${CPP_DIR}/bin"
cmake -S "${CPP_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" \
  -DBSP_MSP_DIR="${BSP_MSP_DIR}" -DONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cp "${BUILD_DIR}/pocket_tts_zh_en" "${CPP_DIR}/bin/"

ORT_LIB="${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.23.0"
if [[ -f "${ORT_LIB}" ]]; then
  cp -a "${ORT_LIB}" "${CPP_DIR}/bin/"
  ln -sfn "libonnxruntime.so.1.23.0" "${CPP_DIR}/bin/libonnxruntime.so.1"
fi
echo "built ${CPP_DIR}/bin/pocket_tts_zh_en"
