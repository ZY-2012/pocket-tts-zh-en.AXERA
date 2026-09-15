cmake_minimum_required(VERSION 3.13)
project(pocket_tts_zh_en_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(BSP_MSP_DIR "/data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out" CACHE PATH "AX650 BSP msp/out")
set(ONNXRUNTIME_DIR "/data/shared/yangrongzhao/onnxruntime-linux-aarch64-1.14.0"
    CACHE PATH "aarch64 ONNX Runtime root")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include" CACHE PATH "ONNX Runtime include root")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" CACHE FILEPATH "ONNX Runtime shared library")
if(NOT EXISTS "${BSP_MSP_DIR}/include/ax_engine_api.h")
    message(FATAL_ERROR "BSP_MSP_DIR is invalid: ${BSP_MSP_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "ONNXRUNTIME_DIR is invalid: ${ONNXRUNTIME_DIR}")
endif()

add_executable(pocket_tts_zh_en
    pocket_tts_zh_en.cpp
    src/engine_wrapper.cpp
)
target_include_directories(pocket_tts_zh_en PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${BSP_MSP_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_directories(pocket_tts_zh_en PRIVATE ${BSP_MSP_DIR}/lib)
target_compile_definitions(pocket_tts_zh_en PRIVATE AX650)
target_compile_options(pocket_tts_zh_en PRIVATE -O3 -fvisibility=hidden -fdata-sections -ffunction-sections)
target_link_libraries(pocket_tts_zh_en PRIVATE ax_sys ax_engine ax_interpreter ${ONNXRUNTIME_LIBRARY} pthread dl m)
set_target_properties(pocket_tts_zh_en PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
target_link_options(pocket_tts_zh_en PRIVATE -Wl,--enable-new-dtags -Wl,--allow-shlib-undefined)
