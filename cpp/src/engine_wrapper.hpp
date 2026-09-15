#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "ax_engine_api.h"

class EngineWrapper {
public:
    EngineWrapper();
    ~EngineWrapper();
    int Init(const char* model_path);
    int Release();
    int SetInputByName(const char* name, const void* data);
    int RunSync();
    int GetOutputByName(const char* name, void* data);
    int CopyOutputSliceToInputByName(const char* output_name, std::size_t output_offset,
                                     const char* input_name, std::size_t input_offset,
                                     std::size_t size);
    int CopyOutputToOtherInputByName(const char* output_name, EngineWrapper& target,
                                     const char* input_name);
    int GetInputSizeByName(const char* name) const;
    int GetOutputSizeByName(const char* name) const;
    int GetInputIndex(const char* name) const;
    int GetOutputIndex(const char* name) const;
private:
    bool has_init_ = false;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* info_ = nullptr;
    AX_ENGINE_IO_T io_{};
    std::unordered_map<std::string, int> inputs_;
    std::unordered_map<std::string, int> outputs_;
};
