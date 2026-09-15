#include "engine_wrapper.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "ax_sys_api.h"

namespace {
constexpr int kAlign = 128;

bool read_model(const char* path, AX_VOID** vir, AX_U64& phy, AX_U32& size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    size = static_cast<AX_U32>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (AX_SYS_MemAlloc(&phy, vir, size, kAlign, reinterpret_cast<AX_S8*>(const_cast<char*>("POCKET-MODEL"))) != 0) return false;
    in.read(reinterpret_cast<char*>(*vir), size);
    return static_cast<bool>(in);
}

void free_io(AX_ENGINE_IO_T& io) {
    for (AX_U32 i = 0; io.pInputs && i < io.nInputSize; ++i)
        if (io.pInputs[i].pVirAddr) AX_SYS_MemFree(io.pInputs[i].phyAddr, io.pInputs[i].pVirAddr);
    for (AX_U32 i = 0; io.pOutputs && i < io.nOutputSize; ++i)
        if (io.pOutputs[i].pVirAddr) AX_SYS_MemFree(io.pOutputs[i].phyAddr, io.pOutputs[i].pVirAddr);
    delete[] io.pInputs;
    delete[] io.pOutputs;
    std::memset(&io, 0, sizeof(io));
}
}  // namespace

EngineWrapper::EngineWrapper() = default;
EngineWrapper::~EngineWrapper() { Release(); }

int EngineWrapper::Init(const char* path) {
    AX_VOID* model = nullptr; AX_U64 phy = 0; AX_U32 size = 0;
    if (!read_model(path, &model, phy, size)) return -1;
    AX_S32 ret = AX_ENGINE_CreateHandle(&handle_, model, size);
    AX_SYS_MemFree(phy, model);
    if (ret != 0 || !handle_) return -1;
    if (AX_ENGINE_CreateContext(handle_) != 0 || AX_ENGINE_GetIOInfo(handle_, &info_) != 0 || !info_) { Release(); return -1; }
    io_.nInputSize = info_->nInputSize; io_.nOutputSize = info_->nOutputSize;
    io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{};
    io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
    for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
        io_.pInputs[i].nSize = info_->pInputs[i].nSize;
        if (AX_SYS_MemAlloc(&io_.pInputs[i].phyAddr, &io_.pInputs[i].pVirAddr, io_.pInputs[i].nSize, kAlign, reinterpret_cast<AX_S8*>(const_cast<char*>("POCKET-IN"))) != 0) { Release(); return -1; }
        inputs_[info_->pInputs[i].pName] = static_cast<int>(i);
    }
    for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
        io_.pOutputs[i].nSize = info_->pOutputs[i].nSize;
        if (AX_SYS_MemAlloc(&io_.pOutputs[i].phyAddr, &io_.pOutputs[i].pVirAddr, io_.pOutputs[i].nSize, kAlign, reinterpret_cast<AX_S8*>(const_cast<char*>("POCKET-OUT"))) != 0) { Release(); return -1; }
        outputs_[info_->pOutputs[i].pName] = static_cast<int>(i);
    }
    has_init_ = true;
    std::printf("loaded %s inputs=%u outputs=%u\n", path, io_.nInputSize, io_.nOutputSize);
    return 0;
}

int EngineWrapper::Release() {
    free_io(io_);
    if (handle_) AX_ENGINE_DestroyHandle(handle_);
    handle_ = nullptr; info_ = nullptr; has_init_ = false; inputs_.clear(); outputs_.clear();
    return 0;
}
int EngineWrapper::GetInputIndex(const char* n) const { auto i=inputs_.find(n); return i==inputs_.end()?-1:i->second; }
int EngineWrapper::GetOutputIndex(const char* n) const { auto i=outputs_.find(n); return i==outputs_.end()?-1:i->second; }
int EngineWrapper::GetInputSizeByName(const char* n) const { int i=GetInputIndex(n); return i<0?-1:static_cast<int>(io_.pInputs[i].nSize); }
int EngineWrapper::GetOutputSizeByName(const char* n) const { int i=GetOutputIndex(n); return i<0?-1:static_cast<int>(io_.pOutputs[i].nSize); }
int EngineWrapper::SetInputByName(const char* n, const void* p) { int i=GetInputIndex(n); if (!has_init_||i<0||!p) return -1; std::memcpy(io_.pInputs[i].pVirAddr,p,io_.pInputs[i].nSize); return 0; }
int EngineWrapper::RunSync() {
    if (!has_init_) return -1;
    const AX_S32 ret=AX_ENGINE_RunSync(handle_, &io_);
    if (ret!=0) std::fprintf(stderr,"AX_ENGINE_RunSync failed ret=0x%x\\n",static_cast<unsigned int>(ret));
    return ret;
}
int EngineWrapper::GetOutputByName(const char* n, void* p) { int i=GetOutputIndex(n); if (!has_init_||i<0||!p) return -1; std::memcpy(p,io_.pOutputs[i].pVirAddr,io_.pOutputs[i].nSize); return 0; }
int EngineWrapper::CopyOutputSliceToInputByName(const char* on, std::size_t oo, const char* in,
                                                std::size_t io, std::size_t size) {
    const int oi=GetOutputIndex(on), ii=GetInputIndex(in);
    if(!has_init_||oi<0||ii<0||oo>io_.pOutputs[oi].nSize||size>io_.pOutputs[oi].nSize-oo||io>io_.pInputs[ii].nSize||size>io_.pInputs[ii].nSize-io)return -1;
    std::memcpy(reinterpret_cast<uint8_t*>(io_.pInputs[ii].pVirAddr)+io,
                reinterpret_cast<const uint8_t*>(io_.pOutputs[oi].pVirAddr)+oo,size); return 0;
}
int EngineWrapper::CopyOutputToOtherInputByName(const char* on, EngineWrapper& target, const char* in) {
    const int oi=GetOutputIndex(on),ii=target.GetInputIndex(in);
    if(!has_init_||oi<0||ii<0||io_.pOutputs[oi].nSize!=target.io_.pInputs[ii].nSize)return -1;
    return target.SetInputByName(in,io_.pOutputs[oi].pVirAddr);
}
