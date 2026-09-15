// pocket-tts-zh-en AX650 hybrid C++ runtime.
//
// Pipeline (identical to board/pocket_tts_axera.py):
//   voice enc (NPU tier) -> flow prefill (ORT int8) x {voice, text}
//   -> loop { flow AR (ORT int8) -> flow_net (NPU fp32)
//             -> mimi transformer (ORT int8) -> mimi conv (NPU U16) }
//
// Text is tokenized on the host (see python/prepare_tokens.py) into a request
// file: one line per chunk, comma-separated token ids. Reference wav must be
// 24 kHz 16-bit mono PCM.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "engine_wrapper.hpp"
#include "onnxruntime_cxx_api.h"

namespace {

constexpr int kSampleRate = 24000;
constexpr int kFrameSize = 1920;
constexpr int kLatentDim = 32;
constexpr int kModelDim = 1024;
constexpr int kFlowLayers = 6;
constexpr int kFlowHeads = 16;
constexpr int kFlowHeadDim = 64;
constexpr int kMimiLayers = 2;
constexpr int kMimiHeads = 8;
constexpr int kMimiHeadDim = 64;
constexpr int kMimiKvLen = 266;
constexpr int kStepsPerLatent = 16;
constexpr int kConvState = 14720;
constexpr int kFlowRow = kFlowLayers * 2 * 1 * kFlowHeads * kFlowHeadDim;  // 12288
constexpr int kMimiRow = kMimiLayers * 2 * 1 * kMimiHeads * kMimiHeadDim;  // 2048
constexpr int kUpsampleState = 8192;
constexpr int kEncoderFrames = 40;                 // NPU tier, 3.2 s
constexpr float kEosThreshold = -1.0f;
constexpr int kFlowWindow = 512;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Cache {
    std::vector<float> data;
    size_t length = 0;  // rows
    int row = 0;
    void reserve(size_t rows) { data.reserve(rows * row); }
    void append(const float* src, size_t rows) {
        data.insert(data.end(), src, src + rows * row);
        length += rows;
    }
    void truncate(size_t rows) { length = rows; data.resize(rows * row); }
};

struct OrtGraph {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "pocket-tts-zh-en"};
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_ptrs;
    std::vector<const char*> output_ptrs;
    Ort::MemoryInfo mem{nullptr};

    void load(const std::string& path, int threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            input_names.emplace_back(session->GetInputNameAllocated(i, alloc).get());
        }
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            output_names.emplace_back(session->GetOutputNameAllocated(i, alloc).get());
        }
        for (auto& n : input_names) input_ptrs.push_back(n.c_str());
        for (auto& n : output_names) output_ptrs.push_back(n.c_str());
        mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }

    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs) {
        return session->Run(Ort::RunOptions{nullptr}, input_ptrs.data(), inputs.data(),
                            inputs.size(), output_ptrs.data(), output_ptrs.size());
    }
};

std::vector<float> read_ref_wav(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open reference wav: " + path);
    in.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<char> bytes(size);
    in.read(bytes.data(), size);
    if (size < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0)
        throw std::runtime_error("invalid wav: " + path);
    const char* p = bytes.data() + 12;
    const char* end = bytes.data() + size;
    int channels = 1, bits = 16, rate = 0;
    const char* data_ptr = nullptr;
    size_t data_bytes = 0;
    while (p + 8 <= end) {
        char id[5] = {0};
        std::memcpy(id, p, 4);
        uint32_t chunk = 0;
        std::memcpy(&chunk, p + 4, 4);
        const char* body = p + 8;
        if (std::strcmp(id, "fmt ") == 0 && body + 16 <= end) {
            uint16_t ch = 0, b = 0;
            std::memcpy(&ch, body + 2, 2);
            std::memcpy(&rate, body + 4, 4);
            std::memcpy(&b, body + 14, 2);
            channels = ch;
            bits = b;
        } else if (std::strcmp(id, "data") == 0) {
            data_ptr = body;
            data_bytes = std::min<size_t>(chunk, static_cast<size_t>(end - body));
        }
        p = body + chunk + (chunk & 1);
    }
    if (!data_ptr || rate != kSampleRate || bits != 16)
        throw std::runtime_error("reference wav must be 24kHz 16-bit PCM");
    if (channels < 1) throw std::runtime_error("bad channel count");
    size_t n = data_bytes / (2 * channels);
    std::vector<float> mono(n);
    const int16_t* src = reinterpret_cast<const int16_t*>(data_ptr);
    for (size_t i = 0; i < n; ++i) {
        float acc = 0.f;
        for (int c = 0; c < channels; ++c) acc += src[i * channels + c] / 32768.0f;
        mono[i] = acc / channels;
    }
    return mono;
}

std::vector<std::vector<int64_t>> read_request(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tokens file: " + path);
    std::vector<std::vector<int64_t>> chunks;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<int64_t> ids;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            ids.push_back(std::stoll(item));
        }
        if (!ids.empty()) chunks.push_back(std::move(ids));
    }
    if (chunks.empty()) throw std::runtime_error("empty tokens file");
    return chunks;
}

struct WavWriter {
    std::ofstream out;
    uint64_t samples = 0;
    void open(const std::string& path) {
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + path);
        write_header(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    void write_header(uint32_t riff_size, uint32_t data_size) {
        out.seekp(0);
        const char* riff = "RIFF";
        out.write(riff, 4);
        out.write(reinterpret_cast<const char*>(&riff_size), 4);
        const char* wave = "WAVEfmt ";
        out.write(wave, 8);
        uint32_t fmt_size = 16;
        out.write(reinterpret_cast<const char*>(&fmt_size), 4);
        uint16_t audio_fmt = 1, channels = 1, block_align = 2, bits = 16;
        uint32_t byte_rate = kSampleRate * 2;
        out.write(reinterpret_cast<const char*>(&audio_fmt), 2);
        out.write(reinterpret_cast<const char*>(&channels), 2);
        out.write(reinterpret_cast<const char*>(&kSampleRate), 4);
        out.write(reinterpret_cast<const char*>(&byte_rate), 4);
        out.write(reinterpret_cast<const char*>(&block_align), 2);
        out.write(reinterpret_cast<const char*>(&bits), 2);
        const char* data = "data";
        out.write(data, 4);
        out.write(reinterpret_cast<const char*>(&data_size), 4);
    }
    void write(const float* audio, size_t n) {
        std::vector<int16_t> pcm(n);
        for (size_t i = 0; i < n; ++i) {
            float v = std::max(-1.0f, std::min(1.0f, audio[i]));
            pcm[i] = static_cast<int16_t>(std::lround(v * 32767.0f));
        }
        out.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * 2);
        out.flush();
        samples += n;
    }
    void close() {
        uint32_t data_bytes = static_cast<uint32_t>(samples * 2);
        write_header(36 + data_bytes, data_bytes);
        out.close();
    }
};

struct Options {
    std::string models_dir;
    std::string reference;
    std::string tokens_file;
    std::string output;
    int threads = 4;
    int prefill_threads = 8;
    int max_frames = 375;
    float temp = 0.0f;
    int seed = 0;
    int pause_ms = 120;
    int streaming = 1;
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--models-dir") opt.models_dir = next();
        else if (a == "--reference") opt.reference = next();
        else if (a == "--tokens-file") opt.tokens_file = next();
        else if (a == "--output") opt.output = next();
        else if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--prefill-threads") opt.prefill_threads = std::stoi(next());
        else if (a == "--max-frames") opt.max_frames = std::stoi(next());
        else if (a == "--temp") opt.temp = std::stof(next());
        else if (a == "--seed") opt.seed = std::stoi(next());
        else if (a == "--pause-ms") opt.pause_ms = std::stoi(next());
        else if (a == "--stream") opt.streaming = std::stoi(next());
        else {
            std::cerr << "unknown arg: " << a << "\n";
            return 2;
        }
    }
    if (opt.models_dir.empty() || opt.reference.empty() || opt.tokens_file.empty() ||
        opt.output.empty()) {
        std::cerr << "usage: pocket_tts_zh_en --models-dir DIR --reference wav "
                     "--tokens-file f --output out.wav [--threads 4] [--prefill-threads 8] "
                     "[--max-frames 375] [--temp 0] [--pause-ms 120] [--stream 1]\n";
        return 2;
    }

    const std::string md = opt.models_dir;
    const auto t_load0 = Clock::now();
    OrtGraph flow, flow_ar, mimi_tf;
    flow.load(md + "/flow_step_int8.onnx", opt.prefill_threads);
    flow_ar.load(md + "/flow_ar_step_int8.onnx", opt.threads);
    mimi_tf.load(md + "/mimi_split/mimi_transformer_step_int8.onnx", opt.threads);

    EngineWrapper enc, flow_net, mimi_conv;
    if (enc.Init((md + "/encoder/step_encoder_40f.axmodel").c_str()) != 0)
        throw std::runtime_error("failed to load encoder axmodel");
    if (flow_net.Init((md + "/flow_net_step_fp32.axmodel").c_str()) != 0)
        throw std::runtime_error("failed to load flow_net axmodel");
    if (mimi_conv.Init((md + "/mimi_conv_step.axmodel").c_str()) != 0)
        throw std::runtime_error("failed to load mimi_conv axmodel");

    auto ref = read_ref_wav(opt.reference);
    auto chunks = read_request(opt.tokens_file);
    std::printf("load %.1fms, reference %.2fs, chunks %zu\n", ms_since(t_load0),
                ref.size() / double(kSampleRate), chunks.size());

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> zero_holder(1, 0.f);
    auto tensor_f32 = [&](std::vector<int64_t> shape, void* data) {
        int64_t count = std::accumulate(shape.begin(), shape.end(), 1LL,
                                        std::multiplies<int64_t>());
        float* ptr = static_cast<float*>(data);
        if (count == 0) {
            ptr = zero_holder.data();
            count = 0;
        }
        return Ort::Value::CreateTensor<float>(mem, ptr, static_cast<size_t>(count),
                                               shape.data(), shape.size());
    };
    auto tensor_i64 = [&](std::vector<int64_t> shape, void* data) {
        return Ort::Value::CreateTensor<int64_t>(mem, static_cast<int64_t*>(data),
                                                 static_cast<size_t>(std::accumulate(
                                                     shape.begin(), shape.end(), 1LL,
                                                     std::multiplies<int64_t>())),
                                                 shape.data(), shape.size());
    };

    // ---------------- voice conditioning (NPU tier 40f) ----------------
    const size_t tier_samples = size_t(kEncoderFrames) * kFrameSize;
    std::vector<float> clip(tier_samples, 0.0f);
    if (ref.size() >= tier_samples) {
        std::copy(ref.end() - tier_samples, ref.end(), clip.begin());
    } else {
        std::copy(ref.begin(), ref.end(), clip.begin() + (tier_samples - ref.size()));
    }
    if (enc.SetInputByName("audio", clip.data()) != 0 || enc.RunSync() != 0)
        throw std::runtime_error("encoder run failed");
    const int cond_bytes = enc.GetOutputSizeByName("cond");
    const int cond_frames = cond_bytes / (kModelDim * 4);
    std::vector<float> cond(cond_bytes / 4);
    enc.GetOutputByName("cond", cond.data());

    // ---------------- prefill voice into a flow KV prefix ----------------
    Cache flow_kv;
    flow_kv.row = kFlowRow;
    flow_kv.reserve(kFlowWindow);
    {
        std::vector<int64_t> tokens(cond_frames, 0);
        std::vector<float> latent(size_t(cond_frames) * kLatentDim, 0.f);
        std::vector<float> is_bos(cond_frames, 0.f);
        std::vector<float> gates{0.f, 0.f, 1.f};
        int64_t offset = 0;
        auto tokens_t = tensor_i64({1, cond_frames}, tokens.data());
        auto latent_t = tensor_f32({1, cond_frames, kLatentDim}, latent.data());
        auto bos_t = tensor_f32({1, cond_frames, 1}, is_bos.data());
        auto cond_t = tensor_f32({1, cond_frames, kModelDim}, cond.data());
        auto gates_t = tensor_f32({3}, gates.data());
        auto kv_t = tensor_f32({0, kFlowLayers, 2, 1, kFlowHeads, kFlowHeadDim},
                               flow_kv.data.data());
        auto off_t = tensor_i64({}, &offset);
        std::vector<Ort::Value> feed;
        feed.push_back(std::move(tokens_t));
        feed.push_back(std::move(latent_t));
        feed.push_back(std::move(bos_t));
        feed.push_back(std::move(cond_t));
        feed.push_back(std::move(gates_t));
        feed.push_back(std::move(kv_t));
        feed.push_back(std::move(off_t));
        auto out = flow.run(feed);
        auto* kv_new = out[2].GetTensorData<float>();
        flow_kv.append(kv_new, cond_frames);
    }
    Cache voice_prefix = flow_kv;  // snapshot for chunk reuse

    // ---------------- per-chunk synthesis ----------------
    WavWriter writer;
    writer.open(opt.output);
    double total_ms = 0.0, first_frame_ms = -1.0;
    long frames_total = 0, samples_total = 0;
    const int pause_samples = opt.pause_ms * kSampleRate / 1000;
    std::vector<float> silence(pause_samples, 0.f);
    std::mt19937 rng(opt.seed);
    std::normal_distribution<float> normal(0.f, 1.f);

    std::vector<float> audio_buf(kFrameSize);
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const auto chunk_start = Clock::now();
        const auto& ids = chunks[ci];
        const int seq = static_cast<int>(ids.size());
        flow_kv = voice_prefix;
        flow_kv.reserve(kFlowWindow + seq + opt.max_frames);
        {
            std::vector<float> latent(size_t(seq) * kLatentDim, 0.f);
            std::vector<float> is_bos(seq, 0.f);
            std::vector<float> cond_zero(size_t(seq) * kModelDim, 0.f);
            std::vector<float> gates{1.f, 0.f, 0.f};
            int64_t offset = static_cast<int64_t>(flow_kv.length);
            auto tokens_t = tensor_i64({1, seq}, const_cast<int64_t*>(ids.data()));
            auto latent_t = tensor_f32({1, seq, kLatentDim}, latent.data());
            auto bos_t = tensor_f32({1, seq, 1}, is_bos.data());
            auto cond_t = tensor_f32({1, seq, kModelDim}, cond_zero.data());
            auto gates_t = tensor_f32({3}, gates.data());
            auto kv_t = tensor_f32({static_cast<int64_t>(flow_kv.length), kFlowLayers, 2, 1,
                                    kFlowHeads, kFlowHeadDim}, flow_kv.data.data());
            auto off_t = tensor_i64({}, &offset);
            std::vector<Ort::Value> feed;
            feed.push_back(std::move(tokens_t));
            feed.push_back(std::move(latent_t));
            feed.push_back(std::move(bos_t));
            feed.push_back(std::move(cond_t));
            feed.push_back(std::move(gates_t));
            feed.push_back(std::move(kv_t));
            feed.push_back(std::move(off_t));
            auto out = flow.run(feed);
            flow_kv.append(out[2].GetTensorData<float>(), seq);
        }

        std::vector<float> latent(kLatentDim, 0.f);
        std::vector<float> is_bos{1.f};
        Cache mimi_kv;
        mimi_kv.row = kMimiRow;
        mimi_kv.reserve(kMimiKvLen + opt.max_frames * kStepsPerLatent);
        std::vector<float> mimi_conv_state(kConvState, 0.f);
        int64_t mimi_offset = 0;

        for (int frame = 0; frame < opt.max_frames; ++frame) {
            const auto t_frame = Clock::now();
            std::vector<float> noise(kLatentDim, 0.f);
            if (opt.temp > 0.f) {
                for (auto& v : noise) v = normal(rng) * std::sqrt(opt.temp);
            }

            // flow AR
            size_t win = flow_kv.length > kFlowWindow ? kFlowWindow : flow_kv.length;
            const float* kv_ptr = flow_kv.data.data() + (flow_kv.length - win) * kFlowRow;
            std::vector<int64_t> tokens0(1, 0);
            std::vector<float> cond_zero(kModelDim, 0.f);
            std::vector<float> gates_ar{0.f, 1.f, 0.f};
            int64_t offset = static_cast<int64_t>(flow_kv.length);
            auto latent_t = tensor_f32({1, 1, kLatentDim}, latent.data());
            auto bos_t = tensor_f32({1, 1, 1}, is_bos.data());
            auto tokens_t = tensor_i64({1, 1}, tokens0.data());
            auto cond0_t = tensor_f32({1, 1, kModelDim}, cond_zero.data());
            auto gates_t = tensor_f32({3}, gates_ar.data());
            auto kv_t = tensor_f32({static_cast<int64_t>(win), kFlowLayers, 2, 1,
                                    kFlowHeads, kFlowHeadDim}, const_cast<float*>(kv_ptr));
            auto off_t = tensor_i64({}, &offset);
            std::vector<Ort::Value> feed;
            feed.push_back(std::move(tokens_t));
            feed.push_back(std::move(latent_t));
            feed.push_back(std::move(bos_t));
            feed.push_back(std::move(cond0_t));
            feed.push_back(std::move(gates_t));
            feed.push_back(std::move(kv_t));
            feed.push_back(std::move(off_t));
            auto out = flow_ar.run(feed);
            const float* conditioning = out[0].GetTensorData<float>();
            const float eos_logit = out[1].GetTensorData<float>()[0];
            flow_kv.append(out[2].GetTensorData<float>(), 1);

            // flow net (NPU FP32)
            if (flow_net.SetInputByName("conditioning", conditioning) != 0 ||
                flow_net.SetInputByName("noise", noise.data()) != 0 || flow_net.RunSync() != 0)
                throw std::runtime_error("flow_net run failed");
            std::vector<float> next_latent(kLatentDim);
            flow_net.GetOutputByName("next_latent", next_latent.data());

            // mimi transformer (ORT int8)
            size_t mkv_len = std::min<size_t>(mimi_kv.length, kMimiKvLen);
            std::vector<float> mkv_window(size_t(kMimiKvLen) * kMimiRow, 0.f);
            if (mkv_len > 0) {
                std::copy(mimi_kv.data.end() - mkv_len * kMimiRow, mimi_kv.data.end(),
                          mkv_window.end() - mkv_len * kMimiRow);
            }
            std::vector<int64_t> next_shape{1, 1, kLatentDim};
            auto latent2_t = tensor_f32(next_shape, next_latent.data());
            auto mkv_t = tensor_f32({kMimiKvLen, kMimiLayers, 2, 1, kMimiHeads, kMimiHeadDim},
                                    mkv_window.data());
            auto mconv_t = tensor_f32({kConvState}, mimi_conv_state.data());
            auto moff_t = tensor_i64({}, &mimi_offset);
            std::vector<Ort::Value> tf_feed;
            tf_feed.push_back(std::move(latent2_t));
            tf_feed.push_back(std::move(mkv_t));
            tf_feed.push_back(std::move(mconv_t));
            tf_feed.push_back(std::move(moff_t));
            auto tf_out = mimi_tf.run(tf_feed);
            std::vector<float> decoder_embedding(512 * 16);
            std::vector<float> upsample_state(kUpsampleState);
            std::copy(tf_out[0].GetTensorData<float>(),
                      tf_out[0].GetTensorData<float>() + decoder_embedding.size(),
                      decoder_embedding.data());
            std::copy(tf_out[2].GetTensorData<float>(),
                      tf_out[2].GetTensorData<float>() + upsample_state.size(),
                      upsample_state.data());
            mimi_kv.append(tf_out[1].GetTensorData<float>(), kStepsPerLatent);

            // mimi conv (NPU U16)
            if (mimi_conv.SetInputByName("decoder_embedding", decoder_embedding.data()) != 0 ||
                mimi_conv.SetInputByName("mimi_conv", mimi_conv_state.data()) != 0 ||
                mimi_conv.SetInputByName("upsample_state", upsample_state.data()) != 0 ||
                mimi_conv.RunSync() != 0)
                throw std::runtime_error("mimi_conv run failed");
            mimi_conv.GetOutputByName("audio", audio_buf.data());
            mimi_conv.GetOutputByName("mimi_conv_out", mimi_conv_state.data());
            mimi_offset += kStepsPerLatent;
            latent = next_latent;
            is_bos[0] = 0.f;
            const double frame_ms = ms_since(t_frame);
            if (first_frame_ms < 0) first_frame_ms = ms_since(chunk_start);
            total_ms += frame_ms;
            if (eos_logit > kEosThreshold) break;
            frames_total += 1;
            samples_total += kFrameSize;
            writer.write(audio_buf.data(), kFrameSize);
        }
        if (ci + 1 < chunks.size() && pause_samples > 0) {
            writer.write(silence.data(), silence.size());
            samples_total += pause_samples;
        }
    }
    writer.close();
    const double audio_s = samples_total / double(kSampleRate);
    std::printf("frames=%ld audio=%.2fs first_frame=%.1fms total=%.2fs RTF=%.4f\n",
                frames_total, audio_s, first_frame_ms, total_ms / 1000.0,
                total_ms / 1000.0 / audio_s);
    return 0;
}
