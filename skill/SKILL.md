---
name: pocket-tts-zh-en-axera
description: 从「供应商导出的融合 ONNX」出发，把 pocket-tts 中英双语语音合成（支持声音克隆）适配到 AX650 板端的完整流程——ONNX 拆图、flow KV 窗口补丁、mimi 拆分子图、int8 量化、静态编码器档位、Pulsar2 量化、混合 NPU/CPU 运行时、C++ 运行时加速、回环 CER 评测与 GitHub/HF 双仓发布。当用户要做「只有 ONNX、没有 PyTorch 权重」的模型上板（尤其 TTS/带 KV cache 的自回归模型），或维护 pocket-tts-zh-en.AXERA 工程时使用。与旧 skill pocket-tts-axera-workflow（从 PyTorch 源权重重导出）不同：本流程全程 ONNX 图手术。
---

# Pocket-TTS 中英双语 → AX650（ONNX 起点版）

## 与旧流程的核心区别 ⚠️

| | 旧 `pocket-tts-axera-workflow`（英文版） | **本 skill（zh-en 版）** |
|---|---|---|
| 起点 | 有 PyTorch `.safetensors`，用 `torch.onnx.export` 重导 | **只有供应商融合 ONNX**（`step_model.onnx` + `step_encoder.onnx`），无源码/权重 |
| 手段 | 模块化重导出（prefix/AR/flow/mimi 分开导出） | **ONNX 图手术**：子图提取、位置补丁、拆图、注意力融合 |
| 风险点 | 导出对齐 PyTorch | 图语义逆向（KV 布局、mask 语义、int64 死代码） |

**不要**把英文流程的结论直接搬过来（模型是个人训练的，结构可能不同）；先做 M0 结构鉴定。

## 环境与路径

- 工程根：`/data/shared/huyuan/TTS_quant/pocket-tts-zh-en-axera`（GitHub `ZY-2012/pocket-tts-zh-en.AXERA`）
- 供应商 ONNX：`/data/shared/huyuan/TTS/pocket-tts-zh-en/`（`step_onnx/step_model.onnx` 融合图 + `step_encoder.onnx` + spm + `Vivian.wav`）
- 推理包：HF `HY-2012/pocket-tts-zh-en.AXERA`（axmodel + int8 ONNX + Python/C++ 运行时）
- benchmark：`/data/shared/huyuan/8860_VoiceTest/Voice_Test.AXERA`（GitHub `ZY-2012/Voice_Test.AXERA`）
- 板端：AX650N `root@10.126.29.50`（密码 123456）；本机 `/data/shared/huyuan/` = 板端 `/root/huyuan/workspace/`（NFS，2.7MB/s）
- 板端本地模型：`/root/pocket_tts_zh_en/{hf_pkg/models, cpp/bin123, output}`
- 主机 python：`/data/huyuan/miniforge3/bin/python`（onnx/onnxruntime 1.23.2）；ASR：`/data/huyuan/miniforge3/envs/funasr/bin/python`（SenseVoiceSmall 缓存）
- Pulsar2：`source /data/huyuan/npu-codebase/script/npu_dev`（base conda）
- C++ 工具链：`/data/shared/huyuan/toolchains/gcc-arm-9.2-...-none-linux-gnu`；BSP `ax650n_bsp_sdk/msp/out`；ORT 1.23 aarch64 在 `.work_tmp/ort/onnxruntime-1.23.0-aarch64`

**纪律**：禁止用 `/tmp`；所有产物落工程目录（`.work_tmp/` 可作临时区）。板端 root 写的 NFS 文件要 `chown -R 1055:1001` 回主用户。

## 模型结构（先鉴定，别假设）

融合 `step_model.onnx` 是「单步图」，所有递归状态显式 I/O：
`flow_kv/past, mimi_kv[266], mimi_conv[14720], flow_offset/mimi_offset`；输出 `audio[1,1,1920]/next_latent[1,32]/eos_logit`。
- FlowLM 6 层 d1024，**无 context 上限**（全因果）；Mimi 2 层有 250 窗口（图内原生 266 滑窗）
- 3 路门控 `gates[3]`：text / latent / cond；每调用解码 16 个子帧（80ms）
- 鉴图脚本：`python/extract_subgraphs.py` 前的只读分析（见 M0 报告 `results/M0_report.md`）

## 复现流程（编号即顺序）

### M0 基线（主机）
```bash
python python/baseline_run.py --models step_onnx_int8   # 金标准 wav + RTF
python python/baseline_run.py --models step_onnx
python python/profile_ort.py --model step_onnx --frames 16   # ORT 热点
```
产物：`results/baseline.json`、`results/audio/step_onnx*_*.wav`（固定 seed 金标准）。

### M1 图手术（主机）
```bash
python python/extract_subgraphs.py      # 融合图 -> flow/flow_net/mimi 三子图（逐位一致）
python python/patch_flow_window.py      # flow mask 位置标签补丁：shift = flow_offset - (total-seq)
python python/validate_subgraphs.py     # 子图 vs 全图逐帧等价（== 0）
```
- Flow 窗口化：**仅当 cache ≤ W 时精确**；超出当步立即分叉（实测 W=256/384/512 的首分叉步 = 135+W）。
- 分块推理（≤48 token/块、缓存 <512）在 W=512 下天然精确。

### M2 拆分与量化（主机 + Pulsar2）
```bash
python python/split_mimi_npu.py         # mimi -> transformer(CPU) + conv(NPU)，逐位一致
python python/make_flow_ar_onnx.py --int8   # 精简 AR（烘焙 gates=latent），75.7MB
python python/quantize_int8.py          # prefill / mimi-tf 的 ORT int8（只量化 MatMul/Gemm！）
python python/make_encoder_static.py    # 编码器 40/41/44/48 帧静态档
python python/generate_calib.py && python python/generate_calib_split.py && python python/generate_calib_encoder.py
CHECK_LEVEL=2 bash scripts/ax650/04_quant.sh   # flow_net + mimi_conv check2
```

**关键坑（本模型实测）**：
1. **Pulsar2 会死代码化 int64 `mimi_offset`**（不同 offset 输出完全一致、offset_out 是垃圾）→ 放弃整图 mimi 量化，改拆图（transformer 留 CPU）。
2. **NPU Softmax 上限 640×640**：编码器 40f（3.2s）可编译，41f 起必失败 → 参考音超档取**尾窗**、不足**左补齐**；头截会让部分文本节奏失控（实测）。
3. ort `quantize_dynamic` 必须 `op_types_to_quantize=["MatMul","Gemm"]`，否则 Conv→ConvInteger 在 CPU EP 无实现。
4. 校准数据必须来自已验证 runtime 轨迹（多音色多文本，864 样本/张量）。
5. `flow_net` 用 **FP32 axmodel**（U16 会让轨迹失控）；`mimi_conv` U16 可用。

### M3 板端 Python 混合运行时
```bash
python3 board/pocket_tts_axera.py ... --npu-flow-net 1 --npu-mimi-conv 1 \
  --flow-ar-model flow/flow_ar_step_int8.onnx --flow-prefill-model flow_step_int8.onnx \
  --mimi-tf-model mimi_transformer_step_int8.onnx --encoder-dir models/encoder
# 或一键
bash scripts/board_run.sh "文本" out.wav        # 工程内
bash run_ax650.sh "文本" out.wav                # HF 包内
python3 board/pocket_tts_batch.py --texts-json configs/listen_texts.json ...  # 批量（RTF 口径）
```
- 线程：AR=4~5、prefill=8、mimi=2；**更多线程退化**（板端争抢）。
- 文本前端与 `demo.py` 一致 + 数字转汉字 + 三级分块（句末→逗号→硬切，≤48 token）+ 块间 120ms。
- `--stream 1` 流式写 WAV；voice 前缀缓存跨 chunk 复用（首帧 2.0s → 0.3s）。

### M4 指标
- 回环 CER（主机 SenseVoiceSmall）：`python python/roundtrip_cer.py --wavs ... --text "..."`（参考文本先过同一前端＝口语形式，与 benchmark `text.ref` 口径一致）
- 试听包：`python python/listen_pack_report.py` → `results/audio/listen_pack/`
- **PCM 波形 cos 不是验收指标**：任何 NPU 入环都会轨迹分叉（0.05~0.9），但 CER 在 ASR 底噪内；验收 = CER + 人工试听。

### M5 C++ 运行时（加速路径）
```bash
python python/prepare_tokens.py --spm ... --text "..." --out req.tokens   # 主机分词
bash cpp/build_ax650.sh        # 交叉编译（ORT 1.23 aarch64）
# 板端
LD_LIBRARY_PATH=... ./bin123/pocket_tts_zh_en --models-dir models --reference models/Vivian.wav \
  --tokens-file req/zh_short.tokens --output out.wav --threads 5 --prefill-threads 8 --mimi-threads 2 \
  --flow-ar-model flow_ar_step_fused_int8.onnx
```
加速清单（按收益排序，全部实测）：
1. **ORT 1.14 → 1.23**（官方 aarch64，GLIBC 2.17/2.27）：RTF 0.73/0.81 → 0.59/0.60（1.26 反而慢 3~4%）
2. **跨帧流水线**：主线程 Flow-AR+flow_net ∥ worker 线程 Mimi 解码（`std::async` + NPU 互斥）→ 0.40/0.41
3. **注意力融合**：`python/fuse_attention.py` 把每层 Transpose/mask 链/Softmax 换成 opset-23 `Attention`（578→422 节点，fp32 逐位一致）→ AR 32.8→29.0ms，RTF ~0.39
4. 线程 5/2 + 关闭每帧 flush；不要静态 QDQ（更慢）、不要盲目加线程

### M6 发布与 benchmark
```bash
# benchmark（板端）
TTS_MODELS=pocket_tts_zh_en TTS_DATASETS="aishell3 ljspeech zh_long zh_hardcase" bash tts/run.sh
# 主机刷新报表
python tools/report.py --readme
```
HF 上传坑：`.so` 和可执行文件必须 LFS 跟踪（`*.so*`、`cpp/bin/*`），**HF 拒绝符号链接**（用实体文件）；`cp staging/. .` 会覆盖 `.gitattributes`，推前检查。

## 实测指标（AX650N，Vivian 音色）

| 项 | 数值 |
|---|---|
| Python 混合运行时（试听集 10 条） | 平均 RTF **0.665**，首帧 167ms |
| C++（ORT1.23+流水线+融合注意力） | zh_short **~0.39** / zh_long **~0.39~0.43**（板端有人负载时波动） |
| 逐帧 | AR ~29ms + flow_net 0.6ms（关键路径）；mimi_tf ~23ms（被流水线隐藏） |
| benchmark | aishell3 CER 8.46%（地板 2.90%）· ljspeech WER 7.84% · zh_long CER **0.68%** · zh_hardcase 2.46%，成功率 100% |

## 板端环境坑

1. **NPU 独占**：他人进程（如 axllm/audio-pipeline）占用时 `AX_ENGINE_Init` 失败，等其结束再跑。
2. 板端 root 经 NFS 创建文件属主 root → `chown -R 1055:1001 <repo>/results <repo>/tools`；跑脚本加 `python3 -B`。
3. CPU 锁定 1.7GHz（userspace governor，无调频余量）；板端常有他人负载，测 RTF 要 A/B 交替看相对值。
4. 重模型必须拷到板端本地盘（NFS 2.7MB/s；ORT 冷加载 400MB 会多等 2~3 分钟）。
5. 板端 SenseVoice 大模型加载曾导致 SSH 断连（内存压力），ASR 评测优先用 benchmarking 的 ASR 脚本或主机 SenseVoiceSmall。

## 工程文件索引

```text
python/    extract_subgraphs / patch_flow_window / split_mimi_npu / make_flow_ar_onnx /
           quantize_int8 / make_encoder_static / generate_calib* / make_quant_configs /
           fuse_attention / validate_* / subgraph_runtime / baseline_run / profile_ort /
           roundtrip_cer / listen_pack_report / prepare_tokens
board/     pocket_tts_axera.py（混合运行时）/ pocket_tts_batch.py / diag_stepwise / cmp_ax_onnx
cpp/       pocket_tts_zh_en.cpp / build_ax650.sh / CMakeLists.txt / src/engine_wrapper
scripts/   ax650/04_quant.sh / board_run.sh
configs/   baseline_texts.json / listen_texts.json
results/   M0/M1/M2_M3/M5_M6_M7/M8 报告 + board_metrics.json + 全部音频/指标
```

完整报告：`results/M0_report.md`、`M1_report.md`、`M2_M3_report.md`、`M5_M6_M7_report.md`、`M8_report.md`。
