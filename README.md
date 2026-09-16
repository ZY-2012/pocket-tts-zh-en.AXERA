# Pocket-TTS 中英双语 · AX650 部署工具链

[Pocket-TTS 中英双语社区版](https://www.tulingyun.com/tts_clone.html)（Mimi 编解码器 + FlowLM/LSD，支持零样本声音克隆）在 **AX650N** 上的导出、拆图、量化与板端推理工程。

推理包（axmodel + 运行时 + 音色资源）见 Hugging Face：
**https://huggingface.co/HY-2012/pocket-tts-zh-en.AXERA**

## 实测指标（AX650N，参考音色 Vivian.wav）

| 项目 | Python 运行时 | **C++ 运行时（推荐，更快）** |
|---|---|---|
| 平均 RTF（11 条中英文本，10 条计） | 0.665 | — |
| 长文（3 分块，18s 音频） | 0.674 | **0.39~0.43** |
| 短句 | 0.72~0.81 | **~0.39** |
| 首帧延迟（含音色编码 + prefill） | 167 ms | **~130 ms** |
| 逐帧耗时（80ms/帧） | ≈51 ms | **AR 29.0ms + flow_net 0.6ms（关键路径）；mimi_tf ~23ms 被流水线隐藏** |
| 回环 CER（SenseVoiceSmall） | 常规文本 0%（ASR 底噪内）；古文/绕口令为 ASR 极限 | 同左 |
| benchmark（4 数据集） | aishell3 CER 8.46%（地板 2.90）· ljspeech WER 7.84% · zh_long CER 0.68% / RTF 0.699 · zh_hardcase 2.46% | — |

C++ 加速路径（累计较 Python 快 ~1.7~1.8 倍）：ORT 1.14→**1.23**（int8 算子）→ **跨帧流水线**（Flow-AR+flow_net ∥ Mimi 解码）→ **注意力融合**（每层布局/mask 链/Softmax → opset-23 `Attention`，578→422 节点，fp32 逐位一致）→ 线程 AR=5/mimi=2。

复现 C++：
```bash
python python/prepare_tokens.py --spm <spm> --text "你好，世界。" --out req.tokens
bash cpp/build_ax650.sh                # 交叉编译（aarch64 ORT + axengine）
./bin/pocket_tts_zh_en --models-dir models --reference models/Vivian.wav \
  --tokens-file req.tokens --output out.wav --threads 5 --prefill-threads 8 --mimi-threads 2 \
  --flow-ar-model flow_ar_step_fused_int8.onnx
```

## 运行时架构（混合 NPU / CPU）

| 阶段 | 后端 | 模型 |
|---|---|---|
| 音色编码 | **NPU** | `step_encoder_40f.axmodel`（静态 3.2s 档，24.8MB） |
| FlowLM prefill（文本/音色） | CPU ORT int8 | `flow_step_int8.onnx`（101.5MB） |
| FlowLM AR 步 | CPU ORT int8 | `flow_ar_step_int8.onnx`（75.7MB，KV 窗口 512） |
| flow_net（LSD） | **NPU FP32** | `flow_net_step_fp32.axmodel`（36MB） |
| Mimi transformer（16 子帧） | CPU ORT int8 | `mimi_transformer_step_int8.onnx`（6.6MB，KV 窗口 266） |
| Mimi 卷积解码 | **NPU U16** | `mimi_conv_step.axmodel`（4.4MB） |

文本前端：标点归一化 + 数字转汉字（年份逐位/小数/负数）+ 三级分块（句末→逗号→硬切，≤48 token）+ 块间 120ms 静音；支持 `--stream` 逐帧写 WAV。

## 关键工程结论（本模型实测）

1. **PCM 波形 cos 不是合格指标**：任何 NPU/量化组件入环都会因自回归混沌放大导致波形分叉（cos 0.05~0.9），但回环 CER 保持在 ASR 底噪内。验收用 **CER + 人工试听**。
2. **Pulsar2 会死代码化 int64 位置输入**（`mimi_offset`：不同取值输出完全一致，offset 输出为垃圾值）→ 放弃整图量化，拆为 `mimi_transformer(CPU)` + `mimi_conv(NPU)`，拆分前后逐位一致。
3. **FlowLM 无语义 context 上限**（无 delta<250 掩码），Mimi 才有 250 窗口；flow 窗口化需补位置标签（`python/patch_flow_window.py`，缓存 ≤ W 时与全量逐位一致，W=512 覆盖 ≤48 token/块的常规推理）。
4. **NPU Softmax 尺寸上限 640×640**：编码器 40 帧档（3.2s）可编译，41 帧起（656×656）必失败 → 参考音超档取**尾窗**、不足**左补齐**（头截会让部分文本节奏失控，实测修正）。
5. ORT 线程：seq=1 小算子场景 **AR=5 / mimi=2 最优**（C++ 流水线下），6/8 线程反而退化；prefill 独立 8 线程会话。profiler 显示 AR 耗时里 26% 是动态量化 MatMul、44% 是布局小算子 → 注意力融合收益最大；静态 QDQ 更慢、不要用。

## 主机侧复现（导出 / 验证 / 量化）

依赖：Python 3.10+、`onnx`、`onnxruntime`、`onnxsim`、`sentencepiece`、`numpy`、`soundfile`、`scipy`；
Pulsar2（AX650 工具链）用于编译 axmodel。

```bash
export POCKET_TTS_ROOT=/path/to/pocket-tts-zh-en   # 供应商 ONNX 权重目录
python python/extract_subgraphs.py                 # 融合图 -> flow/flow_net/mimi 子图（逐位一致）
python python/patch_flow_window.py                 # flow KV 窗口位置补丁
python python/split_mimi_npu.py                    # mimi -> transformer + conv（逐位一致）
python python/make_flow_ar_onnx.py --int8          # 精简 AR 图 + ORT int8
python python/quantize_int8.py                     # prefill / mimi-tf 的 ORT int8
python python/make_encoder_static.py               # 编码器静态档（40f 可编译）
python python/generate_calib.py                    # 校准数据（多音色多文本）
python python/generate_calib_split.py
python python/generate_calib_encoder.py
CHECK_LEVEL=2 bash scripts/ax650/04_quant.sh       # Pulsar2 check2（flow_net + mimi_conv）
python python/validate_subgraphs.py                # 数值对齐验证
python python/validate_mimi_split.py
python python/roundtrip_cer.py --wavs out.wav --text "..."   # 回环 CER（需 SenseVoiceSmall）
```

板端一键：

```bash
bash scripts/board_run.sh "你好，世界。" out.wav
```


## C++ 运行时（可选，交叉编译）

源码在 `cpp/`（aarch64 ONNX Runtime + axengine，rust-free 单一可执行文件）。
文本分词在主机侧完成（与 Python 前端逐位一致），生成 request 文件后 C++ 只做推理。

```bash
# 主机：生成 request（每行一个分块的 token ids）
export POCKET_TTS_ROOT=/path/to/pocket-tts-zh-en
python python/prepare_tokens.py --spm $POCKET_TTS_ROOT/chn_jpn_yue_eng_ko_spectok.bpe.model \
  --text "你好，世界。" --out request.tokens

# 交叉编译（工具链/BSP/ORT 路径可用环境变量覆盖）
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-aarch64 BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-aarch64-1.14.0 bash cpp/build_ax650.sh

# 板端：把 cpp/bin/ 与 HF 包的 models/ 拷到板端本地后运行
./bin/pocket_tts_zh_en --models-dir models --reference models/Vivian.wav \
  --tokens-file request.tokens --output out.wav --threads 4 --prefill-threads 8
```

C++ 依赖的 aarch64 ORT 需 **glibc ≤ 板端版本**：用官方 **1.23.0** aarch64 包
（GLIBC 2.17/2.27，实测可用；非官方高版本构建可能要求 GLIBC≥2.38）。
运行时做跨帧流水线（主线程 Flow-AR+flow_net ∥ worker 线程 Mimi 解码），
并把 AR 图每层的注意力（Transpose/Split/mask 链/Softmax）重写为单个 opset-23
`Attention` 算子（578→422 节点，`python/fuse_attention.py`，fp32 逐位一致），
实测 AX650N：zh_short RTF ≈0.39、zh_long RTF ≈0.39~0.43（随板端负载波动）、
首帧 ~0.13s、CER 0%。

## 目录

```
configs/     # 金标准/试听文本集
python/      # 拆图、补丁、静态化、int8、校准、Pulsar2 配置、验证、CER 脚本
board/       # AX650 运行时（axengine + onnxruntime）与批量驱动
cpp/         # C++ 运行时（交叉编译，ORT + axengine 混合）
scripts/     # ax650 量化脚本、板端一键脚本
docs/        # 各阶段报告与指标（M0/M1/M2-M3/M5-M7）
```

## 模型与许可

- 上游模型权重：供应商社区版（中英双语 + 声音克隆），**CC BY-NC 4.0，仅限非商业用途**；商用需另行授权（见上游说明）。
- 本仓代码：Apache License 2.0。
- 声音克隆请确保已获得被克隆者授权。

## 参考与感谢

- **原工程**：[Pocket-TTS 中英双语社区版（图灵云）](https://www.tulingyun.com/tts_clone.html) —— 模型权重与融合 ONNX 导出件来源，本项目全程基于其 ONNX 做图手术（无 PyTorch 源码）
- **上游架构**：[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) —— CALM / Lagrangian Self Distillation
- **转换与量化工具**：[AXERA-TECH/Magnetar](https://github.com/AXERA-TECH/Magnetar) —— 模型 → ONNX → Pulsar2 → AXMODEL 工作流
- **推理与评测栈**：[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[AXERA pyaxengine](https://github.com/AXERA-TECH/pyaxengine)、Pulsar2（AX650 BSP）、[FunASR SenseVoiceSmall](https://github.com/modelscope/FunASR)（回环 CER）
- **板端评测框架**：[ZY-2012/Voice_Test.AXERA](https://github.com/ZY-2012/Voice_Test.AXERA)
- **参考音色**：`Vivian.wav`（来自原工程分发包）
