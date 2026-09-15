# M8 报告：打包、C++ 与 benchmark 接入

日期：2026-09-15

## 1. 交付仓库

| 仓库 | 内容 | 提交 |
|---|---|---|
| GitHub `ZY-2012/pocket-tts-zh-en.AXERA` | 导出/拆图/量化/验证脚本、板端运行时、C++ 源码与构建脚本、阶段报告 | `22e7dc1` |
| HF `HY-2012/pocket-tts-zh-en.AXERA` | 推理包（axmodel + int8 ONNX + 运行时 + C++ 预编译二进制 + 音色 + spm），278MB | `95269cf` |
| GitHub `ZY-2012/Voice_Test.AXERA` | benchmark 注册（builder + batch_driver）、config、README 指标注入 | `6766551` |

HF 包板端独立验证（复制到板端本地后 `bash run_ax650.sh`）：RTF 0.737，首帧 347ms。

## 2. Benchmark 全量结果（AX650N，参考音色 Vivian）

| 数据集 | 语言 | 指标 | 数值 | ASR 地板 | RTF(热启动) | 成功率 | len_ratio |
|---|---|---|---:|---:|---:|---:|---:|
| aishell3 (200) | zh | CER | 8.46% | 2.90% | 0.7373 | 100% | 0.548 |
| ljspeech (200) | en | WER | 7.84% | ~5.81% | 0.7085 | 100% | 0.844 |
| **zh_long (40)** | zh | CER | **0.68%** | — | **0.6987** | 100% | — |
| zh_hardcase (150) | zh | CER | 2.46% | — | 0.7193 | 100% | — |

对照（同集、同 ASR=firered）：zipvoice CER 4.98% / len_ratio 0.588；melotts CER 8.22% / len_ratio 0.968。
本模型在长句（zh_long）CER 0.68%，短句 aishell3 偏弱（语速快、len_ratio 0.55，与 zipvoice 的 0.59 接近）。

## 3. C++ 运行时

- 位置：GitHub `cpp/`（源码 + 交叉编译脚本）；HF 包 `cpp/bin/`（aarch64 预编译）
- 结构：aarch64 ONNX Runtime（官方 1.14，GLIBC_2.17 兼容）+ axengine（AX_SYS/AX_ENGINE Init），同一混合管线；流式 WAV（占位头回填）
- 文本：主机侧 `tools/prepare_tokens.py` 生成 token 请求文件（与 Python 前端一致）

| 用例 | 帧数 | 时长 | 首帧 | 总耗时 | RTF | 回环 CER |
|---|---:|---:|---:|---:|---:|---:|
| zh_short | 45 | 3.60s | 0.13s | 2.63s | 0.7294 | 0.00% |
| zh_long（3 块） | 224 | 18.16s | 0.15s | 14.67s | 0.8077 | 0.00% |

与 Python 同配置基本持平（Python zh_short 0.72~0.80 / zh_long 0.67）；C++ 的价值在免 Python/conda 环境与进程内流式。

### 踩坑记录（已修）

1. 缺少 `AX_SYS_Init` / `AX_ENGINE_Init` → axmodel 加载失败
2. 精简 AR 图只有 4 个输入（latent/is_bos/flow_kv/flow_offset），误按整图 7 入参构造 → 段错误
3. ORT 版本：非官方 1.21.1 aarch64 构建要求 GLIBC_2.38（板端 2.35 无法加载）；改用官方 1.14.0
4. HF 仓库：`.so` 需 LFS 跟踪、符号链接会被 pre-receive 拒绝（改实体文件）

## 4. 复现

```bash
# C++ 交叉编译（主机）
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-aarch64 BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-aarch64-1.14.0 bash cpp/build_ax650.sh
# 板端
TOKENS_FILE=cpp/req/zh_short.tokens bash cpp/run_cpp.sh out.wav
# benchmark（板端）
TTS_MODELS=pocket_tts_zh_en TTS_DATASETS="aishell3 ljspeech zh_long zh_hardcase" bash tts/run.sh
```

## 5. 产物索引

- benchmark：`Voice_Test.AXERA/results/tts.csv`（模型行 + GT 锚点行）+ README 自动表
- C++：`cpp/pocket_tts_zh_en.cpp`、`cpp/build_ax650.sh`、`cpp/bin/`、`python/prepare_tokens.py`
- 对比音频：`results/audio/cpp_zh_short.wav`、`cpp_zh_long.wav`（与 `board_arint8_*` 成对可听）
