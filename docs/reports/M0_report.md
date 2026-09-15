# M0 报告：pocket-tts-zh-en 主机基线

日期：2026-09-15 · 工程：`/data/shared/huyuan/TTS_quant/pocket-tts-zh-en-axera`
模型：`/data/shared/huyuan/TTS/pocket-tts-zh-en`（供应商 ONNX，无 PyTorch 权重）

## 1. 基线指标（主机 CPU，4 线程，seed=0，temp=0）

金标准音频：`results/audio/{step_onnx,step_onnx_int8}_<tag>.wav`，指标：`results/baseline.json`

| 文本 | token | fp32 帧/时长/RTF | int8 帧/时长/RTF | fp32 首帧 | int8 首帧 |
|---|---:|---|---|---:|---:|
| zh_short | 18 | 44 / 3.52s / 0.294 | 44 / 3.52s / 0.204 | 84ms | 60ms |
| zh_notice | 45 | 104 / 8.32s / 0.293 | 100 / 8.00s / 0.212 | 97ms | 70ms |
| en_short | 12 | 44 / 3.52s / 0.279 | 44 / 3.52s / 0.203 | 85ms | 58ms |
| mix_short | 29 | 67 / 5.36s / 0.274 | 67 / 5.36s / 0.174 | 99ms | 63ms |
| zh_long | 93 | 208 / 16.64s / 0.277 | 204 / 16.32s / 0.192 | 124ms | 77ms |

- 模型加载：fp32 1.47s / int8 0.92s；首帧延迟（含 prefill）58~124ms。
- **int8 与 fp32 生成帧数不一致**（zh_notice 100 vs 104，zh_long 204 vs 208）：EOS 判定受量化影响，属预期；两者各自作为金标准。
- 主机 RTF 不代表板端，仅用于相对热点分析。

## 2. 结构核对（与 `step_config.json` 一致）

| 项 | 图内证据 |
|---|---|
| flow 6 层 / mimi 2 层 | 模块名 `in_proj..in_proj_5`（flow）+ `in_proj_6/_7`（mimi） |
| latent 32 / model_dim 1024 / 16 heads | `latent[1,seq,32]`、`cond[1,seq,1024]`、flow_kv `[past,6,2,1,16,64]` |
| mimi cache 266 | 图内常量 `/Constant_318 = 266`，`pos_k = mimi_offset - 266 + arange(266)` |
| 16 子帧/帧 | `mimi_kv_new[16,2,2,1,8,64]`、`decode_steps` 参与 `*16` |
| conv 状态 14720 | `mimi_conv[14720]`，被 9 个 Slice 分流（含 convtr 尾状态） |
| 80ms/帧 | `audio[1,1,1920]` @24kHz |

**位置编码语义（拆图关键）**
- Flow 侧：`pos_q = flow_offset + arange(seq)`，`pos_k = arange(0, past+seq)`，mask `0 <= delta < 250(context)`
  → 要求 **offset == cache 实际总长**；截断窗口必须先改图（补 `offset-window+arange` 平移）。
- Mimi 侧：原生滑窗 266（`Sub_18` 用常量 266 平移），固定窗口可直接静态化。

## 3. ORT 节点画像（prefill + 16 帧，4 线程）

| 模块 | fp32 | int8 | 说明 |
|---|---:|---:|---|
| flow_lm_transformer_6l | 152.4ms (34.0%) | 53.8ms (15.3%) | int8 动态量化后大降 |
| mimi_conv_decoder | 85.1ms (19.0%) | 86.8ms (24.6%) | 4×ConvTranspose 占 59ms，NPU 首选 |
| other（长尾小算子） | 80.3ms (17.9%) | 79.4ms (22.5%) | Unsqueeze/Concat/MatMul 碎片 |
| flow_net | 42.1ms (9.4%) | 39.3ms (11.2%) | 全静态，NPU 候选 |
| shapes_attn_aux | 35.8ms (8.0%) | 37.2ms (10.5%) | mask/shape，CPU glue |
| mimi_transformer_2l | 28.7ms (6.4%) | 31.5ms (8.9%) | 固定窗 266，可静态化 |
| input_switch_embed | 24.1ms (5.4%) | 24.2ms (6.9%) | Gather 词表 25055×1024 |
| quantizer_conv | 0.4ms | 0.3ms | 32→512 卷积 |
| **合计** | **448.9ms / 28.1ms 每帧** | **352.5ms / 22.0ms 每帧** | |

## 4. 对 M1 的直接输入

1. 拆图边界已由现有 I/O 天然给出，最小切割方案：
   - `flow_prefix_step`：tokens/cond/is_bos + flow_kv → flow_kv_new（变长 seq）
   - `flow_ar_step`：latent/is_bos + flow_kv → c、eos_logit、flow_kv_new（seq=1）
   - `flow_net_step`：c + noise → next_latent（全静态）
   - `mimi_decoder_step`：latent + mimi_kv + mimi_conv + mimi_offset → audio、三个新状态（**窗口/状态全静态**）
2. NPU 优先级（按解耦与收益）：`mimi_decoder_step`（状态已静态、卷积为主）→ `flow_net_step` → `flow_ar/prefix`（需 KV 窗口化改图 + 量化敏感性实测）→ `step_encoder`（固定参考音频长度）。
3. 拆图后先做子图↔全图逐帧等价验证（flow_kv 全量 cache 语义），再做窗口化实验。

## 5. 产物

- `results/baseline.json`：全部运行指标 + token ids + PCM blake2b
- `results/audio/*.wav`：10 条金标准
- `results/profile_step_onnx{,_int8}.json`：节点画像
- `python/baseline_run.py` / `python/profile_ort.py`：可复现脚本
