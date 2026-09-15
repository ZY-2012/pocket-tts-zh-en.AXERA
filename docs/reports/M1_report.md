# M1 报告：子图切分、窗口化与静态化可行性

日期：2026-09-15 · 工程：`pocket-tts-zh-en-axera`
模型：`/data/shared/huyuan/TTS/pocket-tts-zh-en/step_onnx`（fp32）

## 1. 拆图结果（最小切割，全部复用图内既有状态 I/O）

| 子图 | 输入 | 输出 | 大小(raw) |
|---|---|---|---|
| `flow_step` | tokens, latent, is_bos, cond, gates, flow_kv, flow_offset | conditioning, eos_logit, flow_kv_new | 405.0 MB |
| `flow_net_step` | conditioning, noise, decode_steps | next_latent | 39.1 MB |
| `mimi_decoder_step` | next_latent, mimi_kv, mimi_conv, mimi_offset | audio, mimi_kv_new, mimi_conv_out, mimi_offset_out | 41.6 MB |

- 脚本：`python/extract_subgraphs.py`；raw + onnxsim 两套；manifest：`models/subgraphs/manifest.json`
- 等价验证（`python/validate_subgraphs.py`）：3 参考音色（Vivian/en_4.5s/zh_4.5s）× 2 文本，
  raw 与 onnxsim 均 **逐位一致**（max_abs=0，min_cos=1.0），结果 `results/validate_subgraphs*.json`

## 2. 结构判定（决定 NPU 策略的关键）

1. **FlowLM 无 context 上限**：6 层 flow 的 attention mask 只有 `pos_k>=0 & delta>=0`（图内
   `/GreaterOrEqual`,`/And`），**没有** `<250` 条件；只有 Mimi 2 层带 `Less(delta,250)`。
   - 全量 flow KV 是唯一精确解；截断窗口是近似
   - Mimi 窗口 266 是原生滑窗（`mimi_offset-266+arange`），**精确**
2. **flow 缓存键为已旋转键**（`flow_kv_new` 随 `flow_offset` 变化，max|diff|=7.6）：
   窗口喂需把 mask 位置标签改成绝对位置 → 已实现补丁 `python/patch_flow_window.py`
   （6 个 per-layer pos_k Add 节点，插 `shift = flow_offset - (total - seq)`）
3. Mimi 的 mask 与 offset 无关：`delta = 266 + q - k` 恒为常数 → 静态化时可整体烘焙，
   `mimi_offset` 可完全去掉（Pulsar2 若对 int64 不友好时启用此方案）

## 3. 窗口实验（禁 EOS，强制 420 帧，zh_long，cache 预填充 135）

| W | 首个分叉步 | 溢出前 | 整段 PCM cos / SNR |
|---:|---:|---|---|
| 256 | 124 | 逐位一致 | 0.346 / -4.5 dB |
| 384 | 252 | 逐位一致 | 0.829 / +3.5 dB |
| 512 | 380 | 逐位一致 | 0.993 / +18.6 dB |

结论：**cache ≤ W 时窗口与全量严格逐位一致**；仅在超出 W 的当步立即分叉（最大误差来自
`mimi_kv_new` 的传导）。分块推理（每块 ≤48 token、缓存上限≈48+75+1+375=499 帧）在
**W=512** 下天然精确 → 静态 NPU flow AR 的窗口选 512（留裕量可 576）。

## 4. 对 M2 的输入

- 直接可量化（形状已全静态）：
  - `mimi_decoder_step`：唯一动态点是 int64 标量 `mimi_offset`（另有 int64 mask 计算链）；若 Pulsar2 不接受，走"烘焙常量 mask + 删除 offset I/O"补丁
  - `flow_net_step`：全静态（scalar `decode_steps` float）
- `step_encoder` 固定长度变体（3~6s）留到 M4 按需
- flow AR 静态 NPU 变体（W=512，seq=1）留到 M4 若板端 RTF 不达标再做
- 校准数据必须来自本 runtime 轨迹（多音色/多文本），脚本：`python/generate_calib.py`

## 5. 产物

- `models/subgraphs/{raw,onnxsim,windowed}/` + `manifest.json` + `patch_info.json`
- `python/{extract_subgraphs,validate_subgraphs,patch_flow_window,window_overflow_check,subgraph_runtime}.py`
- `results/validate_subgraphs{,_onnxsim}.json`、`results/validate_window_w512.json`、
  `results/window_overflow_w{256,384,512}.json`
