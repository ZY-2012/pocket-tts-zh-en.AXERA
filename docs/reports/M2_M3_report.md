# M2/M3 报告：量化、板端混合推理与指标

日期：2026-09-15 · 平台：AX650N（root@10.126.29.50，5.9GB RAM，8 核）
工程：`pocket-tts-zh-en-axera` · 模型：供应商 ONNX（无 PyTorch 源）

## 1. 量化路线（M2）

### 1.1 Pulsar2 关键发现（本模型实测）

| 问题 | 现象 | 处理 |
|---|---|---|
| `mimi_offset`（int64）死代码化 | axmodel 对 offset 0/416/1000 输出完全一致，`offset_out` 为垃圾值 | 放弃整图 mimi 量化，改**拆图** |
| U16 秩 1 广播 Add 崩溃 | `AxQuantizedAdd (16,)+(1,)` 编译失败 | 位置路径随拆分移出 NPU 图 |
| mimi mask 依赖 offset（前 266 子帧窗口裁剪） | mask 非恒定 | mask 留在 CPU 图（拆分后不再进 NPU） |

### 1.2 最终拆图（全部逐位一致，`results/validate_mimi_split.json`）

```text
mimi_decoder_step
  ├── mimi_transformer_step (ONNX, CPU fp32)
  │     next_latent, mimi_kv, mimi_conv, mimi_offset
  │       -> decoder_embedding, mimi_kv_new, upsample_state
  └── mimi_conv_step (axmodel, NPU)             <- 无 int64 / 无 mask / 无 RoPE
        decoder_embedding, mimi_conv, upsample_state -> audio, mimi_conv_out
```

flow 侧另做精简 AR 图（gates 烘焙为 latent 路，去掉 25k 词表 embedding）：

| 模型 | 大小 | 验证 |
|---|---:|---|
| `flow_ar_step.onnx`（fp32 精简） | 302 MB | 与金标准逐位一致（cos 1.000000） |
| `flow_ar_step_int8.onnx`（ORT 动态 int8） | 75.7 MB | 轨迹分叉但回环 CER=0% |

### 1.3 check2 精度（主机 EndToEnd，864 校准样本）

| 模型 | 最低 cosine | 说明 |
|---|---:|---|
| `flow_net_step`（U16） | 0.99957 | 单步好，但入环后轨迹分叉（见 §3.4） |
| `mimi_conv_step`（U16） | 输出 audio cos 0.99897 | 入环后轨迹分叉但 CER 达标 |
| `flow_net_step_fp32`（全 FP32） | — | 仍入环分叉；用于默认配置 |

校准数据：`model_convert/calib_data/subgraphs/`（3 音色 × 9 文本，864 样本/张量，1.5GB）

## 2. 板端运行时（M3）

### 2.1 组件与后端

| 组件 | 后端 | 文件 | 大小 |
|---|---|---|---:|
| flow LM（prefill） | ORT CPU fp32（8 线程） | `flow_step_windowed.onnx` | 405 MB |
| flow LM（AR 步） | ORT CPU int8（4 线程） | `flow/flow_ar_step_int8.onnx` | 75.7 MB |
| flow_net | **NPU FP32** | `flow_net_step_fp32.axmodel` | 36.0 MB |
| mimi transformer | ORT CPU fp32 | `mimi_split/mimi_transformer_step.onnx` | 25.5 MB |
| mimi 卷积解码 | **NPU U16** | `mimi_conv_step.axmodel` | 4.4 MB |
| mimi 编码（音色） | ORT CPU fp32 | `step_encoder.onnx` | 39 MB |

其他：KV 窗口 512（flow，精确范围内）、mimi 窗口 266、voice 前缀缓存跨 chunk 复用、文本按标点/48 token 分块。

### 2.2 端到端指标（默认速度配置，`scripts/board_run.sh`）

| 文本 | token/块 | 音频 | 耗时 | RTF | 回环 CER | fp32 金标准 CER | PCM cos |
|---|---|---:|---:|---:|---:|---:|---:|
| zh_short | 18 | 3.52s | 4.37s | 1.240 | 0.00% | 0.00% | 0.519 |
| zh_notice | 45 | 8.24s | 8.24s | 1.000 | 7.14% | 4.76% | 0.208 |
| zh_long（3 块） | 93 | 18.08s | 15.99s | **0.885** | 0.00% | 0.00% | 0.052 |
| en_short | 12 | 3.52s | 4.30s | 1.221 | 0.00% | 0.00% | 0.432 |
| mix_short | 29 | 5.36s | 5.82s | 1.085 | 4.00% | 10.00% | 0.197 |

质量配置（AR 换 fp32 精简图）：RTF 1.24~1.57，CER 同样在 ASR 底噪。

### 2.3 逐帧阶段耗时（threads=4，中位）

```text
flow AR (int8)   29.8 ms
mimi transformer 21.5 ms
flow_net (NPU)    1.8 ms
mimi conv (NPU)   1.9 ms
--------------------------
合计 ≈ 55 ms/帧 (80ms/帧实时线)
```

首帧延迟 ≈ 2.0s：mimi 编码 1.27s + voice prefill 0.33s + 文本 prefill 0.08~0.35s（同音色后续块走缓存）。

### 2.4 线程实验（重要）

ORT 线程数对 seq=1 小算子过多：4 线程最优（170ms/帧），6/8 线程反而退化到 284/387ms。
AR 用 4 线程、prefill 单独会话 8 线程（`--prefill-threads`）。

## 3. 质量结论（本模型专属）

1. **PCM 波形 cos 不能作为 NPU 配置的验收指标**：任何 NPU 入环（即使 FP32 flow_net）都会因混沌放大而分叉（cos 0.05~0.87），但 **回环 CER 全部落在 ASR 底噪**（zh_notice 的“图灵云”混淆为 4.76% 底噪）。
2. 推荐验收口径：回环 CER + 人工试听（长句/英语需真人确认）。
3. 速度配置（int8 AR）与质量配置（fp32 AR）CER 无差异；默认用速度配置。

## 4. 复现

```bash
# 主机（导出/验证/量化）
python python/extract_subgraphs.py                 # 子图
python python/patch_flow_window.py                 # flow 窗口位置补丁
python python/split_mimi_npu.py                    # mimi 拆分
python python/make_flow_ar_onnx.py --int8          # 精简 AR + int8
python python/generate_calib.py                    # 校准（1.5GB）
CHECK_LEVEL=2 bash scripts/ax650/04_quant.sh       # flow_net + mimi_conv
# 板端
sshpass -p 123456 ssh root@10.126.29.50 \
  "cd /root/pocket_tts_zh_en && bash /root/huyuan/workspace/TTS_quant/pocket-tts-zh-en-axera/scripts/board_run.sh '你好，世界。' out.wav"
# 回环 CER（主机 SenseVoiceSmall）
python python/roundtrip_cer.py --wavs results/audio/board_arint8_zh_short.wav --text "..."
```

## 5. 产物索引

- 板端音频：`results/audio/board_arint8_*.wav`（速度配置）、`results/audio/board_arfp32_*.wav`
- CER：`results/cer_*.json`；PCM 对比：`results/final_pcm_cos.json`
- 拆分/验证：`results/validate_mimi_split.json`、`results/validate_subgraphs*.json`
- axmodel：`model_convert/axmodels_zh_en_2/` + `pulsar2_zh_en_check2/`（配置与日志）
- 运行时：`board/pocket_tts_axera.py`；一键脚本：`scripts/board_run.sh`
