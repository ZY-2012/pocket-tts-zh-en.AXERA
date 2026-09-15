# M5–M7 报告：试听包、延迟优化、文本前端

日期：2026-09-15 · 平台：AX650N（root@10.126.29.50）· 音色：`Vivian.wav`（原工程音色）

## M6 延迟优化（最终配置）

| 组件 | 改动 | 板端效果 |
|---|---|---|
| mimi 编码器 | ONNX CPU fp32 → **NPU 静态档 40f** | 1271ms → **30ms** |
| flow prefill | fp32 → **ORT int8**（405→101.5MB） | 文本 prefill 100→73ms；voice prefill 330→94ms |
| mimi transformer | fp32 → **ORT int8**（25.5→6.6MB） | 21.5 → **15.4ms/帧** |
| flow AR | int8 精简图（75.7MB） | 30.8ms/帧 |
| mimi conv | U16 axmodel | 2.1ms/帧 |
| flow_net | FP32 axmodel | 2.6ms/帧 |

- **首帧 2.0s → 0.30s**；逐帧合计 ≈51ms（80ms 实时线内）
- 试听集 10 条测量项：**平均 RTF 0.665**，平均首帧 167ms（载入 4.2s）

### 编码器档位（重要限制与策略）

- Pulsar2 NPU softmax 尺寸上限：**640×640 可编译（40f 档），656×656 起必失败** → 41f/44f/48f/60f/80f 全部编译不过
- 参考音超 3.2s（40 帧）只能截断：**取尾窗**（实测头截会让部分文本节奏失控：mix_long 头截 3.2s → 125 帧，尾窗 → 78 帧，全量 80 帧）
- 参考音不足 3.2s：**左补齐**（更接近动态编码器行为）
- 修正后板端与主机帧数基本一致（同文本 ±5 帧）

## M7 文本前端与体验

| 项 | 实现 | 验证 |
|---|---|---|
| 数字归一化 | `numbers_to_zh`：年份逐位（2026→二零二六）、负数/小数（-3.5→负三点五）、万/亿分组；仅当文本含 CJK 时生效；ASCII 粘连数字（TTS3/3D）不动 | `zh_number` 从 359 帧失控 → 70 帧正常，CER 0% |
| 长文分块 | 句末标点 → 逗号/顿号 → 硬切 三级；块间 120ms 静音（`--pause-ms`） | zh_long 3 块边界自然 |
| 流式输出 | `--stream`：占位 RIFF 头边生成边写、结束回填 | 主机/板端均验证 |
| CER 参考规范化 | 参考文本先过同一前端（口语形式），与 benchmark `text.ref` 口径一致 | zh_number CER 56.5%→0% |

## M5 试听包（`results/audio/listen_pack/`）

- 11 条文本 × {board, host_ref} 成对音频 + `CER.tsv`/`CER.json`/`README.md`
- 文本覆盖：中/英/混、长文、绕口令、古文、数字、品牌名

| 结果类别 | 结论 |
|---|---|
| 常规文本（6 条） | board CER = host CER（0% 或 ASR 底噪），如 zh_long/en_long 0% |
| ASR 难例（zh_tongue/zh_homophone/zh_brand） | board ≈ host（古文 91.7% 两者相同，属 ASR 极限） |
| mix_long | board 20.4% vs host 13.0%（“bilingual”听成“biual”一处），内容一致 |
| 时长 | 板端与主机基本一致（±0.2s），修正截断策略后无失控 |

## 复现命令

```bash
# 板端批量（试听集 / benchmark 同款入口）
python3 board/pocket_tts_batch.py --texts-json configs/listen_texts.json \
  --out-dir results/audio/listen_pack/board \
  --onnx-dir models --axmodel-dir models --cpu-model-dir models --mimi-split-dir models/mimi_split \
  --spm models/chn_jpn_yue_eng_ko_spectok.bpe.model --reference models/Vivian.wav \
  --threads 4 --prefill-threads 8 --pause-ms 120 \
  --flow-ar-model flow/flow_ar_step_int8.onnx --flow-net-model flow_net_step_fp32.axmodel \
  --flow-prefill-model flow_step_int8.onnx --mimi-tf-model mimi_transformer_step_int8.onnx \
  --encoder-dir models/encoder
# 试听包 CER 汇总（主机）
python python/listen_pack_report.py
```

## 产物

- 新增 int8：`models/subgraphs/windowed/flow_step_int8.onnx`、`models/subgraphs/mimi_split/mimi_transformer_step_int8.onnx`
- 编码器档位：`models/subgraphs/encoder/step_encoder_{40,41,44,48}f.onnx`（41/44/48 未编译）、`model_convert/pulsar2_zh_en_encoder/`（40f axmodel 24.8MB）
- 运行时/脚本：`board/pocket_tts_axera.py`、`board/pocket_tts_batch.py`、`python/{quantize_int8,make_encoder_static,generate_calib_encoder,listen_pack_report}.py`
