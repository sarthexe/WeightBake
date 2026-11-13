---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-3B-Instruct
pipeline_tag: text-generation
tags:
- unsloth
- trl
- sft
- dpo
- context-distillation
- vllm
- low-latency
- voice-agent
datasets:
- sarthexe/weightbake-voice-router-sft
- sarthexe/weightbake-voice-router-dpo
metrics:
- strict_schema_compliance
- ttft_ms
- kv_cache_footprint_mb
---

# WeightBake-Qwen2.5-3B (`sarthexe/WeightBake-Qwen2.5-3B`)

**WeightBake-3B** is a context-distilled, SFT + KL-anchored DPO aligned `Qwen/Qwen2.5-3B-Instruct` model trained to internalize a **1,922-token** enterprise voice-gateway system prompt directly into its weights so that runtime inference only requires a **10-token** trigger (`Respond as the deterministic voice router in raw JSON.`).

## Live NVIDIA Tesla T4 (16GB) Hardware Summary

- **Prefill System Prompt Reduction**: `1,922` tokens -> `10` tokens (**99.48%** reduction)
- **Live Tesla T4 80-Sample Holdout Compliance**: `0.0%` (Base 10-tok) -> `70.0%` (Base 1,922-tok) -> **`91.25%`** (WeightBake-3B 10-tok, **+21.25 pp** over bloated baseline)
- **Live Tesla T4 `vLLM 0.33.0` P50 TTFT Speedup**: **15.77x** at `C=1` (`1,452.85 ms` -> `92.14 ms`) up to **42.98x** P50 (**78.35x** P95) at `C=32` (`24,118.82 ms` -> `561.16 ms`), with **9.77x** RPS gain (`0.60` -> `5.86` RPS)
- **KV-Cache Memory Savings (GQA-2, 36 layers, FP16)**: **71.93 MiB** -> **4.71 MiB** per session (**15.27x** reduction; max concurrent sessions on 8.0 GiB KV VRAM increases from **113** to **1,738**)

## Live Tesla T4 4-Stage Holdout Evaluation & DPO KL-Divergence Ablation (`80` Samples)

| Stage | System Tokens | Full Compliance | No Markdown | Valid JSON | Strict Schema | Spoken 5-22 Words | Zero Sycophancy | TTS-Safe Punct | Routing Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_unprompted` | 10 | 0.0% | 100.0% | 93.75% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `base_bloated` | 1,922 | 70.0% | 100.0% | 97.5% | 95.0% | 97.5% | 97.5% | 93.75% | 73.75% |
| `sft_minimal` (`r=16, alpha=32`) | 10 | **91.25%** | 100.0% | 100.0% | 97.5% | 100.0% | 100.0% | 100.0% | 93.75% |
| `dpo_unanchored_ablation` (`beta=0.1, lr=5e-6, 50 steps`) | 10 | 83.75% | 100.0% | 100.0% | 92.5% | 95.0% | 95.0% | 95.0% | 90.0% |
| `dpo_minimal` (`beta=0.5, lr=1e-6, 15 steps`) | 10 | **91.25%** | **100.0%** | **100.0%** | **97.5%** | **100.0%** | **100.0%** | **100.0%** | **93.75%** |

### DPO KL-Divergence Ablation (`beta=0.1` Alignment Tax vs. `beta=0.5` KL-Anchored Stability)
Because QLoRA SFT with assistant-only `-100` loss masking already saturates all three TTS behavioral checks (`spoken_word_count_5_to_22`, `zero_sycophancy`, `tts_safe_punctuation`) at `100.0%`, running unanchored DPO (`beta=0.1, lr=5e-6, 50 steps`) induces classic **DPO over-optimization / alignment tax**: strict schema drops from `97.5% -> 92.5%`, routing accuracy drops from `93.75% -> 90.0%`, and full compliance falls to `83.75%`. Tightening KL regularization (`beta=0.5`), lowering the learning rate 5x (`1e-6`), and capping alignment to `15` steps (`warmup_steps=3`) completely eliminates the alignment tax and preserves **`91.25%`** compliance (`+21.25%` over `base_bloated`).

## Live Tesla T4 `vLLM 0.33.0` SSE Streaming Benchmark (`--dtype half --enforce-eager --no-enable-prefix-caching`)

| Concurrency | Baseline TTFT (P50 / P95) | WeightBake TTFT (P50 / P95) | TTFT Speedup | Mean ITL (Base -> WB) | P95 E2EL (Base -> WB) | Throughput RPS (Base -> WB) |
|---|---:|---:|---:|---:|---:|---:|
| C=1 | 1,452.85 / 1,473.43 ms | 92.14 / 121.63 ms | **15.77x** (P95: **12.11x**) | 46.23 -> 46.01 ms | 6,306.62 -> 4,772.47 ms | 0.20 -> 0.29 (**1.45x**) |
| C=8 | 1,559.73 / 9,413.11 ms | 139.75 / 188.96 ms | **11.16x** (P95: **49.82x**) | 159.85 -> 47.56 ms | 23,761.75 -> 4,918.91 ms | 0.51 -> 1.95 (**3.82x**) |
| C=16 | 2,986.33 / 21,110.16 ms | 240.72 / 321.37 ms | **12.41x** (P95: **65.69x**) | 257.99 -> 61.04 ms | 46,963.07 -> 6,299.50 ms | 0.57 -> 2.84 (**4.98x**) |
| C=32 | 24,118.82 / 44,472.91 ms | 561.16 / 567.62 ms | **42.98x** (P95: **78.35x**) | 359.87 -> 50.66 ms | 53,300.85 -> 5,457.46 ms | 0.60 -> 5.86 (**9.77x**) |

## Analytical Roofline Simulation Reference (`C=1..32`, Simulated Target: 97.50% vs. 81.25%)

| Concurrency | Baseline TTFT (P50 / P95) | WeightBake TTFT (P50 / P95) | P50 TTFT Speedup | Throughput RPS (Base -> WB) |
|---|---:|---:|---:|---:|
| C=1 | 161.8 / 169.4 ms | 24.8 / 25.7 ms | **6.52x** (-84.66%) | 1.14 -> 1.41 (1.24x) |
| C=8 | 185.7 / 454.4 ms | 26.8 / 46.1 ms | **6.93x** (-85.57%) | 6.95 -> 10.14 (1.46x) |
| C=16 | 238.8 / 896.3 ms | 30.1 / 55.9 ms | **7.93x** (-87.39%) | 10.80 -> 18.56 (1.72x) |
| C=32 | 835.6 / 1913.9 ms | 47.9 / 102.7 ms | **17.45x** (-94.27%) | 11.01 -> 29.21 (2.65x) |

## Training & Export Summary

- **Stage 1 (SFT)**: QLoRA (`r=16`, `alpha=32`, 7 linear projections) on 800 context-stripped samples (`760` train / `40` val) with assistant-only loss masking (`-100` on system & user tokens), `lr=2e-4`, 2 epochs.
- **Stage 2 (DPO)**: KL-anchored Bradley-Terry sigmoid DPO (`beta=0.5`, `lr=1e-6`, `max_steps=15`, `warmup_steps=3`) on 800 preference triplets targeting 6 failure modes (`markdown_fenced`, `sycophantic_spoken`, `word_count_overflow`, `tts_unsafe_chars`, `schema_drift`, `routing_misclassification`).
- **Export**: Folded 16-bit merged weights (`checkpoints/weightbake-3b-merged`) for zero-overhead `vLLM` deployment (`--dtype half --enforce-eager` on Tesla T4, `bfloat16` on Ampere/Ada) or GGUF (`q4_k_m`).

## Intended Usage

```text
System: Respond as the deterministic voice router in raw JSON.
User: Account AC-8841 was billed 499.50 twice this morning. Check that invoice right away.
```
