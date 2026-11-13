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

**WeightBake-3B** is a context-distilled, SFT + DPO aligned `Qwen/Qwen2.5-3B-Instruct` model trained to internalize a **1,922-token** enterprise voice-gateway system prompt directly into its weights so that runtime inference only requires a **10-token** trigger (`Respond as the deterministic voice router in raw JSON.`).

## Key Results

- **Prefill System Prompt Reduction**: `1,922` tokens -> `10` tokens (**99.48%** reduction)
- **Holdout Full Compliance (80 samples)**: **97.50%** (78/80) vs. **81.25%** (65/80) for the bloated-prompt baseline
- **KV-Cache Memory Savings (GQA-2, 36 layers, FP16)**: **71.93 MiB** -> **4.71 MiB** per session (**15.27x** reduction; max concurrent sessions on 8.0 GiB KV VRAM increases from **113** to **1,738**)
- **vLLM P50 TTFT Speedup**: **6.52x** at `C=1` (`161.8 ms` -> `24.8 ms`) and **17.45x** at `C=32` (`835.6 ms` -> `47.9 ms`)

## 4-Stage Alignment Progression (80-Sample Holdout)

| Stage | System Tokens | Full Compliance | Mean Check Score | Valid JSON | Strict Schema | Zero Sycophancy | Spoken 5-22 Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_unprompted` | 10 | 12.50% | 70.0% | 77.5% | 66.2% | 62.5% | 66.2% |
| `base_bloated` | 1,922 | 81.25% | 94.1% | 96.2% | 93.8% | 92.5% | 92.5% |
| `sft_minimal` | 10 | 83.75% | 97.7% | 100.0% | 100.0% | 91.2% | 92.5% |
| `dpo_minimal` | 10 | **97.50%** | **99.6%** | **100.0%** | **100.0%** | **100.0%** | **97.5%** |

## vLLM Latency & Throughput Benchmark

| Concurrency | Baseline TTFT (P50 / P95) | WeightBake TTFT (P50 / P95) | P50 TTFT Speedup | Throughput RPS (Base -> WB) |
|---|---:|---:|---:|---:|
| C=1 | 161.8 / 169.4 ms | 24.8 / 25.7 ms | **6.52x** (-84.66%) | 1.14 -> 1.41 (1.24x) |
| C=8 | 185.7 / 454.4 ms | 26.8 / 46.1 ms | **6.93x** (-85.57%) | 6.95 -> 10.14 (1.46x) |
| C=16 | 238.8 / 896.3 ms | 30.1 / 55.9 ms | **7.93x** (-87.39%) | 10.80 -> 18.56 (1.72x) |
| C=32 | 835.6 / 1913.9 ms | 47.9 / 102.7 ms | **17.45x** (-94.27%) | 11.01 -> 29.21 (2.65x) |

## Training & Export Summary

- **Stage 1 (SFT)**: QLoRA (`r=16`, `alpha=32`, 7 linear projections) on 800 context-stripped samples (`760` train / `40` val) with assistant-only loss masking (`-100` on system & user tokens), `lr=2e-4`, 2 epochs.
- **Stage 2 (DPO)**: Bradley-Terry sigmoid DPO (`beta=0.1`, `lr=5e-6`, 1 epoch) on 800 preference triplets targeting 6 failure modes (`markdown_fenced`, `sycophantic_spoken`, `word_count_overflow`, `tts_unsafe_chars`, `schema_drift`, `routing_misclassification`).
- **Export**: Folded 16-bit merged weights (`checkpoints/weightbake-3b-merged`) for zero-overhead `vLLM` deployment (`bfloat16`) or GGUF (`q4_k_m`).

## Intended Usage

```text
System: Respond as the deterministic voice router in raw JSON.
User: Account AC-8841 was billed 499.50 twice this morning. Check that invoice right away.
```
