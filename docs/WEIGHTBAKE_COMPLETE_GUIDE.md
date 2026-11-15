# WeightBake: The Complete Engineering Walkthrough & Deep-Dive Guide

> A single, exhaustive technical guide to **WeightBake** — why it exists, the physics and math behind every design choice, and how every file and function works together from start to finish.

---

## Table of Contents

1. [The Core Problem: Why System Prompt Bloat Kills Real-Time Voice AI](#1-the-core-problem-why-system-prompt-bloat-kills-real-time-voice-ai)
2. [The WeightBake Thesis: Context Distillation into Model Weights](#2-the-weightbake-thesis-context-distillation-into-model-weights)
3. [Component-by-Component Codebase Tour](#3-component-by-component-codebase-tour)
   - [Component 1: The Prompts](#component-1-the-prompts-prompts)
   - [Component 2: The Deterministic Evaluator](#component-2-the-deterministic-evaluator-srcevalevaluatorpy)
   - [Component 3: The Context-Distilled Data Engine](#component-3-the-context-distilled-data-engine-srcdata_gen)
   - [Component 4: SFT & Assistant-Only Loss Masking](#component-4-supervised-fine-tuning--assistant-only-loss-masking)
   - [Component 5: DPO & The KL-Regularization Discovery](#component-5-direct-preference-optimization--the-kl-regularization-discovery)
   - [Component 6: 16-Bit LoRA Weight Folding](#component-6-16-bit-lora-weight-folding)
   - [Component 7: KV-Cache Physics & vLLM Streaming Benchmark](#component-7-kv-cache-physics--vllm-streaming-benchmark)
4. [Summary of All Verified Results (Live Tesla T4 Hardware)](#4-summary-of-all-verified-results-live-tesla-t4-hardware)
5. [How to Run & Explore Everything Yourself](#5-how-to-run--explore-everything-yourself)

---

## 1. The Core Problem: Why System Prompt Bloat Kills Real-Time Voice AI

Imagine you are building a real-time enterprise voice agent (like a customer-support phone router connected to **ElevenLabs streaming TTS** and a backend API gateway). When a user finishes speaking on the phone, you have roughly **300–500 milliseconds** before a human notices an awkward silence.

To make a standard 3B parameter LLM (`Qwen/Qwen2.5-3B-Instruct`) behave reliably in production, engineers normally stuff a massive **1,922-token system prompt** into every request (`prompts/bloated_system_prompt.txt`). Why is it so long? Because it has to specify:

1. **Strict Raw JSON Output**: Never output ` ```json ` markdown wrappers or conversational text outside `{...}` (otherwise `json.loads()` crashes with an HTTP 500 error).
2. **Exact 6 Top-Level Keys (`extra="forbid"`)**: `"intent"`, `"urgency"`, `"requires_human"`, `"tool_call"`, `"internal_reasoning"`, and `"spoken_response"`.
3. **8 Allowed Intents & 7 Tool Schemas**: Exact argument names and types for `lookup_invoice`, `run_diagnostics`, `lock_account_mfa`, `modify_subscription`, `track_shipment`, `escalate_specialist`, and `none`.
4. **ElevenLabs TTS Constraints**:
   - `"spoken_response"` must be **5 to 22 words** (short enough for natural phone cadence).
   - **Zero sycophancy** (banning 13 phrases like *"Certainly!"*, *"I understand your frustration"*, *"As an AI"*).
   - **Zero TTS-unsafe punctuation** (banning 19 characters like `*`, `#`, `:`, `/`, `@`, `(` that cause TTS engines to read punctuation aloud or glitch).
5. **Adversarial & Edge-Case Rules + 6 Few-Shot Examples**: Deflecting prompt injections, routing GDPR/subpoenas to legal, distinguishing 2FA breaches from technical outages.

### What Happens in Hardware When You Send 1,922 System Tokens?

Every LLM inference has two distinct phases:

1. **Prefill Phase (Compute-Bound)**: The GPU must process all input tokens (`1,922 system + ~44 user = 1,966 tokens`) through all 36 transformer layers to compute their Key-Value (KV) tensors before **Token #1** can be emitted. This determines **Time To First Token (TTFT)**.
2. **Decode Phase (Memory-Bandwidth-Bound)**: The GPU generates output tokens one by one (~80 tokens), reading the entire KV cache from GPU VRAM on every single step. This determines **Inter-Token Latency (ITL)**.

When 32 callers hit your server concurrently (`C=32`) with dynamic per-caller context (where prefix caching cannot help):

- The GPU tensor cores choke trying to prefill $32 \times 1,966 = 62,912$ tokens.
- On a real **NVIDIA Tesla T4 (16GB)** running `vLLM`, P50 TTFT explodes to **24,118.82 ms (24.1 seconds!)** and P95 TTFT hits **44,472.91 ms (44.5 seconds)**.
- Worse: even with the 1,922-token prompt, base `Qwen2.5-3B-Instruct` only achieves **70.0% full compliance** on our 80-sample holdout test set because small models lose track of constraints buried in a 2,000-token prompt ("lost in the middle").

---

## 2. The WeightBake Thesis: Context Distillation into Model Weights

**What if we bake the 1,922-token system prompt directly into the neural network's weights?**

Instead of passing the rules in context at runtime, we train the model so that seeing a tiny **10-token trigger** (`prompts/minimal_system_prompt.txt`):

```text
Respond as the deterministic voice router in raw JSON.
```

activates the exact same internal representations as the 1,922-token prompt — and actually obeys the rules *better* than the base model ever did.

### How Context Distillation Works (Step-by-Step)

```text
[Step 1: Teacher / Procedural Generation]
  Input:  1,922-token Bloated System Prompt + User Query
  Output: Candidate JSON Response
             │
             ▼
[Step 2: Deterministic Rejection Sampling (src/eval/evaluator.py)]
  Grade candidate against 7 strict Pydantic + Regex + Routing checks.
  Keep ONLY responses that score 7/7 (100% compliant).
             │
             ▼
[Step 3: Context Stripping (src/data_gen/synthesizer.py)]
  DELETE the 1,922-token prompt!
  Replace it with the 10-token minimal prompt in the saved training file:
  [{"role": "system", "content": "Respond as the deterministic voice router in raw JSON."},
   {"role": "user",   "content": user_query},
   {"role": "assistant", "content": verified_gold_json}]
             │
             ▼
[Step 4: QLoRA SFT with Assistant-Only Masking (src/train/train_sft.py)]
  Train Qwen2.5-3B-Instruct on the 10-token prompt -> gold JSON pairs.
  Mask system + user tokens with -100 so loss is ONLY computed on the JSON output.
             │
             ▼
[Step 5: KL-Anchored DPO Alignment (src/train/train_dpo.py)]
  Pair each gold JSON (chosen) with a realistic flawed output (rejected) across
  6 failure categories. Use KL-anchored DPO (beta=0.5, lr=1e-6, 15 steps) to
  lock in behavioral boundaries without suffering DPO alignment tax.
             │
             ▼
[Step 6: 16-Bit Weight Folding & vLLM Serving (src/train/merge_weights.py)]
  Mathematically fold LoRA matrices (W_merged = W_0 + (alpha/r) * B @ A) into
  16-bit weights so vLLM serves the model with zero adapter overhead.
```

---

## 3. Component-by-Component Codebase Tour

Here is how every directory and file in the repository works.

```text
WeightBake/
├── prompts/
│   ├── bloated_system_prompt.txt     # 1,922-token full specification + 6 few-shot examples
│   └── minimal_system_prompt.txt     # 10-token runtime trigger
├── configs/
│   ├── sft_config.yaml               # QLoRA SFT hyperparameters
│   └── dpo_config.yaml               # KL-anchored DPO hyperparameters
├── data/
│   ├── golden_eval.jsonl             # 16 smoke-test cases (2 per intent)
│   ├── sft_train.jsonl               # 800 context-stripped SFT conversations
│   ├── dpo_train.jsonl               # 800 DPO preference triplets (prompt, chosen, rejected)
│   └── test_holdout.jsonl            # 80 stratified holdout test cases (10 per intent)
├── src/
│   ├── cli.py                        # Unified `weightbake` CLI orchestrator
│   ├── eval/
│   │   ├── evaluator.py              # 7-check Pydantic v2 deterministic grading engine
│   │   ├── run_baseline_check.py     # Token counter & 16-sample golden smoke test
│   │   └── evaluate_checkpoint.py    # 80-sample holdout evaluator (live vLLM/HF & simulation)
│   ├── data_gen/
│   │   ├── scenarios.py              # Disjoint train/holdout entity pools & 8-intent taxonomy
│   │   ├── negative_miner.py         # 6-category hard-negative generator for DPO `rejected`
│   │   ├── synthesizer.py            # Async Teacher API & procedural synthesis + context stripping
│   │   └── validate_datasets.py      # Audits context stripping, zero leakage & class balance
│   ├── train/
│   │   ├── config.py                 # Pydantic v2 YAML config loaders with safety bounds
│   │   ├── formatting.py             # ChatML formatter & -100 assistant-only label masker
│   │   ├── train_sft.py              # Unsloth/TRL QLoRA SFT trainer & dry-run token auditor
│   │   ├── dpo_utils.py              # DPO ChatML formatter & Bradley-Terry loss/margin math
│   │   ├── train_dpo.py              # Unsloth/TRL DPO trainer (zero-VRAM ref model)
│   │   ├── merge_weights.py          # LoRA W0 + (alpha/r)BA math verifier & 16-bit exporter
│   │   └── export_hub.py             # Dynamic MODEL_CARD.md generator & HF Hub uploader
│   └── benchmark/
│       ├── kv_cache_profiler.py      # Qwen2.5-3B GQA-2 KV-cache byte/VRAM calculator
│       ├── serve_vllm.py             # vLLM OpenAI server launcher (--dtype half --enforce-eager)
│       └── load_tester.py            # Async httpx SSE streaming TTFT/ITL/RPS load tester
├── notebooks/
│   └── WeightBake_Colab_T4_Live.ipynb # Battle-tested Google Colab Tesla T4 notebook
├── results/
│   ├── live_t4_eval_report.json      # Real Tesla T4 5-stage evaluation & DPO KL ablation
│   ├── live_t4_vllm_benchmark.json   # Real Tesla T4 vLLM 0.33.0 SSE benchmark (C=1..32)
│   ├── alignment_progression_report.json
│   └── vllm_latency_benchmark.json
└── tests/                            # 73 unit & integration tests
```

---

### Component 1: The Prompts (`prompts/`)

- **`prompts/minimal_system_prompt.txt`**:
  Contains exactly 9 words (`10` `cl100k_base` tokens):
  `Respond as the deterministic voice router in raw JSON.`
- **`prompts/bloated_system_prompt.txt`**:
  Contains the full **1,922-token** enterprise specification. Comparing the two gives:

$$\text{Prefill Token Reduction} = \frac{1922 - 10}{1922} = 99.48\%$$

---

### Component 2: The Deterministic Evaluator (`src/eval/evaluator.py`)

In many LLM projects, people use "LLM-as-a-Judge" (asking GPT-4 to grade outputs), which is slow, expensive, and non-deterministic. In **WeightBake**, every single requirement is verifiable via deterministic code.

`src/eval/evaluator.py` defines Pydantic v2 models with `ConfigDict(extra="forbid")` and `StrictStr`, `StrictBool`, `StrictFloat`, `StrictInt`:

- **Top-Level Model (`VoiceRouterOutput`)**:
  Requires the exact 6 keys (`intent`, `urgency`, `requires_human`, `tool_call`, `internal_reasoning`, `spoken_response`). If the model hallucinates a 7th key like `"confidence": 0.99`, `extra="forbid"` immediately fails validation.
- **Per-Tool Polymorphic Argument Validation (`ToolCall.validate_tool_arguments`)**:
  Uses a `@model_validator(mode="after")` lookup table (`TOOL_ARG_MODELS`) that maps `self.name` to its dedicated Pydantic model:
  - `"lookup_invoice"` $\rightarrow$ `LookupInvoiceArgs(account_id: StrictStr, charge_amount: StrictFloat | StrictInt)`
  - `"run_diagnostics"` $\rightarrow$ `RunDiagnosticsArgs(service_name: StrictStr, error_code: StrictStr)`
  - `"lock_account_mfa"` $\rightarrow$ `LockAccountMfaArgs(account_id: StrictStr, lock_level: Literal["soft_reset", "hard_freeze"])`
  - `"modify_subscription"` $\rightarrow$ `ModifySubscriptionArgs(action: Literal["upgrade", "downgrade", "pause", "cancel"], target_tier: StrictStr)`
  - `"track_shipment"` $\rightarrow$ `TrackShipmentArgs(order_id: StrictStr)`
  - `"escalate_specialist"` $\rightarrow$ `EscalateSpecialistArgs(department: Literal["legal", "security", "billing_tier2", "engineering_oncall"], priority_code: Literal[1, 2, 3])` (includes a custom `@field_validator` that rejects Python booleans `True`/`False`, since `bool` is a subclass of `int` in Python!)
  - `"none"` $\rightarrow$ `NoneArgs()` (must be an empty dict `{}`).

When you call `evaluate_response(raw_output, expected_intent, expected_tool, expected_requires_human)`, it grades **7 boolean checks**:

1. `no_markdown_wrapper`: Checks that the string does not start/end with ` ``` ` or contain ` ```json `.
2. `valid_json`: Runs `json.loads()` and verifies the root is a JSON object (`dict`).
3. `strict_schema`: Runs `VoiceRouterOutput.model_validate(parsed)`.
4. `spoken_word_count_5_to_22`: Splits `spoken_response` on whitespace and checks `5 <= len(words) <= 22`.
5. `zero_sycophancy`: Scans `spoken_response` against 13 compiled word-boundary regexes (`\bcertainly\b`, `\bsure\b`, `\bi'm sorry\b`, `\bas an ai\b`, etc.) so `"sure"` is caught but `"ensure"` is allowed.
6. `tts_safe_punctuation`: Checks that no character in `spoken_response` belongs to `FORBIDDEN_TTS_CHARS = frozenset("*#`_[](){}:;/\\|<>@~")`.
7. `routing_accuracy`: Checks that `intent == expected_intent`, `tool_call.name == expected_tool`, and `requires_human == expected_requires_human`.

`passed_all` is `True` **if and only if all 7 checks pass**.

---

### Component 3: The Context-Distilled Data Engine (`src/data_gen/`)

To train SFT and DPO models without data leakage, `src/data_gen/` is split into four modules:

1. **`src/data_gen/scenarios.py` (Disjoint Entity Pools & 8-Intent Taxonomy)**:
   - Defines all 8 intents (`billing_dispute`, `technical_outage`, `account_security`, `subscription_change`, `order_logistics`, `compliance_legal`, `general_inquiry`, `adversarial_out_of_scope`) and 6 caller speech styles (`frantic`, `terse`, `polite`, `angry`, `asr_unpunctuated`, `adversarial`).
   - **Crucial Anti-Memorization Design**: It maintains **strictly disjoint entity pools** for `TRAIN` vs. `HOLDOUT`:
     - Train accounts (`AC-4821`, `ENT-902`, ...) never appear in Holdout accounts (`AC-9914`, `ENT-851`, ...).
     - Train services (`us-east-auth-gateway`, `checkout-api`, ...) never appear in Holdout services (`ap-south-media-relay`, `settlement-batch-api`, ...).
     - Train error codes, dollar amounts, and order IDs are similarly disjoint from Holdout.
   - This guarantees that when our model scores **91.25%** on `data/test_holdout.jsonl`, it has learned the **general routing rules**, not memorized specific account or service strings.

2. **`src/data_gen/negative_miner.py` (Hard-Negative Generator for DPO)**:
   - DPO requires pairs of `(chosen, rejected)` responses for the same user prompt.
   - `generate_negative(gold_json, category, ...)` takes a valid gold JSON response and deterministically corrupts it into one of **6 realistic unaligned 3B failure modes**:
     1. `markdown_fenced`: Wraps the JSON in ` ```json ... ``` ` or adds `"Sure! Here is the JSON response:"`.
     2. `sycophantic_spoken`: Injects banned phrases like `"I understand your frustration and I apologize."` into `spoken_response`.
     3. `word_count_overflow`: Appends verbose explanations to make `spoken_response` 26–38 words long.
     4. `tts_unsafe_chars`: Injects markdown asterisks (`*Status*:`), colons, slashes, `#`, or `@` into `spoken_response`.
     5. `schema_drift`: Adds an extra top-level key (`"confidence": 0.98`), deletes `"internal_reasoning"`, or uses wrong tool argument keys/types (e.g., `"priority_code": "P1"` instead of `1`).
     6. `routing_misclassification`: Flips `intent`, calls the wrong `tool_call.name`, or sets `requires_human: false` on a critical outage/breach.
   - Every generated negative is passed through `evaluate_response()` to verify `passed_all is False` before being accepted.

3. **`src/data_gen/synthesizer.py` (Synthesis + Context Stripping)**:
   - Supports two modes:
     - `--mode api`: Calls an external Teacher LLM (`AsyncOpenAI`) using the **1,922-token `bloated_system_prompt.txt`**, validates every response with `evaluate_response()`, retries up to 3 times on failure, and caches raw teacher outputs in `data/raw_cache/`.
     - `--mode procedural`: Uses the deterministic combinatorial generator in `scenarios.py`.
   - **The Context Stripping Step**: When writing `data/sft_train.jsonl` (800 rows, 100 per intent) and `data/dpo_train.jsonl` (800 rows, 100 per intent), `_format_sft_rows()` and `_format_dpo_rows()` strip out `bloated_system_prompt.txt` and write **only `minimal_system_prompt.txt` (10 tokens)** into the `"system"` message!
   - Across 1,600 training conversations, stripping 1,912 tokens per sample saves **3,059,200 prefill tokens** during training and teaches the model to associate the 10-token trigger with the full enterprise schema.

4. **`src/data_gen/validate_datasets.py` (Dataset Integrity Auditor)**:
   - Verifies that:
     - 100% of `sft_train.jsonl` and `dpo_train.jsonl` rows use the 10-token minimal system prompt (0% bloated prompt leakage).
     - 800/800 SFT responses and 800/800 DPO `chosen` responses pass `evaluate_response()`.
     - 800/800 DPO `rejected` responses fail `evaluate_response()`.
     - 0 `user_query` overlap exists between `test_holdout.jsonl` (80 rows) and the training sets.

---

### Component 4: Supervised Fine-Tuning & Assistant-Only Loss Masking

*(`src/train/formatting.py` & `src/train/train_sft.py`)*

#### ChatML Formatting (`format_chatml_messages`)

`Qwen2.5-3B-Instruct` uses the ChatML template:

```text
<|im_start|>system
Respond as the deterministic voice router in raw JSON.<|im_end|>
<|im_start|>user
Account AC-8841 was billed 499.50 twice this morning. Check that invoice right away.<|im_end|>
<|im_start|>assistant
{"intent":"billing_dispute","urgency":"high",...}<|im_end|>
```

#### Why Assistant-Only Loss Masking (`build_masked_sft_labels`) Matters

In standard language modeling, cross-entropy loss is computed over *every* token in the sequence. If you don't mask the prompt, 36.8% of your gradient updates are wasted teaching the 3B model how to predict the system prompt and the caller's question! Worse, on small datasets, unmasked training causes the model to overfit to user phrasing.

In PyTorch `CrossEntropyLoss`, any target label set to **`-100`** (`ignore_index=-100`) contributes **zero loss and zero gradient**.
`build_masked_sft_labels(input_ids, assistant_header_ids, im_end_id)` scans `input_ids` for the token sequence of `<|im_start|>assistant\n` and sets every token **before and including** that header to `-100`:

```text
Tokens:  [<|im_start|>, system, ..., <|im_start|>, assistant, \n,  {"intent":, ..., <|im_end|>, \n]
Labels:  [    -100,      -100,  ...,     -100,        -100,  -100, {"intent":, ..., <|im_end|>, -100]
         └─────────────────── Masked (36.8% of tokens) ──────────┘ └──── Supervised (63.2%) ───┘
```

If the assistant header sequence is missing from `input_ids`, `build_masked_sft_labels` raises a `ValueError` immediately so silent unmasked training can never occur.

#### QLoRA Architecture (`configs/sft_config.yaml` & `src/train/train_sft.py`)

- Base model loaded in 4-bit NormalFloat (`bnb-4bit`) via Unsloth (`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`).
- LoRA adapters (`r=16`, `lora_alpha=32`, scaling factor $\frac{\alpha}{r} = 2.0$, `lora_dropout=0.0`) attached to all **7 linear projections** in every transformer block:
  - Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - SwiGLU MLP projections: `gate_proj`, `up_proj`, `down_proj`
- Trained for 2 epochs (`760` train samples, `40` validation samples, batch size `4` $\times$ grad accum `4` = effective batch size `16`, `96` total steps) at `lr=2.0e-4` with `adamw_8bit`.
- **Result on Live Tesla T4**: SFT jumps holdout compliance from **0.0%** (unprompted base) and **70.0%** (bloated 1,922-token base) to **91.25%** using only the 10-token prompt!

---

### Component 5: Direct Preference Optimization & The KL-Regularization Discovery

*(`src/train/dpo_utils.py` & `src/train/train_dpo.py`)*

#### The Bradley-Terry DPO Objective (`compute_dpo_loss_and_metrics`)

DPO aligns the policy $\pi_\theta$ directly on `(prompt, chosen, rejected)` triplets without training a separate reward model:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \underbrace{\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}}_{\text{Chosen log-ratio}} - \underbrace{\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}_{\text{Rejected log-ratio}} \right) \right) \right]$$

- **Zero-VRAM Reference Model**: Normally, DPO requires keeping two copies of the model in GPU memory ($\pi_\theta$ and frozen $\pi_{\text{ref}}$). By calling Unsloth's `PatchDPOTrainer()` and passing `ref_model=None`, Unsloth computes $\pi_{\text{ref}}$ log-probabilities by temporarily disabling the LoRA adapters on the single 4-bit model in memory — requiring **0 MB of extra VRAM**.

#### Our Live Tesla T4 Discovery: DPO Alignment Tax vs. KL-Anchored Stability

When we ran this pipeline on the real Tesla T4 GPU, we discovered one of the most important practical lessons in modern LLM post-training:

1. **What SFT Achieved (`sft_minimal` = `91.25%`)**:
   On `Qwen2.5-3B-Instruct`, QLoRA SFT with `-100` assistant-only loss masking already achieved **100.0%** on `no_markdown_wrapper`, **100.0%** on `valid_json`, **100.0%** on `spoken_word_count_5_to_22`, **100.0%** on `zero_sycophancy`, and **100.0%** on `tts_safe_punctuation`!
2. **What Happened with Aggressive DPO (`dpo_unanchored_ablation`: $\beta=0.1, \text{LR}=5\times 10^{-6}, 50\text{ steps}$)**:
   Because the SFT model already assigned very low probability to the `rejected` responses, pushing 50 steps of DPO with a weak KL penalty ($\beta=0.1$) forced the model to aggressively distort its token distributions just to widen the reward margin further. This caused classic **DPO over-optimization (alignment tax)**:
   - `strict_schema` degraded from `97.5%` $\rightarrow$ `92.5%`
   - `routing_accuracy` degraded from `93.75%` $\rightarrow$ `90.0%`
   - Overall holdout compliance dropped from **`91.25%` $\rightarrow$ `83.75%`**.
3. **The Fix — KL-Anchored DPO (`dpo_minimal`: $\beta=0.5, \text{LR}=1\times 10^{-6}, 15\text{ steps}$)**:
   By increasing $\beta$ 5x (`0.1 -> 0.5`, which penalizes KL divergence away from $\pi_{\text{ref}}$ 5x more strongly), lowering the learning rate 5x (`5e-6 -> 1e-6`), and limiting training to `15` steps (`warmup_steps=3`), we anchored the policy to the SFT manifold — **completely eliminating the alignment tax** and preserving **`91.25%` full compliance** (+21.25 percentage points above the 1,922-token baseline).

---

### Component 6: 16-Bit LoRA Weight Folding

*(`src/train/merge_weights.py`)*

If you serve a model in `vLLM` with unmerged LoRA adapters, every forward pass has to execute two separate matrix multiplications: the base weight $W_0 x$ plus the low-rank adapter $\frac{\alpha}{r} B (A x)$.
Because LoRA is purely linear, `src/train/merge_weights.py` folds the adapter directly into the base weight matrix before serving:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$

`verify_lora_merge_math()` proves mathematically and numerically across all 7 Qwen2.5-3B projection shapes (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) that:

$$\left\| \left(W_0 x + \frac{\alpha}{r} B(Ax)\right) - W_{\text{merged}} x \right\|_\infty \le 3.81 \times 10^{-6}$$

Then `save_pretrained_merged(..., save_method="merged_16bit")` writes standard 16-bit safetensors to `checkpoints/weightbake-3b-merged` so `vLLM` runs at full native speed.

---

### Component 7: KV-Cache Physics & `vLLM` Streaming Benchmark

*(`src/benchmark/`)*

#### 1. Exact KV-Cache Sizing (`src/benchmark/kv_cache_profiler.py`)

`Qwen2.5-3B` uses **Grouped-Query Attention (GQA)** with:

- $L = 36$ transformer layers (`num_hidden_layers`)
- $H_{kv} = 2$ key-value heads (`num_key_value_heads`, shared across 16 query heads)
- $D = 128$ head dimension (`head_dim`)
- 2 tensors per layer (Key + Value) $\times$ 2 bytes per parameter (FP16/BF16)

$$\text{KV Bytes per Token} = 2 \times 36 \times 2 \times 128 \times 2 = 36,864\text{ bytes} = 36.0\text{ KB/token}$$

Now compare a single active voice call (`44` user tokens + `80` output tokens):

- **Baseline (1,922 system tokens)**:
  $$(1922 + 44 + 80) \times 36\text{ KB} = 2,046 \times 36\text{ KB} = \mathbf{71.93\text{ MiB per call}}$$
- **WeightBake-3B (10 system tokens)**:
  $$(10 + 44 + 80) \times 36\text{ KB} = 134 \times 36\text{ KB} = \mathbf{4.71\text{ MiB per call}}$$

That is a **15.27x reduction in KV-cache VRAM per call**. On an 8.0 GiB KV-cache budget, max concurrent sessions jump from **113 sessions $\rightarrow$ 1,738 sessions** (**15.38x capacity**).

#### 2. `vLLM` Server Wrapper (`src/benchmark/serve_vllm.py`)

Wraps `python -m vllm.entrypoints.openai.api_server`. For NVIDIA Tesla T4 GPUs (Turing `sm_75`, which does not support native `bfloat16`), it supports `--dtype half --enforce-eager --disable-prefix-caching` (which passes `--no-enable-prefix-caching` to `vLLM`).

#### 3. Async SSE Streaming Load Tester (`src/benchmark/load_tester.py`)

`stream_chat_completion()` uses `httpx.AsyncClient` to open a raw Server-Sent Events (`"stream": True`) connection to `/v1/chat/completions` and records high-resolution `time.perf_counter()` timestamps on every token chunk:

- **`ttft_ms`**: `t_first_non_empty_token - t_request_start`
- **`itl_ms`**: `(t_chunk[i] - t_chunk[i-1])` averaged across all generated tokens
- **`e2el_ms`**: `t_stream_done - t_request_start`
- **`rps`**: `num_requests / total_wall_clock_time`

---

## 4. Summary of All Verified Results (Live Tesla T4 Hardware)

### Table A: 80-Sample Stratified Holdout Compliance (`results/live_t4_eval_report.json`)

| Stage | System Tokens | Full Compliance | Valid JSON | Strict Schema | Spoken 5–22 Words | Zero Sycophancy | TTS-Safe Punct | Routing Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_unprompted` | 10 | 0.0% | 93.75% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `base_bloated` | 1,922 | 70.0% | 97.5% | 95.0% | 97.5% | 97.5% | 93.75% | 73.75% |
| `sft_minimal` (`r=16, alpha=32`) | 10 | **91.25%** | 100.0% | 97.5% | 100.0% | 100.0% | 100.0% | 93.75% |
| `dpo_unanchored_ablation` (`beta=0.1, 50 steps`) | 10 | 83.75% | 100.0% | 92.5% | 95.0% | 95.0% | 95.0% | 90.0% |
| `dpo_minimal` (`beta=0.5, 15 steps`) | **10** | **91.25%** | **100.0%** | **97.5%** | **100.0%** | **100.0%** | **100.0%** | **93.75%** |

### Table B: Live Tesla T4 `vLLM 0.33.0` Latency & Throughput (`results/live_t4_vllm_benchmark.json`)

| Concurrency | Baseline P50 / P95 TTFT | WeightBake P50 / P95 TTFT | TTFT Speedup (P50 / P95) | Mean ITL (Base $\rightarrow$ WB) | P95 E2EL (Base $\rightarrow$ WB) | Throughput RPS (Base $\rightarrow$ WB) |
|---|---:|---:|---:|---:|---:|---:|
| **`C = 1`** | 1,452.85 / 1,473.43 ms | **92.14 / 121.63 ms** | **15.77x** / **12.11x** | 46.23 $\rightarrow$ **46.01 ms** | 6,306.62 $\rightarrow$ **4,772.47 ms** | 0.20 $\rightarrow$ **0.29** (**1.45x**) |
| **`C = 8`** | 1,559.73 / 9,413.11 ms | **139.75 / 188.96 ms** | **11.16x** / **49.82x** | 159.85 $\rightarrow$ **47.56 ms** | 23,761.75 $\rightarrow$ **4,918.91 ms** | 0.51 $\rightarrow$ **1.95** (**3.82x**) |
| **`C = 16`** | 2,986.33 / 21,110.16 ms | **240.72 / 321.37 ms** | **12.41x** / **65.69x** | 257.99 $\rightarrow$ **61.04 ms** | 46,963.07 $\rightarrow$ **6,299.50 ms** | 0.57 $\rightarrow$ **2.84** (**4.98x**) |
| **`C = 32`** | 24,118.82 / 44,472.91 ms | **561.16 / 567.62 ms** | **42.98x** / **78.35x** | 359.87 $\rightarrow$ **50.66 ms** | 53,300.85 $\rightarrow$ **5,457.46 ms** | 0.60 $\rightarrow$ **5.86** (**9.77x**) |

---

## 5. How to Run & Explore Everything Yourself

Every subsystem is wired into `src/cli.py` so you can run individual steps or the entire pipeline with a single command:

```bash
# 1. Run all 73 unit and integration tests
python -m pytest -v

# 2. Run the full 10-step end-to-end verification pipeline & executive report
python -m src.cli verify-all

# 3. Inspect individual stages
python -m src.cli audit                                      # Token count & 16-case golden check
python -m src.cli synthesize --mode procedural               # Generate & validate 800 SFT + 800 DPO + 80 Holdout
python -m src.cli train-sft --config configs/sft_config.yaml --dry-run  # Audit -100 token masking
python -m src.cli train-dpo --config configs/dpo_config.yaml --dry-run  # Audit DPO math & token lengths
python -m src.cli merge --dry-run                            # Verify W0 + (alpha/r)BA merge equivalence
python -m src.cli benchmark --mode simulate --concurrency 1 8 16 32     # KV-cache + latency benchmark
```

And to run the full GPU training and `vLLM` benchmark on Google Colab, open `notebooks/WeightBake_Colab_T4_Live.ipynb` directly from <https://github.com/sarthexe/WeightBake>.

---

*WeightBake v0.2.0 — Apache-2.0 — <https://github.com/sarthexe/WeightBake>*
