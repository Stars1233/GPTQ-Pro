# GPTQ-Pro

**GPTQ-Pro** is an experimental, performance-focused fork of
[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel), tuned for practical local quantization and inference work on modern NVIDIA consumer GPUs, especially Ampere-class cards such as the RTX 3090 and RTX 3060.

This project keeps the excellent GPT-QModel foundation intact, while adding a GPTQ-Pro research path around INT4 kernels, activation-aware quantization improvements, Ampere CUDA compatibility, Qwen-family workflows, vLLM serving helpers, and local validation tooling.

> This is not the official ModelCloud release.  
> For stable upstream usage, use [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel).  
> This fork is for GPTQ-Pro experimentation, local benchmarking, kernel validation, and practical model-quantization workflows.

> 📋 See [`docs/ASSESSMENT_AND_ROADMAP.md`](docs/ASSESSMENT_AND_ROADMAP.md) for a fact-checked
> assessment of current quantization quality and a prioritized, Ampere-focused improvement roadmap.

---

## Credits

This fork exists because the upstream work is already strong.

Massive credit to **Qubitium** and the **ModelCloud team** for building and maintaining GPT-QModel, one of the most complete modern GPTQ/AWQ/LLM quantization toolkits available.

Additional credit to:

- **Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh** for the original GPTQ and Marlin work.
- **PanQiWei** for AutoGPTQ, which GPT-QModel is historically based on.
- **FXMarty** for maintaining and supporting AutoGPTQ.
- **Qwopqwop200** for GPTQ-for-LLaMa quantization work.
- **Turboderp** for ExLlama / ExLlamaV2 kernels.
- **FpgaMiner** for GPTQ-Triton kernels.
- **Casper Hansen** for AutoAWQ, which helped shape early AWQ integration.

GPTQ-Pro is a fork, not a reinvention. The upstream authors did the hard foundational engineering. This branch adds a more aggressive local-performance research track on top.

---

## What GPTQ-Pro adds

GPTQ-Pro focuses on the parts that matter when you are actually quantizing and serving models on local hardware:

- **GPTQ-Pro INT4 dequant GEMM path**
- **FP32-accumulator GPTQ-Pro kernel work**
- **GPTQ-Pro as the unconditional default kernel** — auto-selected at top priority (120) above Marlin/ExLlama wherever its device check passes
- **Activation-weighted GPTQ-Pro scale search**
- **Adaptive GPTQ-Pro smoothing / failsafe logic**
- **Named quantization-recipe presets** — `fast_4bit`, `quality_4bit`, `max_quality_4bit`, and `experimental_3bit_rotation` (see [Quantization recipe presets](#quantization-recipe-presets))
- **Qwen3.5-MoE support** — text-only and multimodal (vision tower kept unquantized), MTP passthrough preserved, plus a reusable end-to-end quant script
- **Qwen3.5 quantization, smoke tests, benchmark notes, and vLLM workflows**
- **Local inference wiring for GPTQ-Pro**
- **Gemma 4 GPTQ package validation**
- **RTX 3090 / RTX 3060 Ampere validation**
- **Ampere-focused CUDA build flags for `sm_80`, `sm_86`, and `sm_87`**
- **PTX fallback gencode for better forward compatibility**
- **GPTQ-only quantization restrictions where mixed paths are unsafe**
- **Validation harnesses and regression tests for the experimental path**
- **transformers 5.12 hub compatibility**

The intent is simple: make GPTQ-Pro more useful for people running serious local LLM infrastructure, not just clean-room benchmark demos.

---

## Tested / targeted hardware

This fork is primarily developed around local NVIDIA Ampere hardware:

- RTX 3090
- RTX 3060
- CUDA `sm_86`
- Multi-GPU local quantization and inference workflows

Other CUDA GPUs may work, especially where upstream GPT-QModel already supports them, but the GPTQ-Pro path is optimized and validated first against Ampere-class consumer cards.

---

## Relationship to GPT-QModel

GPTQ-Pro inherits the core GPT-QModel feature surface, including support for:

- GPTQ
- AWQ
- Marlin
- vLLM
- SGLang
- Hugging Face Transformers
- CPU / CUDA / ROCm / XPU paths where supported upstream
- Modern model-family support inherited from GPT-QModel

This fork does **not** try to replace GPT-QModel. It keeps GPT-QModel as the base and layers experimental GPTQ-Pro improvements on top.

Use upstream GPT-QModel when you want the most stable general-purpose package.

Use GPTQ-Pro when you want to test the experimental path, especially around local CUDA kernels, Ampere hardware, Qwen-family workflows, and custom quantization validation.

---

## Install from source

```bash
git clone https://github.com/groxaxo/GPTQ-Pro.git
cd GPTQ-Pro

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -e .
```

> Recommended toolchain: Python 3.13+, PyTorch ≥ 2.8, CUDA 12.x. The build links CUDA
> kernels for Ampere (`sm_80/86/87`) with a PTX fallback for newer cards. To match the
> CUDA toolkit to your installed PyTorch, run `./sync_cuda_toolkit_with_torch.sh`.

After install, verify the package imports and reports the GPTQ-Pro build:

```bash
python -c "import gptqmodel; print(gptqmodel.__version__)"
```

---

## Quickstart

```python
from gptqmodel import GPTQModel, QuantizeConfig

# Balanced 4-bit GPTQ-Pro recipe (GAR + MSE scale search + activation-weighted MSE).
qcfg = QuantizeConfig.quality_4bit(group_size=128)

model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct", qcfg)
model.quantize(["the quick brown fox " * 40] * 16, batch_size=1)
model.save("qwen2.5-0.5b-gptq-pro-4bit")
```

At load time, the GPTQ-Pro kernel is auto-selected at top priority (120) on supported
Ampere+ GPUs — no flags required. The emitted checkpoint stays in **standard GPTQ format**,
so it also runs unchanged under GPTQ / Marlin / ExLlama / vLLM kernels.

### Quantization recipe presets

These are *quantization-time* recipes (independent of the runtime inference kernel). Higher
presets cost more time to quantize but improve accuracy; all emit standard GPTQ checkpoints:

| Preset | Builder | What it adds |
| --- | --- | --- |
| Fast | `QuantizeConfig.fast_4bit()` | Base GPTQ defaults (group-aware reordering on; MSE search & GPTAQ off) |
| Quality | `QuantizeConfig.quality_4bit()` | `gptq_pro()`: GAR + MSE scale search + activation-weighted MSE + adaptive damping |
| Max quality | `QuantizeConfig.max_quality_4bit()` | `quality_4bit` plus GPTAQ activation-aware error feedback (GPTQv2) |
| Experimental 3-bit | `QuantizeConfig.experimental_3bit_rotation()` | 3-bit `max_quality` + Hadamard incoherence rotation (gated to llama/qwen2) |

### Qwen3.5-MoE quantization

GPTQ-Pro supports Qwen3.5-MoE in both text-only and multimodal forms. The vision tower
(`model.visual.*`) and MTP (`mtp.*`) modules are carried through unquantized; only the
language-model decoder layers are quantized. A reusable end-to-end script is provided:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/quant_qwen3_5_moe.py \
  --model <hf_id_or_local_path> --out out-4bit \
  --calib image --nsample 16 --preset quality --offload-disk
```

Use `--offload-disk` and a single GPU for large (256-expert) MoE checkpoints on ≤24 GB
cards — multi-GPU replicates each layer across cards during calibration and can OOM. See the
script header and [`docs/qwen35_vllm_launch.md`](docs/qwen35_vllm_launch.md) for vLLM serving.
