# TensorRT-LLM

TensorRT-LLM is NVIDIA's high-performance inference solution for Large Language Models (LLMs). It provides optimized inference performance through TensorRT, NVIDIA's high-performance deep learning inference optimizer and runtime.

## Overview

TensorRT-LLM offers several key benefits:
- Optimized inference performance for LLMs
- Support for various model architectures (GPT, LLaMA, Falcon, etc.)
- Advanced features like KV-cache management and continuous batching
- FP8 and BF16 precision support
- Tensor and pipeline parallelism

## Exporting Models to TensorRT-LLM

The `export_to_trt_llm.py` script allows you to convert your NeMo (1 or 2) or Hugging Face models to TensorRT-LLM format. Here's how to use it:

### Basic Usage

```bash
python export_to_trt_llm.py \
    --nemo-model <path_to_nemo_checkpoint> \  # or --hf-model <path_to_hf_model>, choose one
    --model-type <model_type> \               # e.g., gpt, llama, falcon
    --output-dir <output_directory> \
    --dtype <precision> \                     # bfloat16 or float16
    -TP 1 \                                   # Tensor parallel
    -PP 1 \                                   # Pipeline parallel
    --max-batch-size <batch_size> \
    --max-input-len <input_length> \
    --max-output-len <output_length>
```

### Key Parameters

#### Required Parameters:
- `--nemo-model` or `--hf-model`: Path to your source model
- `--model-type`: Type of model (gpt, llama, falcon, starcoder, mixtral, gemma)
- `--output-dir`: Directory to save the TensorRT-LLM engine
- `--dtype`: Data type for the model (bfloat16 or float16)

### Advanced Features (only NeMo ckpts are supported)

#### LoRA Support
```bash
python export_to_trt_llm.py \
    --use-lora-plugin float16 \
    --lora-target-modules attn_qkv attn_dense \
    --max-lora-rank 64
```

#### FP8 Quantization
```bash
python export_to_trt_llm.py \
    --fp8 True \
    --fp8-kv-cache True
```


## Example Commands

## Notes
- The export process may take some time depending on the model size and your hardware
- Make sure you have sufficient GPU memory for the export process
- For large models, consider using tensor or pipeline parallelism
- The exported model can be used with NVIDIA's triton inference server (see [TRITON.md](./TRITON.md) for deployment instructions) or NIM (see [NIM.md](./NIM.md) for deployment instructions)
