# Supervised Finetuning Script Guide

This document explains how to use the `finetune_llm.py` script for pretraining the large language model. The script is built on the NVIDIA NeMo platform and utilizes PyTorch Lightning with NeMo for pretraining tasks.

---

## End-to-end pipeline
### Download and Convert Checkpoints (Optional)
See [Checkpoint Conversion](../pretrain/README.md#download-and-convert-checkpoints-optional) for more details.

### Download Example Data (Optional)
We can download Traditional Chinese SQUAD dataset as an example:
```
cd /${WORKSPACE}/NeMo-AI-Workflow/download_example_data/
python download_squad_zhtw.py
```
The data will be downloaded to `/${WORKSPACE}/NeMo-AI-Workflow/download_example_data/squad_zhtw/`. The directory structure is as follows:
```
/${WORKSPACE}/NeMo-AI-Workflow/download_example_data/squad_zhtw/
    ├─ train.jsonl
    ├─ validation.jsonl
    └─ test.jsonl
```

#### Dataset Format
Supervised finetuning requires a dataset in JSON Lines format (one JSON object per line). Each JSON object should contain a `input` field and a `output` field to store the user input and machine response.

The data format is as follows:
```
{"input":"When was the Republic of China established?", "output": "1912."}
{"input":"What is the capital of France?", "output": "Paris."}
```

---

## Start Pretraining

Execute `finetune_llm.py`:

```bash
JOB_NAME=llama31_sft

NUM_NODES=1
NUM_GPUS=8

HF_MODEL_ID=Llama-3.1-8B-Instruct

TP=4
PP=1
CP=1

GBS=2048
MAX_STEPS=100
cd /${WORKSPACE}/NeMo-AI-Workflow/train/llm/sft/

python finetune_llm.py \
    --executor local \
    --experiment ${JOB_NAME} \
    --num-gpus ${NUM_GPUS} \
    --hf-model-id ${HF_MODEL_ID} \
    --nemo-model ${NEMO2_MODEL_DIR} \
    --max-steps ${MAX_STEPS} \
    --global-batch-size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_DIR}
```

---

## Parameter Description

Below is a description of the parameters for the `finetune_llm.py` training script. See more parameters in [parameter](../pretrain/README.md#parameter-description)

## **Parameter Efficient Finetuning**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--peft` | `str` | `None` | Choose from `lora` or `dora` to enable PEFT |
| `--peft-target-modules` | `list` | `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]` | Modules to apply PEFT |
| `--peft-alpha` | `int` | `32` | PEFT alpha |
| `--peft-dim` | `int` | `16` | PEFT dimension |
---

## **Dataset Setting**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-dir` | `str` | **(Required)** | Path to the folder containing the dataset. This folder should include files named in the format: `training.jsonl`, `validation.jsonl`, `test.jsonl` |
| `--prompt-template` | `str` | [prompt-template](./finetune_llm.py#27) | prompt template to formulate training samples from json

---

## Convert to Huggingface format (Optional)
See [HF Checkpoint Conversion](../pretrain/README.md#convert-to-huggingface-format-optional) for more details.