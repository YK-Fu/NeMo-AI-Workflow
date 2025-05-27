# Pretraining Script Guide

This document explains how to use the `pretrain_llm.py` script for pretraining the large language model. The script is built on the NVIDIA NeMo platform and utilizes PyTorch Lightning with NeMo for pretraining tasks.

---

## End-to-end pipeline
### Download and Convert Checkpoints (Optional)
If you want to train from scratch, you can skip this part; otherwise, you need to download an existing model from Huggingface for continual pretraining:
```bash
huggingface-cli download ${HF_MODEL_ID} --repo-type --local-dir ${HF_MODEL_DIR}
``` 
We need to convert the model from Huggingface format to nemo2 format: 
```bash
cd /home/ubuntu/ifu/NeMo-AI-Workflow/train/llm/ckpt_conversion/
python convert_from_hf_to_nemo2.py \
    --source ${HF_MODEL_DIR} \
    --output-path ${NEMO2_MODEL_DIR} \
    --model-name ${NEMO_MODEL_NAME} \
    --model-config-name ${NEMO_CONFIG_NAME} \
    --overwrite           # whether to overwrite existing output-path
```
See `supported_config.txt` and `supported_model.txt` to check supported configurations and models.

Verify the converted model directory structure:
```
${NEMO2_MODEL_DIR}
    ├─ context
    └─ weights
```

### Download Example Data (Optional)
We can download Wiki Traditional Chinese dataset as an example:
```
cd /${WORKSPACE}/NeMo-AI-Workflow/download_example_data/
python download_wiki.py
```
The data will be downloaded to `/${WORKSPACE}/NeMo-AI-Workflow/download_example_data/wiki_data_zh-classical/zh_classicalwiki-{DATE}-pages-articles-multistream.jsonl`

#### Raw Dataset Format
Pretraining requires a dataset in JSON Lines format (one JSON object per line). Each JSON object should contain a `text` field to store the text content.

Using Wikinews as an example, the data format is as follows:

```
{"text":"Republic of China..."}
{"text":"Recently, a programmer..."}
```

To replace with other datasets, you can import JSON Lines in the same format.

#### Dataset Preprocess
For pretraining usually involves access to large datasets, so we need to first preprocess the dataset for efficient data fetching:
```bash
bash preprocess_data.sh \
    --ht-tokenizer ${HF_MODEL_DIR} \
    --data-prefix zh_classicalwiki-{DATE}-pages-articles-multistream \      # jsonl filename without its parents and extension
    --input-dir ${DATASET_DIR} \
    --output-dir ${PROCESSED_DATASET_DIR}
```
After preprocessing, data should be stored in `.bin` and `.idx` formats. Verify the directory structure before model training:
```
<PROCESSED_DATASET_DIR>
└─ preprocessed
    ├─ zh_classicalwiki-{DATE}-pages-articles-multistream_text_document.bin
    └─ zh_classicalwiki-{DATE}-pages-articles-multistream_text_document.idx
```

---

## Start Pretraining

Execute `pretrain_llm.py`:

```bash
JOB_NAME=llama31_pretraining

NUM_NODES=1
NUM_GPUS=8;

HF_MODEL_ID=Llama-3.1-8B-Instruct

TP=4
PP=1
CP=1

GBS=2048
MAX_STEPS=100
cd /${WORKSPACE}/NeMo-AI-Workflow/train/llm/pretrain/

python pretrain_llm.py \
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
    --dataset_path ${PROCESSED_DATASET_DIR}
```

---

## **Parameter Description**

Below is a description of the parameters for the `pretrain_llm.py` training script. You can configure the model training method through commandline parameters, such as selecting the execution environment, specifying the model, and setting batch sizes.

### **Experiment Execution Method**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--executor` | `str` | `local` | Choose execution method, options are `slurm` (using Slurm cluster) or `local` (single machine execution) |
| `-E, --experiment` | `str` | `pretraining` | Set experiment name, which affects output folder naming |

---

### **Slurm Parameter Settings**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-A, --account` | `str` | `root` | Slurm account name, applicable for HPC environments requiring multi-account management |
| `-P, --partition` | `str` | `defq` | Slurm cluster partition name, different clusters may have different partitions |
| `-I, --container-image` | `str` | `nvcr.io/nvidia/nemo:dev` | Specify the NeMo Docker container image to execute (typically provided by NVIDIA NGC) |

---

### **Hardware Settings**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-N, --num-nodes` | `int` | `1` | Set the number of compute nodes to use (applicable for multi-machine environments) |
| `-G, --num-gpus` | `int` | `8` | Number of GPUs to use per compute node (for single node training, typically set to available GPU count) |

---

### **Model Settings**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-M, --model-name` | `str` | **(Required)** | See `supported_model_name.txt` for supported models |
| `--hf-model-id` | `str` | **(Required)** | Specify the Hugging Face model ID to use, e.g., `"meta-llama/Llama-3.1-8B-Instruct"` |
| `--nemo-model` | `str` | `None` | Specify the path to pretrained NeMo model weights |
| `--hf-token` | `str` | `os.getenv("HF_TOKEN)` | Hugging Face API Token for downloading tokenizer |

---

### **Training Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-steps` | `int` | `None` | Set maximum training steps (if not set, calculated based on dataset size for 1 epoch) |
| `-gbs, --global-batch-size` | `int` | `2048` | Global batch size for training, must be a multiple of `micro_batch_size * num_model_replica (data_parallel_size)` |
| `-mbs, --micro_batch_size` | `int` | `1` | Set micro batch size, typically determined by single GPU memory capacity |
| `--fp8` | `store_true` | `false` | Enable fp8 AMP training |

---

### **Model Parallelization Settings**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-TP, --tensor-model-parallel-size` | `int` | `1` | Set Tensor Model Parallelism |
| `-PP, --pipeline-model-parallel-size` | `int` | `1` | Set Pipeline Model Parallelism |
| `-VPP, --virtual-pipeline-model-parallel-size` | `int` | `1` | 
| `-CP, --context-parallel-size` | `int` | `1` | Set Context Parallelism |
| `-SP, --sequence-parallel` | `store_true` | `false` | Enable Context Parallelism |


### **Dataset Path**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-dir` | `str` | **(Required)** | Set the processed training data folder path, which should contain .bin and .idx files |

---

### **Checkpoint Recovery**

If training is interrupted, it can be automatically resumed using the following parameter:

- **Strategy**: `resume_if_exists=True`

Re-executing the training script will automatically check for existing checkpoints and resume training.

---

### **Training Process Output and Logging**

Training process outputs are logged in the `nemo_experiments` directory and support real-time monitoring.

You can also use WandB logger:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--wandb` | `store_true` | `false` | Enable WandB logger |
| `--wandb-project` | `str` | `None` | WandB project name |
| `--wandb-name` | `str` | `None` | WandB experiment name |
| `--wandb-token` | `str` | `os.getenv("WANDB_API_KEY")` | WandB API key for login |

---

### **Activation Checkpointing**
If hardware is limited, and often result in cuda out of memory issues, you can enable activations checkpointing to reduce GPU RAM by recompute the activations.
Note that for FlashAttention, self-attention recomputation is always enabled, which equals to `selective` checkpointing, so choosing `selective` granularity might not save more memory than default setting.
When training with the (virtual) pipeline parallelism, recompute-layers indicates the layers per (virtual) pipeline stage.
See more details about how NeMo implement [Activation Checkpointing](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/activation_recomputation.html#transformer-layer-recomputation).
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--recompute-granularity` | `str` | `None` | Choose from `full` or `selective` to enable gradient recomputation |
| `--recompute-method` | `str` | `None` | For `full` granularity, you should choose method from `block` or `uniform` |
| `--recompute-num-layers` | `int` | `None` | For `full` granularity, specify the number of layers to perform gradient recomputation |
---

### **CPU offloading**
If hardware is extremely limited, even activation checkpointing can not solve cuda out of memory issues, you might need to offload some computing intermediates to CPU. This feature will significantly increase the training time, and can not be used with activation checkpointing.
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cpu-offloading` | `store_true` | `false` | Enable CPU offloading |
| `--cpu-offloading-layers` | `int` | `None` | Number of layers to perform CPU offloading |
| `--cpu-offloading-activations` | `store_true` | `false` | Whether to offload activations |
| `--cpu-offloading-weights` | `store_true` | `false` | Whether to offload model weights |
| `--optim-cpu-offloading` | `store_true` | `false` | Enable optimizer states CPU offloading |
| `--optim-cpu-offloading-frac` | `float` | `1.0` | Fraction of offloaded optimizer states |

---


## Convert to Huggingface format (Optional)
You can also translate the trained model back to Huggingface format:
```
cd /home/ubuntu/ifu/NeMo-AI-Workflow/train/llm/ckpt_conversion/
python convert_from_nemo2_to_hf.py \
    --source ${NEMO2_MODEL_DIR} \
    --output-path ${OUTPUT_DIR}
```