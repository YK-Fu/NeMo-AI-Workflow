# NeMo LLM Training Guide

This folder shows you how to train Large Language Models (LLMs) using NVIDIA's NeMo toolkit. It includes everything you need to know, from the first training steps to making your model work better.

## How to Train Your Model

### 1. Pre-training and Fine-tuning
- **Pre-training**: Learn how to train a new model or improve an existing one
  - [See the Pre-training Guide](pretrain/README.md)
- **Fine-tuning**: Make your model better at specific tasks
  - [See the Fine-tuning Guide](sft/README.md)

### 2. Downsize Your Models
- **Pruning**: Make your model smaller without losing quality
- **Distillation**: Copy knowledge from a big model to a smaller one
  - [See the Distillation Guide](distil/README.md)

### 3. Model Alignment
- Teach your model to be more helpful and follow instructions (using methods like RLHF and DPO)
- [See the Alignment Guide](align/README.md)

## Folder Structure

```
llm/
├── ckpt_conversion/  # Tools to change model file formats
├── pretrain/         # Examples for first-time training
├── sft/              # Examples for fine-tuning
├── distil/           # Tools for making models smaller
└── align/            # Examples for making models more helpful
```
