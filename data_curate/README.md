# NeMo Data Curator

This directory contains tools for downloading, processing, and filtering data for LLM training.

## Wiki Page Downloader

`nemo_wiki_downloader.py` is an example script that demonstrates how to download Traditional Chinese Wikipedia pages and convert them to JSONL format. You can use this as a template to create your own data downloaders for different sources.

Example usage:
```bash
python nemo_wiki_downloader.py
```

## Data Filtering Tools

### Pre-training Data Filtering (`pretrain_data_filter.py`)

This module provides various filtering strategies for pre-training data:

1. **Deduplication Methods**:
   - Exact Deduplication: Available in both CPU and GPU modes
   - Fuzzy Deduplication: GPU mode only
   - Semantic Deduplication: GPU mode only

2. **Other Filtering Strategies** (CPU-based):
   - Cleaning: Unicode fixing 
   - Language Identification (LID): language filtering
   - PII: Personal information removing (name is only supported in English)
   - Heuristic: Remove too short document and statistically unreasonable sentences.

Example usage:
```bash
# input-dir contains several *.jsonl files
# output-dir will use the input file name as file names
python pretrain_data.py \
    --input-dir /path/to/input/directory \
    --output-dir /path/to/output/directory \
    --cleaning \
    --lid \
    --pii \
    --language zh \
    --exact \
    --fuzzy \
    --semantic \
    --heuristic \
    --device gpu
```

### SFT Data Filtering (`sft_data_filter.py`)

This tool is specifically designed for filtering Supervised Fine-Tuning (SFT) data. It takes a JSONL file as input where each line contains an input/output pair in the format:
```json
{"input": "Please explain the main functions of Python.", "output": "Python is a high-performance programming language that is easy to learn, has a powerful standard library, and supports a wide range of applications."}
```

The tool uses NeMotron NIM to score input/output pairs based on various metrics:
- Helpfulness
- Correctness
- Coherence
- Complexity
- Verbosity

You can filter the data based on your preferred metrics and thresholds.

Example usage:
```bash
export NVIDIA_API_KEY="<YOUR_NIM_API_KEY>"
python sft_data_filter.py \
    --input-path /path/to/input/files \
    --output-dir /path/to/output/directory \
    --filter-field helpfulness \
    --threshold 3.5
```

## Requirements

- For GPU-based operations:
  - CUDA-compatible GPU
  - PyTorch with CUDA support
  - NeMotron for SFT filtering

- For CPU-based operations:
  - Sufficient RAM for large datasets
  - NVIDIA NIM API keys for SFT data filtering

## Best Practices
1. Use [NeMo docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) `nvcr.io/nvidia/nemo` from NGC (version `25.04.rc2`)
2. Always start with a small subset of data to test your filtering pipeline
3. Monitor memory usage when processing large datasets
4. Use GPU mode for faster processing when available
5. Keep original data backed up before applying filters
6. Document your filtering criteria and thresholds
