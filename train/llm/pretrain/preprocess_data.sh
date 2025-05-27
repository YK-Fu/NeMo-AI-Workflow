#!/bin/bash
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-tokenizer)
            hf_tokenizer="$2"
            shift 2
            ;;
        --data-prefix)
            data_prefix="$2"
            shift 2
            ;;
        --input-dir)
            input_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
input_path=$input_dir/$data_prefix.jsonl

# Create output directory
mkdir -p $output_dir

# Run preprocessing
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=$input_path \
    --json-keys=text \
    --dataset-impl=mmap \
    --tokenizer-library=huggingface \
    --tokenizer-type=$hf_tokenizer \
    --output-prefix=$output_dir/$data_prefix \
    --append-eod