# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import pprint
from typing import Optional

from nemo.export.tensorrt_llm import TensorRTLLM

LOGGER = logging.getLogger("NeMo")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Exports NeMo checkpoint to TensorRT-LLM engine",
    )
    parser.add_argument("--nemo-model", default=None, type=str, help="Source model path")
    parser.add_argument("--hf-model", default=None, type=str, help="HF model path or name")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of the TensorRT-LLM model.",
    )
    parser.add_argument(
        "--output-dir", required=True, default=None, type=str, help="Folder for the trt-llm model files"
    )
    parser.add_argument(
        "--multi-block-mode",
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
            It is beneifical when batchxnum_heads cannot fully utilize GPU. \
            available when using c++ runtime.',
    )
    parser.add_argument("--debug-mode", action='store_true', help="Enable debug mode")

    # Common model parameters
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="Data type of the model on TensorRT-LLM")
    parser.add_argument("-TP", "--tensor-model-parallel-size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-PP", "--pipeline-model-parallel-size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument("--seq-length", default=8192, type=int, help="Max number of tokens")
    parser.add_argument("--max-input-len", default=1024, type=int, help="Max input length of the model")
    parser.add_argument("--max-output-len", default=1024, type=int, help="Max output length of the model")
    parser.add_argument("--max-batch-size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("--opt-num-tokens", default=None, type=int, help="Optimum number of tokens")
    parser.add_argument("--max-prompt-embedding-table-size", default=None, type=int, help="Max prompt embedding table size")
    parser.add_argument("--use-parallel-embedding", action='store_true', help="Use parallel embedding.")
    parser.add_argument("--use-embedding-sharing", action='store_true', help="Use embedding sharing.")
    parser.add_argument("--no-paged-kv-cache", default=False, action='store_true', help="Disable paged kv cache.")
    parser.add_argument("--disable-remove-input-padding", default=False, action='store_true', help="Disable remove input padding.")
    parser.add_argument("--paged-context-fmha", default=False, action='store_true', help="Use paged context fmha.")
    parser.add_argument("--delete-existing-files", default=True, action='store_true', help="Delete existing files in the output directory.")
    parser.add_argument("--multiple-profiles", default=False, action='store_true', help="Enable multiple profiles.")

    # NeMo-related parameters
    parser.add_argument(
        "--use-mcore-path",
        action="store_true",
        help="Use Megatron-Core implementation on exporting the model. If not set, use local NeMo codebase",
    ) 
    # LoRA parameters
    parser.add_argument(
        '--use-lora-plugin',
        nargs='?',
        const=None,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        '--lora-target-modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        '--max-lora-rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )
    # FP8 parameters
    parser.add_argument(
        "--fp8",
        default=None,
        type=bool,
        help="Enables exporting to a FP8-quantized TRT LLM checkpoint.",
    )
    parser.add_argument(
        "--fp8-kv-cache",
        default=None,
        type=bool,
        help="Enables exporting with FP8-quantizatized KV-cache.",
    )

    args = parser.parse_args()
    return args


def nemo_export_trt_llm():
    args = get_args()
    # Check that only one of nemo_model or hf_model is provided
    assert (args.nemo_model is not None) != (args.hf_model is not None), "You must provide either --nemo-model or --hf-model, but not both"

    loglevel = logging.DEBUG if args.debug_mode else logging.INFO
    LOGGER.setLevel(loglevel)
    LOGGER.info(f"Logging level set to {loglevel}")
    LOGGER.info(pprint.pformat(vars(args)))

    trt_llm_exporter = TensorRTLLM(
        model_dir=args.output_dir, load_model=False, multi_block_mode=args.multi_block_mode
    )
    export_args = {
        "model_type": args.model_type,
        "dtype": args.dtype,
        "tensor_parallelism_size": args.tensor_model_parallel_size,
        "pipeline_parallelism_size": args.pipeline_model_parallel_size,
        "max_seq_len": args.seq_length,
        "max_input_len": args.max_input_len,
        "max_output_len": args.max_output_len,
        "max_batch_size": args.max_batch_size,
        "opt_num_tokens": args.opt_num_tokens,
        "max_prompt_embedding_table_size": args.max_prompt_embedding_table_size,
        "use_parallel_embedding": args.use_parallel_embedding,
        "use_embedding_sharing": args.use_embedding_sharing,
        "paged_kv_cache": not args.no_paged_kv_cache,
        "remove_input_padding": not args.disable_remove_input_padding,
        "paged_context_fmha": args.paged_context_fmha,
        "delete_existing_files": args.delete_existing_files,
        "multiple_profiles": args.multiple_profiles,
    }

    LOGGER.info("Export to TensorRT-LLM function is called.")
    if args.nemo_model: # Export from NeMo checkpoint
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_model,
            use_lora_plugin=args.use_lora_plugin,
            lora_target_modules=args.lora_target_modules,
            max_lora_rank=args.max_lora_rank,
            fp8_quantized=args.fp8,
            fp8_kvcache=args.fp8_kv_cache,
            use_mcore_path=args.use_mcore_path,
            **export_args,
        )
    else: # Export from HF checkpoint
        trt_llm_exporter.export_hf_model(
            hf_model_path=args.hf_model,
            **export_args,
        )

    LOGGER.info("Export is successful.")


if __name__ == '__main__':
    nemo_export_trt_llm()
