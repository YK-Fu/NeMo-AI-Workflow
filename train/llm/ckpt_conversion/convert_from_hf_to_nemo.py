"""
This script is migrate from https://github.com/wcks13589/NeMo-Tutorial, thanks for the author's great work!
"""
from pathlib import Path
from argparse import ArgumentParser

from nemo.collections import llm

def import_checkpoint(args):
    """
    Imports a checkpoint from Hugging Face to NeMo format.
    """
    # Step 1: Initialize configuration and model
    if args.model_type == 'nemotron3':
        cfg = llm.NemotronConfig()
        model = llm.NemotronModel(config=cfg)
    elif args.model_type == 'llama3':
        cfg = llm.Llama3Config()
        model = llm.LlamaModel(config=cfg)
    elif args.model_type == 'llama31':
        cfg = llm.Llama31Config()
        model = llm.LlamaModel(config=cfg)
    elif args.model_type == 'deepseek':
        cfg = llm.DeepSeekConfig()
        model = llm.DeepSeekModel(config=cfg)
    elif args.model_type == 'gemma':
        cfg = llm.GemmaConfig()
        model = llm.GemmaModel(config=cfg)
    elif args.model_type == 'gemma2':
        cfg = llm.Gemma2Config()
        model = llm.Gemma2Model(config=cfg)
    elif args.model_type == 'qwen2':
        cfg = llm.Qwen2Config()
        model = llm.Qwen2Model(config=cfg)
    elif args.model_type == 'phi3':
        cfg = llm.Phi3Config()
        model = llm.Phi3Model(config=cfg)
    elif args.model_type == 'mistral':
        cfg = llm.MistralConfig7B()
        model = llm.MistralModel(config=cfg)
    elif args.model_type == 'mixtral':
        cfg = llm.MixtralConfig()
        model = llm.MixtralModel(config=cfg)

    # Step 2: Log the process
    print(f"Initializing model with HF model ID: {args.source}")
    print(f"Output will be saved to: {args.output_path}")

    # Step 3: Import the checkpoint
    try:
        llm.import_ckpt(
            model=model,
            source=f"hf://{args.source}",
            output_path=Path(args.output_path),
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Error during checkpoint conversion: {e}")
        raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface checkpoints",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to output folder.")
    parser.add_argument(
        '--model_type',
        type=str,
        default='llama31',
        choices=[
            'nemotron3', 'llama31', 'llama3', 'deepseek', 'gemma', 'gemma2', 'qwen2', 'phi3', 'mistral', 'mixtral'
        ],
        help=
        """
            Model type to use for conversion. 
                Some model share the same config: 
                llama32 -> llama31
                qwen2.5 -> qwen2
                deepseekv2, deepseekv3 -> deepseek
                phi3 mini -> phi3
        """
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="If set to True, existing files at the output path will be overwritten.")
    args = parser.parse_args()

    import_checkpoint(args)