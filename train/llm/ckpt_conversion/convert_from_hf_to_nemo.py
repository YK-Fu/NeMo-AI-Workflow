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
    config = getattr(llm, args.model_config_name)
    model = getattr(llm, args.model_name)(config=config)

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
        "--output-path",
        type=str,
        default=None,
        required=True,
        help="Path to output folder.")
    parser.add_argument(
        '--model-name',
        type=str,
        default='LlamaModel',
        help="Model name"
    )
    parser.add_argument(
        '--model-config-name',
        type=str,
        default='Llama32Config1B',
        help="Model config name"
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="If set to True, existing files at the output path will be overwritten."
    )
    args = parser.parse_args()

    import_checkpoint(args)