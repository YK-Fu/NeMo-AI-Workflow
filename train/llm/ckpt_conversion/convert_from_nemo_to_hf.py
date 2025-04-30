"""
This script is migrate from https://github.com/wcks13589/NeMo-Tutorial, thanks for the author's great work!
"""
from pathlib import Path
from argparse import ArgumentParser

from nemo.collections.llm import export_ckpt

def export_checkpoint(args):
    """
    Imports a checkpoint from NeMo to Huggingface format.
    """
    print(f"Huggingface weight will be saved to: {args.output_path}")

    # Export the checkpoint
    try:
        export_ckpt(
            path=args.source,
            output_path=Path(args.output_path),
            target="hf",
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
        help="Path to nemo folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hf_ckpt",
        help="Output HF model path",
    )
    args = parser.parse_args()

    export_checkpoint(args)