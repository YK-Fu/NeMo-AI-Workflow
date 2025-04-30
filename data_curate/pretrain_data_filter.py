import os
import argparse
import requests
import yaml
from nemo_curator import (
    AddId, 
    Modify, 
    FuzzyDuplicates, 
    FuzzyDuplicatesConfig, 
    ScoreFilter, 
    Sequential, 
    SemDedup, 
    SemDedupConfig
)
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.filters import FastTextLangId, WordCountFilter, RepeatingTopNGramsFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier

from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_all_files_paths_under, separate_by_metadata

"""
This script is used to curate the pretrain data.

The input directory contains jsonl files.
The output directory contains the curated data.

Usage example:
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

Noting that when fuzzy-dedup is used, the device must be gpu.
"""

def pre_imports() -> None:
    import cudf  # noqa: F401

def download_lid_model(model_file):
    if not os.path.exists(model_file):
        print("Downloading language identification model...")
        response = requests.get("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")

def main(args):
    dataset_dir = args.input_dir
    output_directory = args.output_dir

    client = get_client(**ArgumentHelper.parse_client_args(args))
    input_dataset = DocumentDataset.read_json(dataset_dir, add_filename=True)
    backend = "pandas"
    # Add id to the dataset
    if "id" not in input_dataset.df.columns:
        add_id = AddId(id_field="id", id_prefix="id")
        input_dataset = add_id(input_dataset)

    # Clean the unicode characters
    if args.cleaning:
        output_cleaning_dir = f"{output_directory}/cleaning"
        expand_outdir_and_mkdir(output_cleaning_dir)
        cleaner = Modify(UnicodeReformatter(), text_field=args.text_field)
        cleaned_dataset = cleaner(input_dataset)
        cleaned_dataset.to_json(output_cleaning_dir, write_to_filename=True)
        input_dataset = cleaned_dataset

    # Language identification
    if args.lid:
        language_output_dir = f"{output_directory}/language"
        expand_outdir_and_mkdir(language_output_dir)
        model_file = os.path.join(language_output_dir, "lid.176.bin")
        download_lid_model(model_file)

        lang_filter = FastTextLangId(model_file)
        language_id_pipeline = ScoreFilter(
            lang_filter,
            score_field="language",
            text_field=args.text_field,
            score_type="object"
        )
        input_dataset = language_id_pipeline(input_dataset)
        input_dataset.df["language"] = input_dataset.df["language"].apply(
            lambda score: score[1],
            meta=("language", "object"),
        )
        language_stats = separate_by_metadata(
            input_dataset.df,
            language_output_dir,
            remove_metadata=False,
            metadata_field="language",
        ).compute()

        if args.language:
            input_dataset = DocumentDataset.read_json(f"{language_output_dir}/{args.language.upper()}", add_filename=True)
        else:
            import glob
            input_dataset = DocumentDataset.read_json(glob.glob(f"{language_output_dir}/**/*.jsonl"), add_filename=True)

    # Heuristic filtering
    if args.heuristic:
        heuristic_dir = f"{output_directory}/heuristic"
        expand_outdir_and_mkdir(heuristic_dir)
        filter_steps = Sequential([
            ScoreFilter(WordCountFilter(min_words=50), text_field=args.text_field, score_field="word_count"),
            ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2, lang=args.language if args.language else "en"), text_field=args.text_field),
            ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18, lang=args.language if args.language else "en"), text_field=args.text_field),
            ScoreFilter(RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16, lang=args.language if args.language else "en"), text_field=args.text_field),
        ])
        input_dataset = filter_steps(input_dataset)
        input_dataset.to_json(heuristic_dir, write_to_filename=True)
    
    # Remove personal identifiable information
    if args.pii:
        pii_dir = f"{output_directory}/pii"
        expand_outdir_and_mkdir(pii_dir)
        modifier = PiiModifier(
            supported_entities=args.pii_entities,
            anonymize_action=args.pii_anonymize_action,
            batch_size=1000,
            device=args.device,
        )
        modify = Modify(modifier)
        input_dataset = modify(input_dataset)
        input_dataset.to_json(pii_dir, write_to_filename=True)

    # Exact deduplication
    if args.exact:
        if args.device == "gpu":
            client.run(pre_imports)
            backend = "cudf"
            input_dataset = DocumentDataset(input_dataset.df.to_backend(backend))

        exact_dup_dir = f"{output_directory}/exact_duplicates"
        expand_outdir_and_mkdir(exact_dup_dir)
        
        exact_dup = ExactDuplicates(
            logger=exact_dup_dir,
            id_field="id",
            text_field=args.text_field,
            perform_removal=True,
            cache_dir=exact_dup_dir
        )
        input_dataset = exact_dup(input_dataset)
        input_dataset.to_json(exact_dup_dir, write_to_filename=True)

    # Fuzzy deduplication
    if args.fuzzy:
        assert args.device == "gpu", "Fuzzy dedup is currently only supported on GPU"
        if backend == "pandas":
            client.run(pre_imports)
            backend = "cudf"
            input_dataset = DocumentDataset(input_dataset.df.to_backend(backend))
        import dask
        with dask.config.set({"dataframe.backend": backend}):
            fuzzy_dedup_dir = f"{output_directory}/fuzzy_dedup"
            expand_outdir_and_mkdir(fuzzy_dedup_dir)

            # Load the fuzzy dedup config file
            with open(args.fuzzy_config, "r") as config_file:
                config_dict = yaml.safe_load(config_file)
                config_dict["cache_dir"] = f"{fuzzy_dedup_dir}/cache"

            fuzzy_dedup_config = FuzzyDuplicatesConfig(**config_dict)
            fuzzy_dedup = FuzzyDuplicates(
                logger=fuzzy_dedup_dir,
                config=fuzzy_dedup_config
            )
            result = fuzzy_dedup(input_dataset)
            # Get unique IDs by keeping first occurrence in each group
            if result is not None:
                input_dataset = fuzzy_dedup.remove(input_dataset, result)
            input_dataset.to_json(fuzzy_dedup_dir, write_to_filename=True)

    # Semantic deduplication
    if args.semantic:
        assert args.device == "gpu", "Semantic dedup is currently only supported on GPU"
        if backend == "pandas":
            client.run(pre_imports)
            backend = "cudf"
            input_dataset = DocumentDataset(input_dataset.df.to_backend(backend))
        semantic_dir = f"{output_directory}/semantic"
        expand_outdir_and_mkdir(semantic_dir)
        with open(args.semantic_config, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            config_dict["clustering_save_loc"] = f"{semantic_dir}/clustering"
            config_dict["embeddings_save_loc"] = f"{semantic_dir}/embeddings"
            config_dict["cache_dir"] = f"{semantic_dir}/cache"
        config = SemDedupConfig(**config_dict)
        semantic_dedup = SemDedup(
            logger=semantic_dir,
            input_column=args.text_field,
            id_column="id",
            config=config
        )
        result = semantic_dedup(input_dataset)
        if result is not None:
            input_dataset = semantic_dedup.remove(input_dataset, result)
        input_dataset.to_json(semantic_dir, write_to_filename=True)

    client.close()

def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    # Common arguments, if no language is specified, all languages are used
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing jsonl files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--text-field", type=str, default="text", help="Text field to use")
    parser.add_argument("--language", type=str, default="", help="Language of the data")

    # Unicode, language identification options
    parser.add_argument("--cleaning", action="store_true", default=False, help="Clean the unicode characters")
    parser.add_argument("--lid", action="store_true", default=False, help="Run language identification")

    # pii removal
    parser.add_argument("--pii", action="store_true", default=False, help="Run pii removal")
    parser.add_argument("--pii-entities", default=["PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS"], type=list, nargs="+", help="Entities to anonymize. See https://github.com/microsoft/presidio/blob/main/docs/supported_entities.md for more supported entities")
    parser.add_argument("--pii-anonymize-action", default="replace", choices=["hash", "replace", "redact", "encrypt", "mask"], type=str, help="Action to anonymize the entities")

    # heuristic filtering and its hyperparameters
    parser.add_argument("--heuristic", action="store_true", default=False, help="Run heuristic filtering: remove short stories, remove repeating ngrams")
    parser.add_argument("--heuristic-min-words", type=int, default=50, help="Minimum number of words for heuristic filtering")

    # exactfuzzy deduplication and its hyperparameters
    parser.add_argument("--exact", action="store_true", default=False, help="Run exact deduplication")
    parser.add_argument("--fuzzy", action="store_true", default=False, help="Run fuzzy deduplication")
    parser.add_argument("--fuzzy-config", type=str, default="config/fuzzy_dedup.yaml", help="Fuzzy dedup config file")

    # semantic deduplication and its hyperparameters
    parser.add_argument("--semantic", action="store_true", default=False, help="Run semantic deduplication")
    parser.add_argument("--semantic-config", type=str, default="config/semantic_dedup.yaml", help="Semantic config file")

    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
