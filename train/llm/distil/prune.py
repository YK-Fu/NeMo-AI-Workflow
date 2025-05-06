import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.modelopt.recipes import prune_recipe

WORK_PATH = os.getcwd()
os.environ['NEMORUN_HOME'] = WORK_PATH

def run_pruning(args):
    recipe = prune_recipe(
        nemo_checkpoint=args.nemo_checkpoint,
        save_path=args.save_path,
    )
    executor = run.LocalExecutor(
        launcher="torchrun",
        ntasks_per_node=args.num_gpus,
        env_vars=env_vars
    )
    # Override the configuration with desired components:
    recipe.devices = args.num_gpus
    recipe.pp_size = args.pipeline_model_parallel_size
    recipe.data = run.Config(
        llm.PreTrainingDataModule,
        paths=args.data_paths,
        seq_length=args.seq_length,
        micro_batch_size=args.batch_size,
        global_batch_size=args.batch_size,
    )
    if args.target_ffn_hidden_size is not None:
        recipe.pruning_config.target_ffn_hidden_size = args.target_ffn_hidden_size

    if args.target_hidden_size is not None:
        recipe.pruning_config.target_hidden_size = args.target_hidden_size

    if args.target_num_attention_heads is not None:
        recipe.pruning_config.target_num_query_groups = args.target_num_attention_heads

    if args.target_num_query_groups is not None:
        recipe.pruning_config.target_num_query_groups = args.target_num_query_groups

    if args.target_num_layers is not None:
        recipe.pruning_config.target_num_layers = args.target_num_layers
    if len(args.drop_layers) > 0:
        recipe.pruning_config.drop_layers = args.drop_layers
    if args.legacy_ckpt:
        recipe.pruning_config.legacy_ckpt = args.legacy_ckpt

    with run.Experiment(args.experiment, base_dir=WORK_PATH) as exp:
        exp.add(recipe, executor=executor, name="pruning")
        exp.run(sequential=True, tail_logs=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="pruning", help="The name of the experiment")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("-PP", "--pipeline-model-parallel-size", type=int, default=8)
    parser.add_argument("--nemo-checkpoint", type=str, default="/path/to/llama3.1-8b/nemo-ckpt/")
    parser.add_argument("--save-path", type=str, default="/path/to/pruned/llama3.1-8b/nemo-ckpt/")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size of the model")

    # Data for pruning
    parser.add_argument("--data-paths", type=list, nargs="+", default=["1.0", "path/to/tokenized/data"], help="The weight and path to the tokenized data. e.g., 1.0 path/to/tokenized/data")
    parser.add_argument("--seq-length", type=int, default=8192, help="The sequence length of the model input")

    # Model pruning parameters
    parser.add_argument("--target-ffn-hidden-size", type=int, default=None, help="The target ffn hidden size of the model")
    parser.add_argument("--target-hidden-size", type=int, default=None, help="The target hidden size of the model")
    parser.add_argument("--target-num-attention-heads", type=int, default=None, help="The target number of attention heads of the model")
    parser.add_argument("--target-num-query-groups", type=int, default=None, help="The target number of query groups of the model. Must specify --target-num-attention-heads")
    parser.add_argument("--target-num-layers", type=int, default=None, help="The target number of layers of the model")
    parser.add_argument("--drop-layers", type=list, nargs="*", default=[], help="The layers to drop. Do not use with --target-num-layers. e.g., 1 2 3")
    parser.add_argument("--legacy-ckpt", action="store_true", help="Use legacy checkpoint for pruning. If you face Missing key(s) errors, use this option.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pruning(args)