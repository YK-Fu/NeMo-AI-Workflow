import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.modelopt.recipes import distillation_recipe

# Override the configuration with desired components:
recipe.data = run.Config(llm.PreTrainingDataModule, ...)
recipe.trainer.strategy.tensor_model_parallel_size = 8


"""
This script is migrate from https://github.com/wcks13589/NeMo-Tutorial, thanks for the author's great work!
"""
import os
import glob
import argparse

import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

WORK_PATH = os.getcwd()
os.environ['NEMORUN_HOME'] = WORK_PATH

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

import nemo_run as run
from nemo_run.core.tunnel.client import LocalTunnel

# Prompt template for Llama3
SYSTEM_PROMPT = (
    "You are a knowledgeable assistant trained to provide accurate and helpful information. "
    "Please respond to the user's queries promptly and politely."
)

PROMPT_TEMPLATE = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{{input}}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{output}}\
"""

def find_latest_checkpoint(ckpt_path: str) -> str:
    checkpoint_files = glob.glob(os.path.join(ckpt_path, "**", "*last*"), recursive=True)
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    return latest_checkpoint

def configure_dataset(
    args,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    data_path = os.path.join(WORK_PATH, args.dataset_path)
    dataset = run.Config(
        llm.FineTuningDataModule,
        dataset_root=data_path,
        seq_length=seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.hf_model_id),
        dataset_kwargs={"prompt_template": PROMPT_TEMPLATE}
    )

    return dataset

def configure_recipe(args):
    try:
        model = getattr(llm, args.model_name)
    except AttributeError:
        raise ValueError(f"Model type {args.model_name} is not supported")
        
    recipe = distillation_recipe(
        student_model_path=args.student_nemo_model,
        teacher_model_path=args.teacher_nemo_model,
        dir="nemo_experiments",  # Path to store logs and checkpoints
        name=args.experiment,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus,
    )


    # PEFT parameters setting
    if args.peft is not None:
        recipe.peft.target_modules = args.peft_target_modules
        recipe.peft.dim = args.peft_dim
        recipe.peft.alpha = args.peft_alpha

    # Activation checkpointing parameters setting
    if args.recompute_granularity is not None:
        recipe.model.config.recompute_granularity = args.recompute_granularity
    if args.recompute_granularity == "full":
        assert args.recompute_method is not None, "recompute_method must be specified when recompute_granularity is full"
        assert args.recompute_num_layers is not None, "recompute_num_layers must be specified when recompute_method is used"
        recipe.model.config.recompute_method = args.recompute_method
        recipe.model.config.recompute_num_layers = args.recompute_num_layers

    # Model parameters and activations CPU offloading setting
    if args.cpu_offloading:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_num_layers = args.cpu_offloading_layers
        recipe.model.config.cpu_offloading_activations = args.cpu_offloading_activations
        recipe.model.config.cpu_offloading_weights = args.cpu_offloading_weights

    recipe.data = configure_dataset(args, seq_length=recipe.data.seq_length)

    # Training configuration
    recipe.trainer.devices = args.num_gpus
    recipe.trainer.max_steps = args.max_steps
    recipe.trainer.val_check_interval = args.max_steps // 5 if args.max_steps > 100 else recipe.trainer.max_steps
    recipe.trainer.num_sanity_val_steps = 0

    # Model Parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_model_parallel_size
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    recipe.trainer.strategy.context_parallel_size = args.context_parallel_size
    recipe.trainer.strategy.sequence_parallel = args.sequence_parallel

    # FP8 training
    if args.fp8:
        recipe.trainer.plugins.fp8 = "hybrid"
        recipe.trainer.plugins.fp8_recipe = "delayed"
        recipe.trainer.plugins.fp8_margin = 0
        recipe.trainer.plugins.fp8_amax_history_len = 1024
        recipe.trainer.plugins.fp8_amax_compute_algo = "max"
        recipe.trainer.plugins.fp8_params = True

    # Setup optimizer and scheduler configuration
    recipe.optim.config.lr = args.lr
    recipe.optim.config.optimizer = args.optimizer
    recipe.optim.config.weight_decay = args.weight_decay
    recipe.optim.config.adam_beta1 = args.adam_beta1
    recipe.optim.config.adam_beta2 = args.adam_beta2
    recipe.optim.config.clip_grad = args.clip_grad
    recipe.optim.config.use_distributed_optimizer = not args.disable_dist_optim
    recipe.optim.lr_scheduler.warmup_steps = args.warmup_steps
    recipe.optim.lr_scheduler.constant_steps = args.constant_steps
    recipe.optim.lr_scheduler.min_lr = args.min_lr
    recipe.optim.lr_scheduler.max_steps = args.max_steps

    # Optimizer CPU offloading
    if args.optim_cpu_offloading:
        recipe.optim.config.optimizer_cpu_offload = True
        recipe.optim.config.optimizer_offload_fraction = args.optim_cpu_offloading_frac

    # Checkpoint configuration
    recipe.log.ckpt.save_optim_on_train_end = True
    recipe.log.ckpt.monitor = "val_loss"
    recipe.log.ckpt.save_top_k = 10

    # WandB logging configuration
    if args.wandb:
        recipe.log.wandb = run.Config(
            WandbLogger,
            project=args.wandb_project or args.experiment,
            name=args.wandb_name or recipe.log.name,
            config={},
        )
    
    return recipe

def configure_executor(args):
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "HF_TOKEN": args.hf_token,
    }

    if args.wandb:
        if args.wandb_token:
            env_vars["WANDB_API_KEY"] = args.wandb_token
        else:
            print("⚠️ WandB is enabled, but WANDB_API_KEY is missing! Please provide a valid wandb_token.")
    
    if args.executor == "slurm":
        # Custom mounts are defined here.
        container_mounts = [f"{WORK_PATH}:{WORK_PATH}"]
        srun_args = ["--container-writable"]

        tunnel = LocalTunnel(job_dir=os.path.join(WORK_PATH, "experiments"))

        # This defines the slurm executor.
        executor = run.SlurmExecutor(
            packager=run.Packager(),
            env_vars=env_vars,
            account=args.account,
            partition=args.partition,
            time="30-00:00:00",
            nodes=args.num_nodes,
            ntasks_per_node=args.num_gpus,
            gpus_per_node=args.num_gpus,
            mem="0",
            gres="gpu:8",
            exclusive=True,
            container_image=args.container_image,
            container_mounts=container_mounts,
            srun_args=srun_args,
            tunnel=tunnel,
        )
    else:
        executor = run.LocalExecutor(
            launcher="torchrun",
            ntasks_per_node=args.num_gpus, 
            env_vars=env_vars
        )

    return executor

def run_finetuning(args):
    recipe = configure_recipe(args)
    executor = configure_executor(args)

    if args.nemo_model:
        checkpoint = args.nemo_model
    else:
        checkpoint = find_latest_checkpoint(ckpt_path=os.path.join(WORK_PATH, f"nemo_experiments/{args.model_type}_sft/checkpoints"))
    
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=run.Config(
            nl.RestoreConfig,
            path=checkpoint,
            load_model_state=True,
            load_optim_state=False
        ),
        resume_if_exists=True
    )

    with run.Experiment(args.experiment, base_dir=WORK_PATH) as exp:
        exp.add(recipe, executor=executor, name="finetuning")
        exp.dryrun(delete_exp_dir=False) if args.executor == "slurm" else exp.run(sequential=True, tail_logs=True)

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Finetuning Arguments")
    
    # Execution mode
    parser.add_argument("--executor", type=str, choices=["slurm", "local"], default="local",
                        help="Select execution mode: 'slurm' (Multiple Nodes) or 'local' (Single Node).")
    parser.add_argument("-E", "--experiment", type=str, default="llama31_sft", help="Name of experiment")
    
    # Slurm parameters
    parser.add_argument("-A", "--account", type=str, default="root", help="Slurm partition name")
    parser.add_argument("-P", "--partition", type=str, default="defq", help="Slurm partition name")
    parser.add_argument("-I", "--container-image", type=str, default="nvcr.io/nvidia/nemo:dev", help="NEMO image path")
    
    # Hardware configuration
    parser.add_argument("-N", "--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("-G", "--num-gpus", type=int, default=8, help="Number of GPUs")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="llama32_1b", help="Select model type")
    parser.add_argument("--hf-model-id", type=str, required=True, help="Huggingface Model ID")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Huggingface Token for downloading tokenizer")
    parser.add_argument("--student-nemo-model", type=str, required=True, help="Student NeMo Model path")
    parser.add_argument("--teacher-nemo-model", type=str, required=True, help="Teacher NeMo Model path")
    parser.add_argument("--seq-length", type=int, default=8192, help="Sequence length for the training")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training mode")

    # training hyperparameters
    parser.add_argument("--max-steps", type=int, default=None,
                        help="The number of training steps (updates) for the model. "
                        "Each step updates the model parameters once. If not set, the default training schedule will be used.")
    parser.add_argument("-gbs", "--global-batch-size", type=int, default=2048, help="Global batch size (must be multiple of micro_batch_size * data parallel size)")
    parser.add_argument("-mbs", "--micro-batch-size", type=int, default=1, help="Micro batch size per data parallel group")

    # Optimizer and scheduler settings
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--constant-steps", type=int, default=0, help="Number of constant steps")
    parser.add_argument("--disable-dist-optim", action="store_true", help="Disable distributed optimizer")

    # Model Parallelism Parameters
    parser.add_argument("-TP", "--tensor-model-parallel-size", type=int, default=1,
                        help="Tensor model parallelism size")
    parser.add_argument("-PP", "--pipeline-model-parallel-size", type=int, default=1,
                        help="Pipeline model parallelism size")
    parser.add_argument("-VPP", "--virtual-pipeline-model-parallel-size", type=int, default=None,
                        help="Virtual pipeline model parallelism size")
    parser.add_argument("-CP", "--context-parallel-size", type=int, default=1,
                        help="Context parallelism size (usually 1, unless using advanced parallelism)")
    parser.add_argument("-SP", "--sequence-parallel", action="store_true",
                        help="Enable sequence parallelism")

    # Activation checkpointing
    parser.add_argument("--recompute-granularity", type=str, default=None, choices=["full", "selective"], 
                        help="Enable activation checkpointing. (For FlashAttention, self-attention recomputation is always enabled.)"
                        "full: full model checkpointing."
                        "selective: only MHA is recomputed, but which will disable FlashAttention. (generally does not save more memory than default setting)")
    parser.add_argument("--recompute-method", type=str, default=None, choices=["block", "uniform"], 
                        help="uniform: uniform checkpointing."
                        "block: block checkpointing, and specify the number of layers to recompute.")
    parser.add_argument("--recompute-num-layers", type=int, default=None, 
                        help="For block recompute, specify the number of layers to recompute, "
                        "When training with the pipeline parallelism, recompute-layers indicates the layers per pipeline stage. "
                        "When using virtual pipelining, recompute_num_layers specifies the number of layers per virtual pipeline stage.")

    # PEFT parameters
    parser.add_argument("--peft", type=str, default=None, choices=["lora", "dora"], help="Enable PEFT training mode")
    parser.add_argument("--peft-target-modules", type=list, default=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], nargs="+",
                        help="Target modules to apply PEFT to")
    parser.add_argument("--peft-alpha", type=float, default=32, help="PEFT alpha")
    parser.add_argument("--peft-dim", type=int, default=16, help="PEFT rank")

    # CPU offloading. 
    # Optimizer offloading reduce a lot of memory, but heavily affect the training speed.
    # Model parameters and activations offloading can not be used with activation checkpointing and fp8.
    parser.add_argument("--cpu-offloading", action="store_true", help="Enable CPU offloading for model parameters or activations.")
    parser.add_argument("--cpu-offloading-layers", type=int, default=None, help="Number of layers to offload to CPU. From 0 to total number of layers in the model minus one.")
    parser.add_argument("--cpu-offloading-activations", action="store_true", help="Activations to offload to CPU, which can not be used with activation checkpointing.")
    parser.add_argument("--cpu-offloading-weights", action="store_true", help="Parameters to offload to CPU.")
    parser.add_argument("--optim-cpu-offloading", action="store_true", help="Enable CPU offloading for optimizer states.")
    parser.add_argument("--optim-cpu-offloading-frac", type=float, default=1.0, help="Fraction of optimizer states to offload to CPU.")

    # Dataset settings
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the folder containing the preprocessed dataset. "
                        "This folder should include files named in the format: "
                        "'training.jsonl', 'validation.jsonl' 'test.jsonl'.")
    parser.add_argument("--prompt-template", type=str, default=PROMPT_TEMPLATE, help="Prompt template")

    # WandB logging parameters
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-token", type=str, default=None, help="WandB personal token")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_finetuning(args)