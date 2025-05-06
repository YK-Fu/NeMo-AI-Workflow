import os
import glob
import argparse

import pytorch_lightning as pl
from transformers import AutoImageProcessor
from lightning.pytorch.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

WORK_PATH = os.getcwd()
os.environ['NEMORUN_HOME'] = WORK_PATH

import nemo_run as run
from nemo_run.core.tunnel.client import LocalTunnel

class ImageProcessor():
    def __init__(self, pretrained_model_name):
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    def preprocess(self, image, return_tensors):
        return self.processor(image, return_tensors=return_tensors)

def find_latest_checkpoint(
    ckpt_path="experiments/mllama_finetuning/checkpoints"
) -> str:
    checkpoint_files = glob.glob(os.path.join(ckpt_path, "**", "*last*"), recursive=True)
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
    return latest_checkpoint

def configure_dataset(
    args,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    image_dir = os.path.join(WORK_PATH, args.image_dir)
    data_path = os.path.join(WORK_PATH, args.dataset_path)
    data_config = run.Config(
        vlm.ImageDataConfig,
        image_folder=image_dir,
        conv_template="mllama",  # Customize based on your dataset needs
    )
    dataset = run.Config(
        vlm.MLlamaPreloadedDataModule,
        paths=data_path,  # Path to your llava-like dataset
        data_config=data_config,
        seq_length=6406,
        decoder_seq_length=seq_length,
        global_batch_size=args.global_batch_size,  # Global batch size
        micro_batch_size=args.micro_batch_size,  # Micro batch size
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.hf_model_id),
        image_processor=run.Config(ImageProcessor, pretrained_model_name=args.hf_model_id),
        num_workers=0,  # Number of workers for data loading
    )

    return dataset

def configure_recipe(args):
    if args.model_size.lower() == "11b":
        model = vlm.mllama_11b
    elif args.model_size.lower() == "90b":
        model = vlm.mllama_90b

    recipe = model.finetune_recipe(
        dir="nemo_experiments",
        name=args.experiment,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus,
        peft_scheme=args.peft,
    )
    recipe.peft.freeze_vision_model = args.freeze_vision_model

    recipe.data = configure_dataset(args, seq_length=args.seq_length)
    recipe.trainer.devices = args.num_gpus
    
    recipe.trainer.max_steps = args.max_steps
    recipe.trainer.val_check_interval = 5 # args.max_steps // 5 if args.max_steps > 100 else recipe.trainer.max_steps
    recipe.trainer.num_sanity_val_steps = 0
    
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_model_parallel_size
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    recipe.trainer.strategy.context_parallel_size = args.context_parallel_size
    recipe.trainer.strategy.sequence_parallel=True
    # Set False, if you have an issue when loading checkpoint.
    # recipe.trainer.strategy.ckpt_load_strictness = False

    if args.fp8:
        recipe.trainer.plugins.fp8 = "hybrid"
        recipe.trainer.plugins.fp8_amax_history_len = 1024
        recipe.trainer.plugins.fp8_amax_compute_algo = "max",
        recipe.trainer.plugins.fp8_params = True

    recipe.optim.config.lr = 5e-6
    
    recipe.log.ckpt.save_optim_on_train_end = True
    recipe.log.ckpt.monitor = "val_loss"
    recipe.log.ckpt.save_top_k = 10

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
        checkpoint = find_latest_checkpoint(ckpt_path=os.path.join(WORK_PATH, "nemo_experiments/mllama_finetuning/checkpoints"))

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
    # run.run(recipe, direct=True)
    with run.Experiment(args.experiment, base_dir=WORK_PATH) as exp:
        exp.add(recipe, executor=executor, name="finetuning")
        exp.dryrun(delete_exp_dir=False) if args.executor == "slurm" else exp.run(sequential=True, tail_logs=True)
    

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Finetuning Arguments")
    
    # 實驗執行方式
    parser.add_argument("--executor", type=str, choices=["slurm", "local"], default="local",
                        help="Select execution mode: 'slurm' (Multiple Nodes) or 'local' (Single Node).")
    parser.add_argument("-E", "--experiment", type=str, default="mllama_finetuning", help="Name of experiment")

    # Slurm 參數設定
    parser.add_argument("-a", "--account", type=str, default="root", help="Slurm partition name")
    parser.add_argument("-p", "--partition", type=str, default="defq", help="Slurm partition name")
    parser.add_argument("-i", "--container_image", type=str, default="nvcr.io/nvidia/nemo:dev", help="NEMO image path")
    
    # 硬體設定
    parser.add_argument("-N", "--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("-G", "--num_gpus", type=int, default=8, help="Number of GPUs")
    
    # 模型設定
    parser.add_argument("-M", "--model_size", type=str, choices=["11B", "11b", "90B", "90b"], default="11B", 
                        help="Select MLlama model size: '11B' or '90B'")
    parser.add_argument("--hf_model_id", type=str, required=True, help="Huggingface Model ID")
    parser.add_argument("--hf_token", type=str, required=False, help="Huggingface Token for downloading tokenizer")
    parser.add_argument("-n", "--nemo_model", type=str, nargs="?", help="Pretrained NeMo Model path")
    parser.add_argument("-s", "--seq_length", type=int, default=8192, help="Sequence length for the training")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training mode")
    parser.add_argument("--peft", type=str, choices=["lora", "none"], default="lora", help="Parameter efficient fine-tuning scheme (default to None)")
    parser.add_argument("--freeze_vision_model", action="store_true", help="Whether to freeze image encoder during fine-tuning")

    # 訓練參數
    parser.add_argument("--max_steps", type=int, default=None,
                        help="The number of training steps (updates) for the model. "
                        "Each step updates the model parameters once. If not set, the default training schedule will be used.")
    parser.add_argument("-g", "--global_batch_size", type=int, default=2048, help="Global batch size (must be multiple of micro_batch_size * data parallel size)")
    parser.add_argument("-m", "--micro_batch_size", type=int, default=1, help="Micro batch size per data parallel group")

    # 模型平行化參數
    parser.add_argument("-T", "--tensor_model_parallel_size", type=int, default=1,
                        help="Tensor model parallelism size")
    parser.add_argument("-P", "--pipeline_model_parallel_size", type=int, default=1,
                        help="Pipeline model parallelism size")
    parser.add_argument("-C", "--context_parallel_size", type=int, default=1,
                        help="Context parallelism size (usually 1, unless using advanced parallelism)")

    # 資料集路徑
    parser.add_argument("-I", "--image_dir", type=str, required=True, help="image folder")
    parser.add_argument("-D", "--dataset_path", type=str, required=True,
                        help="Path to the folder containing the preprocessed dataset. "
                        "This folder should include files named in the format: "
                        "'training.jsonl', 'validation.jsonl' 'test.jsonl'.")
    
    # WandB 相關參數
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb_token", type=str, default=None, help="WandB personal token")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_finetuning(args)