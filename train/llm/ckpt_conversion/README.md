# Convert Llama Nemotron 4b/8b Models from Huggingface to NeMo
The following scripts are tested under `nvcr.io/nvidia/nemo:25.07`
You should always run this line when you open a new terminal or reboot the system.
```
export HF_TOKEN=<HF_TOKEN>
```

To convert the checkpoint, run the following commands:
```
python convert_from_hf_to_nemo2.py \
    --source nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
    --output-path ./nemo_llama_nemotron_model/ \
    --overwrite
```
You should see the following output; if not, please run it a few more times.
```
...
[NeMo I 2025-08-19 14:02:29 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 0 : Start time: 1755612135.735s : Save duration: 13.567s
[NeMo I 2025-08-19 14:02:36 nemo_logging:393] Successfully saved checkpoint from iteration       0 to nemo_llama_nemotron_model
[NeMo I 2025-08-19 14:02:36 nemo_logging:393] Async finalization time took 7.538 s
Converted Llama model to Nemo, model saved to nemo_llama_nemotron_model in torch.bfloat16.
✓ Checkpoint imported to nemo_llama_nemotron_model
Imported Checkpoint
├── context/
│   ├── artifacts/
│   │   └── generation_config.json
│   ├── nemo_tokenizer/
│   │   ├── chat_template.jinja
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── io.json
│   └── model.yaml
└── weights/
    ├── .metadata
    ├── __0_0.distcp
    ├── __0_1.distcp
    ├── common.pt
    └── metadata.json
```