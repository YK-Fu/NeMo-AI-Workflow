# Triton Inference Server
For deploying the trained models, we can use Triton Inference Server to open this service. You can use in-framework python implementation to launch the model, or use TensorRT-LLM
## NeMo Backend Inference Server
To deploy the the python model in NeMo, we can launch a Triton inference server, and use a HTTP protocol to query the server.

For launch a Triton Inference Server:
```bash
python /opt/NeMo/scripts/deploy/nlp/deploy_inframework_triton.py \
 --nemo_checkpoint <NEMO_MODEL_PATH> \
 --triton_model_name <USER_DEFINED_NAME> \
 --triton_port <PORT> \                         # default to 8000
 --triton_http_address <HTTP_ADDRESS> \         # default to 0.0.0.0
 --num_gpus 8 \
 --tensor_parallelism_size <TP> \
 --pipeline_parallelism_size <PP> \
 --context_parallelism_size <CP> \
 --max_batch _size <BATCH_SIZE> \
 # --enable_flash_decode \                      # Enable if needed
 # --debug_mode \
 # --legacy_ckpt                                # If you face ckpt issue, enable this flag
```
You can also deploy Huggingface models in Triton inference server, see `/opt/NeMo/scripts/deploy/nlp/deploy_inframework_hf_triton.py` for more information.

After you successfully launch the server, you can query the model by the following command:
```bash
python /opt/NeMo/scripts/deploy/nlp/query_inframework.py \
 --model_name <USER_DEFINED_NAME> \
 --url <HTTP_ADDRESS> \
 --prompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>What is the color of a banana?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
 # --prompt_file <PATH_TO_PROMPT_FILE> \                # You can also pass a txt file containing several lines of prompts
 --max_output_len 128 \
 --top_k 20 \
 --top_p 1.0 \
 --temperature 1.0 \
 # --compute_logprob                                    # Retrun log probability
```
Output:
```
{'id': 'cmpl-1747809964', 'object': 'text_completion', 'created': 1747809964, 'model': 'llama1b', 'choices': [{'text': array([['\n\nThe color of a banana is yellow.']], dtype='<U34')}]}
```


## TensorRT-LLM Backend Inference Server
To deploy the compiled TensorRT-LLM engine model, we can launch a Triton inference server, and use a HTTP protocol to query the server.

For launch a Triton Inference Server:
```
python /opt/NeMo/scripts/deploy/nlp/deploy_triton.py \
 --triton_model_repository <TRT_ENGINE_DIR> \
 --triton_model_name <USER_DEFINED_NAME> \
 --triton_port <PORT> \                         # default to 8000
 --triton_http_address <HTTP_ADDRESS> \         # default to 0.0.0.0
 --num_gpus 8 \
 --tensor_parallelism_size <TP> \
 --pipeline_parallelism_size <PP> \
 --max_batch_size <BATCH_SIZE> \
 # --debug_mode \
```

After you successfully launch the server, you can query the model by the following command:
```bash
python /opt/NeMo/scripts/deploy/nlp/query.py \
 --model_name <USER_DEFINED_NAME> \
 --url <HTTP_ADDRESS> \
 --prompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>What is the color of a banana?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
 # --prompt_file <PATH_TO_PROMPT_FILE> \                # You can also pass a txt file containing several lines of prompts
 --max_output_len 128 \
 --top_k 20 \
 --top_p 1.0 \
 --temperature 1.0 \
```
Output:
```
The color of a banana is yellow.
```