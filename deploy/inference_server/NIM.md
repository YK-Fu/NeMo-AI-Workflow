# NVIDIA Inference Microservice (NIM)

NVIDIA Inference Microservice (NIM) offers an optimized inference pipeline with simplified deployment options. You can run NIM either locally or deploy it to NVIDIA's hosting endpoint. Note that deploying to NVIDIA's hosting endpoint requires verification for legal compliance and performance optimization.

## Running NIM Locally with Custom Model Checkpoints

If you want to use your own model checkpoint with existing NIM containers from the [NVIDIA NGC NIM Container Registry](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim%7CNVIDIA+NIM%7Cnimmcro_nvidia_nim&orderBy=weightPopularDESC&query=&page=&pageSize=), follow these steps:

1. First, ensure your model is converted to TensorRT (TRT) format. (See [TRT.md](./TRT.md) for instructions)

2. Run the NIM container with your model:
```bash
docker run --gpus all -d --rm --shm-size=4g -u 0 \
    --name nim_server \
    -v <TRT_ENGINE_PATH>:/opt/checkpoints \
    -v <WORKSPACE_DIR>:/workspace \
    -e NGC_API_KEY=$NGC_API_KEY \
    -e NIM_SERVED_MODEL_NAME=/opt/checkpoints \
    -p <PORT>:8000 \
    -w /workspace \
    <NIM_NGC_IMAGES> \
    bash -c "/opt/nim/start_server.sh > server.log 2>&1"
```

   Replace `<PORT>` with your desired port number. The `<WORKSPACE_DIR>` will contain a `server.log` file where you can monitor the service status and check for any errors.

3. Verify the service is running:
   - Check the `server.log` file in your workspace directory
   - You should see the message: `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`
   - Once you see this message, the service is ready to accept requests

4. Test the service with a sample API request:
```bash
curl -X 'POST' \
  'http://0.0.0.0:<PORT>/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/opt/checkpoints",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      }
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": false,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }'
```
or you can also post requests via OpenAI API:
```python
from openai import OpenAI

client = OpenAI(
  base_url = "http://0.0.0.0:<PORT>/v1/",
)

completion = client.chat.completions.create(
  model="/opt/checkpoints",
  messages=[{
        "role":"user",
        "content":"Hello! How are you?"
      }],
  temperature=0.6,
  top_p=1,
  max_tokens=15,
  frequency_penalty=0,
  presence_penalty=0,
  stream=False
)

print(completion.choices[0].message)
```
5. To stop the service when needed:
```bash
docker stop nim_server
# rm server.log
```