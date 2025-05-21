# NeMo Model Inference

This directory contains deployment options for serving NeMo models in production environments. You have several choices for model inference, depending on your performance requirements and deployment preferences.

## Inference Options

### 1. Native NeMo Python Server
For simple deployment or development purposes, you can use NeMo's native Python inference server. This is the easiest way to get started but may not provide optimal performance (See [nemo-native-inference](./TRITON.md#nemo-native-inference) for deployment instructions).

### 2. Optimized Inference
For production-grade performance, we recommend using TensorRT-LLM, which provides significant speedup.

To use TensorRT-LLM, you'll need to:
1. Export your model to TensorRT-LLM (See [TRT.md](./TRT.md) for compile instructions)
2. Choose one of the following deployment options:
   - NVIDIA Triton Inference Server (See [tensorrt-llm-inference](./TRITON.md#tensorrt-llm-inference) for deployment instructions)
   - NVIDIA Inference Microservice (NIM) (See [NIM.md](./NIM.md) for deployment instructions)

## Additional Resources

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA Triton Documentation](https://github.com/triton-inference-server/server)
- [NVIDIA NIM Documentation](https://build.nvidia.com/)
