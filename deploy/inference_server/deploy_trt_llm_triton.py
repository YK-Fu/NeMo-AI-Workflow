import argparse
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.deploy import DeployPyTriton

def launch_service(args):
    trt_llm = TensorRTLLM(model_dir=args.trt_model)
    server = DeployPyTriton(
        model=trt_llm,
        triton_model_name=args.model_name,
        address=args.address,
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        allow_http=args.allow_http,
        allow_grpc=args.allow_grpc,
        streaming=args.streaming
    )
    server.deploy()
    server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-model", type=str, required=True, help="The path to the TensorRT model")
    parser.add_argument("--model-name", type=str, required=True, help="The name of the model")
    parser.add_argument("--address", type=str, default="0.0.0.0", help="The address to run the server on")
    parser.add_argument("--http-port", type=int, default=8000, help="The HTTP port to run the server on")
    parser.add_argument("--grpc-port", type=int, default=8001, help="The gRPC port to run the server on")
    parser.add_argument("--allow-http", type=bool, default=True, help="Allow HTTP requests")
    parser.add_argument("--allow-grpc", type=bool, default=True, help="Allow gRPC requests")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming")
    args = parser.parse_args()
    launch_service(args)