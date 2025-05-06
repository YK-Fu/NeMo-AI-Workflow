import argparse
from nemo.collections import llm

def list_supported_model(model_type: str="gpt"):
    return [obj for obj in getattr(llm, model_type).model.__dir__() if "Model" in obj]

def list_supported_config(model_type: str="gpt"):
    return [obj for obj in getattr(llm, model_type).model.__dir__() if "Config" in obj]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="gpt", help="Model type")
    parser.add_argument("--object-type", type=str, default="model", choices=["model", "config"], help="Object type")
    parser.add_argument("--output-path", type=str, default=None, help="Output path")
    args = parser.parse_args()

    models = []
    configs = []
    
    if args.object_type == "model":
        models = list_supported_model(args.model_type)
    elif args.object_type == "config":
        configs = list_supported_config(args.model_type)
    
    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            if args.object_type == "model":
                f.write('\n'.join(models))
            elif args.object_type == "config":
                f.write('\n'.join(configs))
    else:
        if args.object_type == "model":
            print(models)
        elif args.object_type == "config":
            print(configs)