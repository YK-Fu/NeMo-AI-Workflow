import argparse
from nemo.deploy.nlp import NemoQueryLLM

def query_triton(args):
    nq = NemoQueryLLM(model_name=args.model_name, url=f"{args.address}:{args.port}")
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = f.read()
    else:
        prompts = args.prompts
    if args.streaming:
        for output in nq.query_llm_streaming(
            prompts=prompts,
            max_output_len=args.max_output_length,
            stop_words_list=args.stop_words_list,
            bad_words_list=args.bad_words_list,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            random_seed=args.random_seed,
        ):
            print(output)
    else:
        output = nq.query_llm(
            prompts=prompts,
            stop_words_list=args.stop_words_list,
            random_seed=args.random_seed,
            bad_words_list=args.bad_words_list,
            max_output_len=args.max_output_length,
            min_output_len=args.min_output_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print(output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="The name of the model.")
    parser.add_argument("--address", type=str, default="0.0.0.0", help="The address to run the server on.")
    parser.add_argument("--port", type=int, default=8000, help="The HTTP port to run the server on.")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming.")

    # Inference parameters
    parser.add_argument("--prompt-file", type=str, default=None, help="The path to the file containing the prompts to query the model with.")
    parser.add_argument("--prompts", type=list, nargs="+", help="The prompts to query the model with.")
    parser.add_argument("--min-output-length", type=int, default=None, help="The minimum number of tokens to generate (not used in streaming mode).")
    parser.add_argument("--max-output-length", type=int, default=None, help="The maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature to use for the model.")
    parser.add_argument("--top-p", type=float, default=0.9, help="The top-p value to use for the model.")
    parser.add_argument("--top-k", type=int, default=50, help="The top-k value to use for the model.")
    parser.add_argument("--random-seed", type=int, default=42, help="The random seed to use for the model.")
    parser.add_argument("--stop-words-list", type=list, default=None, help="The list of stop words to use for the model.")
    parser.add_argument("--bad-words-list", type=list, default=None, nargs="+", help="The list of bad words to use for the model.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None, help="The size of the n-gram to use for the model.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="The repetition penalty to use for the model (not used in streaming mode).")
    args = parser.parse_args()
    query_triton(args)
