import argparse
import json
import os
from tqdm import tqdm
from openai import OpenAI
from nemo_curator import OpenAIClient

def main(args):
    openai_client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
    )
    client = OpenAIClient(openai_client)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_path, "r") as f, open(os.path.join(args.output_dir, "filtered_data.jsonl"), "w") as filtered, open(os.path.join(args.output_dir, "sft_data_score.jsonl"), "w") as scored:
        for line in tqdm(f):
            data = json.loads(line)
            score_data = data.copy()

            # get reward
            messages = [
                {"role": "user", "content": data[args.context_field]},
                {
                    "role": "assistant",
                    "content": data[args.response_field],
                },
            ]
            rewards = client.query_reward_model(messages=messages, model=args.model)
            rewards["overall"] = (rewards["helpfulness"] + rewards["correctness"] + rewards["coherence"]) / 3
            score_data["reward"] = rewards
            reward = rewards[args.filter_field]

            # write to files
            scored.write(json.dumps(score_data) + "\n")
            if reward > args.threshold:
                filtered.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="https://integrate.api.nvidia.com/v1", help="NVIDIA / OpenAI API endpoint")
    parser.add_argument("--api-key", type=str, default=os.environ["NVIDIA_API_KEY"], help="NVIDIA / OpenAI API key")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-4-340b-reward", help="Reward model to use")
    parser.add_argument("--input-path", type=str, default="sft_data.jsonl", help="Input path")
    parser.add_argument("--output-dir", type=str, default="sft_data_results", help="Output directory")
    parser.add_argument("--context-field", type=str, default="input", help="User query field")
    parser.add_argument("--response-field", type=str, default="output", help="Response field")
    parser.add_argument("--filter-field", type=str, default="overall", choices=["helpfulness", "correctness", "coherence", "complexity", "verbosity", "overall"], help="Field to filter, avg is to use the average of helpfulness, correctness, and coherence")
    parser.add_argument("--threshold", type=float, default=3.5, help="Threshold to filter")

    args = parser.parse_args()

    main(args)