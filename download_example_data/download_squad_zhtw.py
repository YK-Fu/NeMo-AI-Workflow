import os
import json
import random
from datasets import load_dataset

for split in ['train', 'validation', 'test']:
    dataset = load_dataset('erhwenkuo/squad-cmrc2018-zhtw')[split]
    output_path = f'squad_zhtw/{split}.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for context, question, answer in zip(dataset['context'], dataset['question'], dataset['answer']):
            f.write(json.dumps({'input': "{}\n\n{}".format(context.strip(), question.strip()), 'output': answer['text'].strip()}, ensure_ascii=False)+ '\n')