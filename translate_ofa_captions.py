import os
import torch
import json
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import pipeline


def main():
    with open('ofa_generated_captions.json', 'r') as f:
        caption_file = json.load(f)
    model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    annotations = {}
    for key, value in tqdm(caption_file.items()):
        annotations[key] = {
            'en': value,
            'ru': translator(value)[0]['translation_text'],
        }
    with open('ofa_translated_captions.json', 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == '__main__':
    main()
