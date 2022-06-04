import os
import torch
import json
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import pipeline


def main():
    with open('annotations/captions_val2017.json', 'r') as f:
        caption_file = json.load(f)
    model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
    translator = pipeline("translation", model=model_checkpoint, device=0)
    annotations = defaultdict(list)
    for annotation in tqdm(caption_file['annotations']):
        image_id = int(annotation['image_id'])
        annotations[image_id].append({
            'en': annotation['caption'],
            'ru': translator(annotation['caption'])[0]['translation_text'],
        })
    with open('coco_annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == '__main__':
    main()
