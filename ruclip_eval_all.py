import os
import torch
import json
import numpy as np
from tqdm.auto import tqdm
from base64 import b64decode

import ruclip


def main():
    device = 'cuda'
    clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
    predictor = ruclip.Predictor(clip, processor, device, bs=8, templates=['{}'])
    with open('blip_translated_captions.json', 'r') as f:
        blip_captions = {int(k): ('на картинке ' + v['ru']) for k, v in json.load(f).items()}
    with open('ofa_translated_captions.json', 'r') as f:
        ofa_captions = {int(k): ('на картинке ' + v['ru']) for k, v in json.load(f).items()}
    with open('rudolph_captions.json', 'r') as f:
        rudolph_captions = {int(k): v for k, v in json.load(f).items()}
    with open('ruclip_coco_img_latents.json', 'r') as f:
        ruclip_coco_img_latents = json.load(f)
    with open('ruclip_coco_caption_latents.json', 'r') as f:
        coco_scores = {int(k): max(x['score'] for x in v) for k, v in json.load(f).items()}
    image_latents = {int(k): np.frombuffer(b64decode(v.encode('utf-8')), dtype='float32') for k, v in ruclip_coco_img_latents.items()}
    results = {}
    with torch.no_grad():
        for key, img_latent in tqdm(image_latents.items()):
            latents = predictor.get_text_latents([blip_captions[key][:200], ofa_captions[key][:200], rudolph_captions[key][:200]]).cpu().numpy()
            results[key] = {
                'blip_score': float(img_latent @ latents[0].T),
                'ofa_score': float(img_latent @ latents[1].T),
                'rudolph_score': float(img_latent @ latents[2].T),
                'coco_translated_score': coco_scores[key],
            }
    with open('ruclip_final_scores.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
