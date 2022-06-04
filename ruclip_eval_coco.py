import os
import torch
import json
import numpy as np
from tqdm.auto import tqdm
from base64 import b64encode, b64decode

import ruclip


def main():
    device = 'cuda'
    clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
    predictor = ruclip.Predictor(clip, processor, device, bs=8)
    with open('coco_annotations.json', 'r') as f:
        annotations = json.load(f)
    with open('ruclip_coco_img_latents.json', 'r') as f:
        ruclip_coco_img_latents = json.load(f)
    image_latents = {int(k): np.frombuffer(b64decode(v.encode('utf-8')), dtype='float32') for k, v in ruclip_coco_img_latents.items()}
    with torch.no_grad():
        for key, batch in tqdm(annotations.items()):
            latents = predictor.get_text_latents([x['ru'] for x in batch]).cpu().numpy()
            image_latent = image_latents[key]
            for idx in range(len(batch)):
                batch[idx]['latent'] = b64encode(latents[idx].tobytes()).decode('utf-8')
                batch[idx]['score'] = float(image_latent @ latents[idx].T)
    with open('ruclip_coco_caption_latents.json', 'w') as outfile:
        json.dump(latents_map, outfile)


if __name__ == '__main__':
    main()
