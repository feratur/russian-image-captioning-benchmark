import os
import torch
import json
from PIL import Image
from base64 import b64encode

import ruclip


def main():
    device = 'cuda'
    clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
    predictor = ruclip.Predictor(clip, processor, device, bs=8, templates=[])
    images = {int(img.split('.')[0]): Image.open(f'val2017/{img}') for img in os.listdir('val2017') if img.endswith('.jpg')}
    sorted_imgs = sorted(((k, v) for k, v  in images.items()))
    sorted_keys = [k for k, v in sorted_imgs]
    sorted_values = [v for k, v in sorted_imgs]
    with torch.no_grad():
        latents = (predictor.get_image_latents(sorted_values).cpu().numpy())
    latents_map = {key: b64encode(latents[idx].tobytes()).decode('utf-8') for idx, key in enumerate(sorted_keys)}
    with open('ruclip_coco_img_latents.json', 'w') as outfile:
        json.dump(latents_map, outfile)


if __name__ == '__main__':
    main()
