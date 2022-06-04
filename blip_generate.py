import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

sys.path.append('BLIP')
from models.blip import blip_decoder


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 384


def transform_image(pil_img, image_size, device):
    raw_image = pil_img.convert('RGB')   
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def main():
    images = {int(img.split('.')[0]): Image.open(f'val2017/{img}') for img in os.listdir('val2017') if img.endswith('.jpg')}
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, med_config = 'BLIP/configs/med_config.json', image_size=IMAGE_SIZE, vit='base')
    model.eval()
    model = model.to(DEVICE)
    transformed_images = {}
    for k, pil_img in tqdm(images.items()):
        transformed_images[k] = transform_image(pil_img, IMAGE_SIZE, DEVICE)
    results = {}
    with torch.no_grad():
        for k, img in tqdm(transformed_images.items()):
            results[k] = model.generate(img, sample=False, num_beams=16, max_length=20, min_length=5)[0]
    with open('blip_generated_captions.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
