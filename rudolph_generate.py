import os
import torch
import json
from PIL import Image


from rudalle import get_tokenizer, get_vae
from rudalle.utils import seed_everything
from rudalle.image_prompts import ImagePrompts

from rudolph.model import get_rudolph_model
from rudolph.pipelines import zs_clf, generate_codebooks, self_reranking_by_image, self_reranking_by_text, show, generate_captions, generate_texts
from rudolph import utils


def main():
    device = 'cuda'
    model = get_rudolph_model('350M', fp16=True, device=device)
    model.to(device)
    tokenizer = get_tokenizer()
    vae = get_vae(dwt=False).to(device)
    images = {int(img.split('.')[0]): Image.open(f'val2017/{img}') for img in os.listdir('val2017') if img.endswith('.jpg')}
    results = {}
    for key, pil_img in images.items():
        texts = generate_captions(pil_img, tokenizer, model, vae, template='на картинке ', top_k=16, captions_num=16, bs=32, top_p=0.6, temperature=0.8, seed=43, limit_eos=False)
        ppl_text, ppl_image = self_reranking_by_image(texts, pil_img, tokenizer, model, vae, bs=32, seed=42)
        results[key] = texts[ppl_image.argmin()]
    with open('rudolph_captions.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
