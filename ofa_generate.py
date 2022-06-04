import sys

sys.path.append('OFA')

import torch
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
import os
import json


def main():
    tasks.register_task('caption',CaptionTask)
    overrides={"bpe_dir":"OFA/utils/BPE", "eval_cider":False, "beam":16, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths('OFA/checkpoints/caption.pt'),
        arg_overrides=overrides
    )
    for model in models:
        model.eval()
        model.cuda()
        model.prepare_for_inference_(cfg)
    generator = task.build_generator(models, cfg.generation)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s
    def construct_sample(image: Image):
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample
    results = {}
    with torch.no_grad():
        images = {int(img.split('.')[0]): construct_sample(Image.open(f'val2017/{img}')) for img in os.listdir('val2017') if img.endswith('.jpg')}
        for key, pil_img in tqdm(images.items()):
            result, scores = eval_step(task, generator, models, utils.move_to_cuda(pil_img))
            results[key] = result[0]['caption']
    with open('ofa_generated_captions.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
