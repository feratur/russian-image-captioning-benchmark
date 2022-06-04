#!/usr/bin/env bash
set -Eeuo pipefail

echo "Loading COCO dataset..."
curl http://images.cocodataset.org/zips/val2017.zip --output val2017.zip
unzip val2017.zip
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip --output annotations_trainval2017.zip
unzip annotations_trainval2017.zip

echo "Loading models..."
git clone https://github.com/salesforce/BLIP
git clone https://github.com/OFA-Sys/OFA.git
wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt
mkdir -p OFA/checkpoints/
mv caption_large_best_clean.pt OFA/checkpoints/caption.pt

echo "Generating ruCLIP embeddings for COCO images..."
python3 ruclip_generate_img_latents.py

echo "Translating COCO annotations into Russian..."
python3 translate_coco_captions.py

echo "Getting ruCLIP scores for translated COCO captions..."
python3 ruclip_eval_coco.py

echo "Generating COCO annotations (in Russian) using RuDOLPH ..."
python3 rudolph_generate.py

echo "Generating COCO annotations using BLIP..."
python3 blip_generate.py
echo "Translation BLIP-generated annotations from English into Russian..."
python3 translate_blip_captions.py

echo "Generating COCO annotations using OFA..."
python3 ofa_generate.py
echo "Translation BLIP-generated annotations from English into Russian..."
python3 translate_ofa_captions.py

echo "Evaluating all models using ruCLIP..."
python3 ruclip_eval_all.py

echo "Drawing result histogram..."
python3 savefig.py

echo "Copying all generated data into artifacts directory..."
mkdir -p artifacts
cp ./ruclip_coco_img_latents.json artifacts/ruclip_coco_img_latents.json
cp ./coco_annotations.json artifacts/coco_annotations.json
cp ./ruclip_coco_caption_latents.json artifacts/ruclip_coco_caption_latents.json
cp ./rudolph_captions.json artifacts/rudolph_captions.json
cp ./blip_generated_captions.json artifacts/blip_generated_captions.json
cp ./blip_translated_captions.json artifacts/blip_translated_captions.json
cp ./ofa_generated_captions.json artifacts/ofa_generated_captions.json
cp ./ofa_translated_captions.json artifacts/ofa_translated_captions.json
cp ./ruclip_final_scores.json artifacts/ruclip_final_scores.json
cp ./hist.png artifacts/hist.png

echo "All done!"
