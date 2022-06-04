# A survey of models for generating image captions in Russian
A small benchmark of State-of-the-Art Deep Learning models for generating Russian annotations for images.

## Comparison method
The most widely used metrics for evaluating the performance of models on Image Captioning task are BLEU, METEOR, ROUGE, CIDEr and SPICE. All of them (excluding SPICE) mainly measure the overlap of words between generated and reference captions. SPICE is slightly different - it requires parsing the sentences as graphs and calculating F-score over tuples in the candidate and reference scene graphs.

The results produced by evaluation using the metrics above are usually either random and unstable or require a large amount of generated captions for building confident statistics - but the price is considerable computational resources.

My proposed method is to use [RuCLIP](https://github.com/ai-forever/ru-clip) model (`ruclip-vit-base-patch32-384`, the best available [CLIP](https://arxiv.org/abs/2103.00020)-like model in Russian) for evaluating semantic similarity between images and their captions in Russian. Even though this method introduces additional bias and relies heavily on the quality of the said model, it allows for less randomized measurements and more reliable comparison as the generated scores are well-distributed and tend to correlate more with human judgement.

## Compared models
I will be using fresh (year 2022) State-of-the-Art open-sourced models that show the best performance on [COCO Captions](https://github.com/tylin/coco-caption) and [nocaps](https://nocaps.org/) datasets. According to [paperswithcode](https://paperswithcode.com/), those models are [OFA](https://github.com/OFA-Sys/OFA) and [BLIP](https://github.com/salesforce/BLIP) - as shown on the leaderboards [[here]](https://paperswithcode.com/sota/image-captioning-on-coco-captions) and [[here]](https://paperswithcode.com/sota/image-captioning-on-nocaps-val-out-domain). Those models were trained on English corpus, so I will be translating their output into Russian using  **Helsinki-NLP Opus-MT en-ru** translation model ([github](https://github.com/Helsinki-NLP/Opus-MT), [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)) - one of the best models for Enlish to Russian translation.

I will compare the quality of captions, generated by [OFA](https://github.com/OFA-Sys/OFA) and [BLIP](https://github.com/salesforce/BLIP) and translated into Russian, with image captions, generated by [RuDOLPH](https://github.com/ai-forever/ru-dolph) Hyper-Modal Transformer model, originally trained on Russian corpus.

So the full list of the utilized models is the following:
- **OFA** ([github](https://github.com/OFA-Sys/OFA), [arXiv](https://arxiv.org/pdf/2202.03052.pdf)): checkpoint `caption_large_best_clean`
- **BLIP** ([github](https://github.com/salesforce/BLIP), [arXiv](https://arxiv.org/pdf/2201.12086.pdf): checkpoint `model_base_capfilt_large`
- **RuDOLPH** ([github](https://github.com/ai-forever/ru-dolph)): checkpoint `350M`
- For scoring: **RuCLIP** ([github](https://github.com/ai-forever/ru-clip)): checkpoint `ruclip-vit-base-patch32-384`
- For translation: **Helsinki-NLP Opus-MT en-ru** ([github](https://github.com/Helsinki-NLP/Opus-MT), [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru))

For each model I will be using text generation method, proposed by the authors of the corresponding model. For a more universal approach I will set `num_beams=16` in every model.

## About the dataset
There is no existing Russian open-source image-caption pair dataset for comparing the performance of different models on Image-to-Text task, so I used a standard MSCOCO dataset ([[Link]](https://cocodataset.org/)) and translated original captions into Russian.

I used val2017 split - 5000 images, each image has around 5 different annotation variants. All annotations were translated into Russian using **Helsinki-NLP Opus-MT en-ru** translation model ([github](https://github.com/Helsinki-NLP/Opus-MT), [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)). Then translated annotations were ranked using [RuCLIP](https://github.com/ai-forever/ru-clip) and for each image the annotation with the highest cosine similarity score was selected as the ground truth.

## How to reproduce
All artifacts (RuCLIP embeddings, generated and translated annotations, scores), generated during the benchmark, can be found in `artifacts` directory. The file `artifacts/ruclip_final_scores.json` contains the final cosine similarity scores between COCO images and Russian annotations, generated by compared models (and original COCO captions, translated into Russian) - as evaluated by [RuCLIP](https://github.com/ai-forever/ru-clip).

To reproduce the results a Docker-enabled machine with NVIDIA GPU (at least 16 GB of VRAM and CUDA 11.3 [compatible](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) drivers are required) can be used:
```
git clone git@github.com:feratur/russian-image-captioning-benchmark.git
cd russian-image-captioning-benchmark
docker run --rm -it --gpus all -v $(pwd)/artifacts:/workspace/artifacts $(docker build -q .)
```
The process includes downloading the dataset, downloading models and their checkpoints, generating and translating (when applicable) image captions and evaluating cosine similarity scores between COCO image embeddings and annotation embeddings using [RuCLIP](https://github.com/ai-forever/ru-clip). The whole process may take up to a day on a machine with NVIDIA T4 GPU.

After the evaluation process is finished the files in `artifacts` directory will be overwritten with newly generated ones.

## Conclusions and future work
The each model the [RuCLIP](https://github.com/ai-forever/ru-clip) similarity scores of 5000 COCO image-caption pairs were normally distributed with the following parameters:

Model | mean | stdev
--- | --- | ---
RuDOLPH ru | 0.2300 | 0.0902
BLIP en-ru | 0.3326 | 0.0732
OFA en-ru | **0.3696** | 0.0597
Ground truth: COCO en-ru | 0.4133 | 0.0596

It is quite evident that, according to my evaluation method, [OFA](https://github.com/OFA-Sys/OFA)-generated captions turned out to be the best, very close to the score of human-written and translated annotations. [BLIP](https://github.com/salesforce/BLIP) model was only slightly behind the first place and [RuDOLPH](https://github.com/ai-forever/ru-dolph), surprisingly, demonstrated the worst performance - even though it generates captions already in Russian so no possible translation error, that may harm the annotation quality, is introduced.

### Histogram of RuCLIP similirity scores for models on COCO Captions (val2017) dataset
![Alt text](artifacts/hist.png?raw=true "Comparison histogram")

### Next steps
- Use completely the same text generation parameters across all models, generate multiple annotaion variants at once and evaluate the models using standard metrics (BLEU, METEOR, ROUGE, CIDEr, SPICE);
- Add more models to comparison;
- Evaluate the models on different datasets ([nocaps](https://nocaps.org/), [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)), translated into Russian (or make/find original Russian image-text pair dataset);
- Train/finetune popular architectures ([OFA](https://github.com/OFA-Sys/OFA), [BLIP](https://github.com/salesforce/BLIP)) on Russian corpus for generating Russian annotations without intermediate translation;
- Implement [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/pdf/2111.09734v1.pdf) paper, using recently released [CLIP](https://github.com/openai/CLIP) `ViT-L/14@336px` weights and [ruGPT3](https://github.com/ai-forever/ru-gpts) or [mGPT](https://github.com/ai-forever/mgpt) as the Decoder;
- ...
