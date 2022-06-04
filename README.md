# A survey of models for generating image captions in Russian
A small benchmark of Deep Learning models for generating Russian captions for images.

## Comparison method
The most widely used metrics for evaluating the performance of models on Image Captioning task are BLEU, METEOR, ROUGE, CIDEr and SPICE. All of them (excluding SPICE) mainly measure the overlap of words between generated and reference captions. SPICE is slightly different - it requires parsing the sentences as graphs and calculating F-score over tuples in the candidate and reference scene graphs.

The results produced by evaluation using the metrics above are usually either random and unstable or require a large amount of generated captions for building confident statistics - but the price is considerable computational resources.

My proposed method is to use [RuCLIP](https://github.com/ai-forever/ru-clip) model (**ruclip-vit-base-patch32-384**, the best available [CLIP](https://arxiv.org/abs/2103.00020)-like model in Russian) for evaluating semantic similarity score between images and their captions in Russian. Even though this method introduces additional bias and relies heavily on the quality of the said model, it allows for less randomized measurements and more reliable comparison as the generated scores are well-distributed and tend to correlate more with human judgement.

## About the dataset
There is no existing Russian open-source image-caption pair dataset for comparing the performance of different models on Image Captioning task, so I used a standard MSCOCO dataset ([[Link]](https://cocodataset.org/)) and translated original captions into Russian.

I used val2017 split - 5000 images, each image has around 5 different annotation variants. All annotations were translated into Russian using **Helsinki-NLP Opus-MT en-ru** translation model ([github](https://github.com/Helsinki-NLP/Opus-MT), [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)). Then translated annotations were ranked using [RuCLIP](https://github.com/ai-forever/ru-clip) and for each image the annotation with the highest cosine similarity score was selected as the ground truth.

### Histogram of ruCLIP scores for compared models
![Alt text](artifacts/hist.png?raw=true "Comparison histogram")
