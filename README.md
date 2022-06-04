# A survey of models for generating image captions in Russian
A small benchmark of Deep Learning models for generating Russian captions for images.

## About the dataset
For comparing the performance of different models on Image Captioning task I used a standard MSCOCO dataset: [[Link]](https://cocodataset.org/)

I used val2017 split - 5000 images, each image has around 5 different annotation variants. All annotations were translated into Russian using **Helsinki-NLP Opus-MT en-ru** translation model ([github](https://github.com/Helsinki-NLP/Opus-MT), [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)). Then translated annotations were ranked using [RuCLIP](https://github.com/ai-forever/ru-clip) and for each image the annotation with the highest cosine similarity score was selected as the ground truth.

### Histogram of ruCLIP scores for compared models
![Alt text](artifacts/hist.png?raw=true "Comparison histogram")
