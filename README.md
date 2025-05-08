# Image Captioning with BLIP

A PyTorch-based implementation to fine-tune Salesforce's BLIP model on the Flickr30k dataset and benchmark it against a CNN+LSTM baseline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Usage](#usage)
6. [Training Details](#training-details)
7. [Evaluation](#evaluation)
8. [Saving the Model](#saving-the-model)
9. [Results](#results)
12. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository provides a complete pipeline to:

* Load and preprocess the Flickr30k dataset (images + 5 reference captions per image).
* Fine-tune the `Salesforce/blip-image-captioning-base` model for 3 epochs on all image-caption pairs.
* Train a CNN+LSTM baseline for 50 epochs.
* Generate and evaluate captions on a held-out 10% sample using BLEU-1, BLEU-2, and METEOR.

---

## Baseline Models

Before using BLIP, this project also implements a classic **CNN + LSTM** encoder–decoder pipeline as a baseline:

1. **CNN Encoder**

   * Uses a pre-trained convolutional network (e.g., VGG16) to extract high-level image features.
   * Remove the final classification layer and add a global pooling layer to get a fixed-length feature vector.
2. **LSTM Decoder**

   * Tokenize and integer-encode all captions; pad sequences to the same length.
   * Use the CNN feature vector as the initial state for the LSTM.
   * At each time step, input the previous word embedding and predict the next word using teacher forcing.

---

## Requirements

* Python **3.8**+
* PyTorch
* 🤗 Transformers
* 🤗 Evaluate
* pandas
* Pillow
* torchvision

Install dependencies:

```bash
pip install torch transformers evaluate pandas pillow
```

---

## Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/iqrazehra/Image-Caption-Generator.git
cd Image-Caption-Generator
pip install -r requirements.txt
```

Prepare your data directories:

* **Images**: Place Flickr30k images in `images_dir` (default: `/content/drive/MyDrive/flickr30k_images/flickr30k_images`).
* **Captions**: Ensure `results.csv` is at `captions_csv` (default: `/content/drive/MyDrive/flickr30k_images/results.csv`).

---

## Dataset

`results.csv` format:

```
image_name | comment_number | comment
```

Each image must have exactly 5 associated captions.

---

## Usage

### BLIP Model

#### Inference Only

Generate captions with the pre-trained BLIP model (no fine-tuning):

```bash
  ./BLIP.ipnyb
```

#### Fine-tuning + Evaluation

Fine-tune BLIP for 3 epochs and evaluate on held-out 10%:

```bash
./BLIPtuned.ipynb
```

### CNN+LSTM Baseline

#### Training

Train the CNN+LSTM baseline for 50 epochs:

```bash
CNN.ipynb
```

#### Inference + Evaluation

Generate captions with the trained baseline and compute metrics:

```bash
python baseline/evaluate_baseline.py \
  --images_dir "/path/to/images" \
  --captions_csv "/path/to/results.csv" \
  --model_dir "baseline/models" \
  --output_file "baseline/predictions.json"
```

---

## Training Details

* **BLIP Fine-tuning**: 3 epochs, AdamW (lr=5e-5), max caption length=32.
* **CNN+LSTM Baseline**: 50 epochs, Adam (configurable), max caption length=32.
* **Batching**: Optional DataLoader for BLIP / mini-batch generator for baseline.
* **Mixed Precision**: Optional AMP for faster BLIP training.
* **Hardware**: Automatically uses GPU if available.

---

## Evaluation

Caption generation on the held-out 10% uses:

* **BLEU-1** (unigram precision)
* **BLEU-2** (bigram precision)
* **METEOR** (F1 variant with synonyms)

Results are printed or saved to the specified output.

---

## Saving the Model

Save the fine-tuned BLIP model and processor:

```python
model.save_pretrained(model_dir)
processor.save_pretrained(model_dir)
```

Reload for inference:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
processor = BlipProcessor.from_pretrained(model_dir)
model.eval()
```

---

## Results

### CNN+LSTM Baseline

* **Greedy Search**: BLEU-1=0.3186, BLEU-2=0.1350, METEOR=0.1621
* **Beam Search**: BLEU-1=0.3210, BLEU-2=0.1380, METEOR=0.1650

### BLIP

* **Pre-fine-tuning**: BLEU-1=0.5174, BLEU-2=0.3589, METEOR=0.3251
* **Post-fine-tuning**: BLEU-1=0.5920, BLEU-2=0.4100, METEOR=0.3400



---


## Acknowledgements

* Salesforce BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base
* Flickr30k Dataset: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
* Hugging Face Transformers & Evaluate libraries
