# Image Segmentation for Autonomous Vehicles Using K-Means Clustering

Image segmentation using the k-means algorithm on the Cityscapes dataset.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). 
Registration is required to download the data.

Download the following files:
- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Extract both archives into the `data/` folder:
```bash
unzip leftImg8bit_trainvaltest.zip -d data/
unzip gtFine_trainvaltest.zip -d data/
```

Expected structure:
```
data/
  leftImg8bit/
    train/
      aachen/
        aachen_000000_000019_leftImg8bit.png
        ...
  gtFine/
    train/
      aachen/
        aachen_000000_000019_gtFine_labelIds.png
        ...
```

## Usage

Open the notebook:
```bash
jupyter notebook notebook.ipynb
```

Run cells in order.

## Authors

Łukasz Rudnik, Mikołaj Suchan  
Mathematical Foundations of Artificial Intelligence  
Jagiellonian University, 2025/2026