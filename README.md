# Mammography Preprocessing Pipeline

A modular mammography preprocessing framework for artifact suppression, pectoral muscle removal, and image standardization prior to deep learning analysis. 

## 🔬 Research Context
This repository contains selected components developed during a six-month research internship focused on mammography preprocessing and medical image analysis. The associated manuscript is currently under review.

## 🚀 Key Engineering Features
* **Deep Segmentation Suppression:** Utilizing a U-Net architecture integrated with a ResNet18 encoder to identify and isolate the pectoral muscle wall.
* **Traditional Computer Vision Blending:** Implementation of mathematical distance transforms, Gaussian blurring, and morphological opening/closing operations to feather tissue edges smoothly into background space.
* **Reproducible Normalization Engine:** Multi-stage min-max contrast stretching, background heuristic calculation, and single-channel intensity scaling.

## 📂 Repository Layout
* `pipeline.py`: Main integration module structuring the unified execution sequence.
* `artifact_removal.py`: Implements pixel boundary cleanup and distance mapping suppression.
* `pectoral_suppression.py`: Handles deep-learning inference configurations and post-prediction mask processing.
* `mask_blending.py`: Isolates spatial feathering algorithms and matrix weight vectors.
* `normalization.py`: Manages resolution resizing, intensity mapping, and grayscale transformations.
* `utils.py`: Modular visualization subroutines, comparison matrix utilities, and panel configurations.

## 🛠️ Usage
```python
from pipeline import MammographyPreprocessingPipeline

# Initialize pipeline with model weights path
pipeline = MammographyPreprocessingPipeline(weights_path="path/to/weights.pth")

# Process single mammogram file
cleaned_tissue = pipeline.preprocess_image("sample_mammogram.png")