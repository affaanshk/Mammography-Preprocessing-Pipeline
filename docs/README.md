# Preprocessing Visualization Assets

This directory contains the documentation diagrams, qualitative comparisons, and evaluation charts for the **Mammography Preprocessing Pipeline**.

---

## 🖼️ Included Visual Assets

### 1. `pipeline_overview.png`
* **Description:** End-to-end architectural flowchart of the preprocessing system.
* **Workflow Stages:** Dataset Ingestion ──> Image Intensity Normalization ──> Artifact & Text Removal ──> Pectoral Muscle Segmentation & Feathering ──> Clean Mammogram Output.

---

### 2. `artifact_removal_examples.png`
* **Description:** Qualitative examples illustrating automated detection and suppression of non-breast structured artifacts.
* **Details:** Highlights the removal of scanner metadata, view markers, L/R orientation overlays, and high-contrast acquisition borders while preserving underlying parenchymal tissue.

---

### 3. `preprocessing_examples.png`
* **Description:** Visual progression of screening mammograms across preprocessing stages.
* **Details:** A multi-panel comparison showcasing the step-by-step transition from raw screening images containing acquisition noise to standardized, breast-only representations.

---

### 4. `pectoral_suppression_examples.png`
* **Description:** Qualitative evaluation of pectoral muscle identification and suppression on mediolateral oblique (MLO) views.
* **Details:** Displays the application of predicted U-Net segmentation masks combined with feathered edge blending to prevent sharp artificial boundary transitions.

---

### 5. `preprocessing_metrics.png`
* **Description:** Quantitative metrics summary for the preprocessing segmentation modules.
* **Details:** Reports boundary precision and region overlap using Dice Similarity Coefficient (Dice), Intersection-over-Union (IoU), Boundary Accuracy, and Structural Similarity Index Measure (SSIM).
