# Brain Tumour Classification with Bias Mitigation

This project is a machine learning solution for classifying brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary. By integrating bias mitigation techniques, the model achieves improved accuracy (target: 89%) and ensures fairer predictions across different groups.

## Project Highlights

- **Bias Mitigation:** Incorporates fairness-aware training and evaluation to reduce bias in predictions, leading to more equitable healthcare outcomes.
- **Improved Accuracy:** Bias mitigation not only addresses fairness but also helps the model generalize better, resulting in higher accuracy.
- **YOLOv8-based Detection:** Utilizes the latest YOLOv8 architecture for robust and efficient image classification.
- **Comprehensive Evaluation:** Includes fairness metrics and statistical tests to assess and report model bias.

## Project Structure

- `data.yaml` — Dataset configuration file (paths to training, validation, and test sets, class names, and dataset source)
- `train/`, `valid/`, `test/` — Folders containing image data (not included in the repository; see below)
- `train.py` — Main training and evaluation script, including bias mitigation and fairness analysis
- Jupyter notebooks — For exploratory data analysis and visualization 

## Classes

The model classifies MRI images into the following four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Dataset

- The dataset is referenced in `data.yaml` and is sourced from [Roboflow](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset/dataset/1).
- **Note:** The actual image data (`train/`, `valid/`, `test/`) is not included in this repository. Please download the dataset from the provided Roboflow link and place it in the appropriate folders.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/brain-tumour-with-bias-mitigation.git
    cd brain-tumour-with-bias-mitigation
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib pyyaml jupyter numpy pandas ultralytics
    ```

## Usage

1. **Prepare the dataset:**
    - Download the dataset from Roboflow and extract it into the `train/`, `valid/`, and `test/` folders as specified in `data.yaml`.

2. **Train the model with bias mitigation:**
    ```bash
    python train.py --config data.yaml
    ```

    The training script will:
    - Train the YOLOv8 model on the MRI dataset
    - Apply bias mitigation strategies during training
    - Evaluate the model on the test set
    - Compute fairness metrics and perform bias analysis

3. **Jupyter Notebooks:**
    - For data exploration or visualization, launch Jupyter:
      ```bash
      jupyter notebook
      ```

## Results

- The model aims to achieve an accuracy of **89%** on the test set.
- Bias mitigation techniques (such as fairness-aware loss functions, disparate impact analysis, and chi-square bias testing) help ensure that the model’s predictions are both accurate and fair.

## Bias Mitigation Approach

- **Fairness Metrics:** The training pipeline computes false positive/negative rates and disparate impact ratios to quantify bias.
- **Statistical Testing:** Chi-square tests are used to detect significant bias in predictions.
- **Impact:** By addressing bias, the model not only becomes fairer but also achieves better generalization and accuracy, as observed in validation results.


