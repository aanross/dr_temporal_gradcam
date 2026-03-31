# Time-aware Diabetic Retinopathy Detection Study

A comparative study framework for time-aware diabetic retinopathy (DR) detection with lesion mapping using GradCAM, utilizing the APTOS 2019 Blindness Detection dataset.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A Kaggle account and API token (`kaggle.json`)

### 2. Installation
Clone the repository and install the dependencies:
```bash
cd dr_temporal_gradcam
pip install -r requirements.txt
```

### 3. Kaggle Configuration
Ensure your `kaggle.json` file is correctly placed in `~/.kaggle/kaggle.json` and has the correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```
You will also need to accept the rules for the [APTOS 2019 competition](https://www.kaggle.com/c/aptos2019-blindness-detection/rules) on Kaggle.

### 4. Running the Pipeline
You can run the full cross-validation pipeline passing arguments for the model and the GradCAM method.

```bash
# Example
python src/train.py --model_type resnet50_lstm --cam_method gradcam++ --epochs 20
```

## Models Supported
- `resnet_baseline`: ResNet50 (Single time point)
- `resnet50_lstm`: ResNet50 + LSTM
- `efficientnet_bilstm`: EfficientNet-B3 + BiLSTM
- `vit_temporal`: Vision Transformer + Temporal Transformer
- `timesformer`: TimeSformer
- `convlstm`: ConvLSTM Network

## GradCAM Methods Supported
- `gradcam`
- `gradcam++`
- `scorecam`
- `layercam`

## Documenting Your Results
When you finish your evaluation using the real data, you can use the provided **`results_report_template.md`** template. 
This template structurally organizes your:
1. Classification vs Regression metrics
2. Generated Critical Difference (DeLong) ranks
3. Spatiotemporal GradCAM galleries and Lesion IoU
4. Plot imagery (`ROC`, `Radar Chart`) generated natively into the `results/` folder

Just copy the template to your destination paper/repository and fill in the bracketed `[Value]` placeholders from your Weights & Biases experiment logs!
