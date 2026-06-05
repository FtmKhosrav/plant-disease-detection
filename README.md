# Plant Disease Detection Project (Corn, Tomato, Rice, Wheat)

## Overview
This project is a deep learning-based plant disease classification system for four crops: Corn, Tomato, Rice, and Wheat.

Two transfer learning models were used:
- VGG16 for Corn and Rice
- MobileNetV2 for Tomato and Wheat

The system classifies leaf images into healthy or diseased categories and provides a recommended treatment for each disease.


## Datasets
The project uses a merged dataset from Kaggle:

- https://www.kaggle.com/datasets/yogeshwaran005/plants-disease-detection
- https://www.kaggle.com/datasets/emmarex/plantdisease

The datasets were cleaned, merged, and reorganized into a unified structure.


## Model Performance

| Plant  | Model        | Accuracy |
|--------|-------------|----------|
| Corn   | VGG16       | 94.06%   |
| Tomato | MobileNetV2 | 91.51%   |
| Rice   | VGG16       | 65.91%   |
| Wheat  | MobileNetV2 | 96.58%   |


## Project Structure
plant-disease-detection/
├── src/
│   ├── data_preprocessing.py
│   ├── model_corn.py
│   ├── model_tomato.py
│   ├── model_rice.py
│   ├── model_wheat.py
│   ├── inference.py
├── models/
│   ├── corn_model.h5
│   ├── tomato_model.h5
│   ├── rice_model.h5
│   ├── wheat_model.h5
├── notebooks/
│   ├── corn_model.ipynb
│   ├── tomato_model.ipynb
│   ├── rice_model.ipynb
│   ├── wheat_model.ipynb
├── requirements.txt
├── main.py
└── README.md


## Workflow

1. Data Preprocessing
- Dataset merging
- Train/Validation/Test split (70/15/15)
- Organized folder structure per plant and disease

2. Data Augmentation
- Rotation
- Zoom
- Shift
- Horizontal flip

3. Model Training
- Transfer learning (VGG16 / MobileNetV2)
- Frozen base layers
- Fully connected classifier
- Softmax output

4. Evaluation
- Accuracy calculation
- Confusion matrix

5. Inference
- Upload image
- Predict disease
- Output class + confidence + solution


## How to Run

Install dependencies:
pip install -r requirements.txt

Run inference:
python main.py


## Features
- Multi-crop disease detection
- Deep transfer learning models
- Disease treatment suggestions
- Modular structure
- Separate models per plant


## Results Summary
Best performance:
- Wheat: 96.58%
- Corn: 94.06%
- Tomato: 91.51%
- Rice: 65.91% (due to dataset imbalance)


## Authors
Fateme Khosravi


## Note
Dataset files are not included due to Kaggle licensing. Only code and trained models are provided.
