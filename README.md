# Breast Cancer Detection

This project implements a machine learning model to predict whether a breast tumor is malignant or benign using the Wisconsin Breast Cancer dataset.

## Dataset

The Wisconsin Breast Cancer dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The features describe characteristics of the cell nuclei present in the image.

Features include:
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter² / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python breast_cancer_detection.py
```

The script will:
1. Load and preprocess the dataset
2. Split the data into training and testing sets
3. Train a machine learning model
4. Evaluate the model's performance
5. Display results and visualizations

## Model Performance

The model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

## License

This project is open source and available under the MIT License. 