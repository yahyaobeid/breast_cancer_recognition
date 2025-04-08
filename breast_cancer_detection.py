import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import tensorflow as tf

def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction."""
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No tumor contour found in the image")
    
    # Get the largest contour (assumed to be the tumor)
    tumor_contour = max(contours, key=cv2.contourArea)
    
    # Create a copy of the original image for visualization
    processed_img = img.copy()
    cv2.drawContours(processed_img, [tumor_contour], -1, (0, 255, 0), 2)
    
    # Save the processed image
    processed_path = image_path.rsplit('.', 1)[0] + '_processed.' + image_path.rsplit('.', 1)[1]
    cv2.imwrite(processed_path, processed_img)
    
    # Calculate features
    features = calculate_features(tumor_contour, gray)
    
    # Reshape for prediction
    features = np.array(features).reshape(1, -1)
    
    return features, processed_path

def calculate_features(contour, image):
    """Calculate tumor features from the contour."""
    # Basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate center of mass
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise ValueError("Invalid contour")
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Calculate distances from center to points
    distances = []
    for point in contour.reshape(-1, 2):
        dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Create mask for the tumor region
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    tumor_region = image[mask == 255]
    
    # Calculate features that match the Wisconsin dataset
    # Mean features
    radius_mean = np.mean(distances)
    texture_mean = np.std(tumor_region)  # Standard deviation of gray-scale values
    perimeter_mean = perimeter
    area_mean = area
    smoothness_mean = np.std(distances) / radius_mean
    compactness_mean = (perimeter ** 2) / (4 * np.pi * area)
    concavity_mean = np.mean(np.abs(distances - radius_mean))
    concave_points_mean = len(distances[distances < radius_mean]) / len(distances)
    symmetry_mean = np.abs(np.mean(distances[:len(distances)//2]) - np.mean(distances[len(distances)//2:]))
    fractal_dimension_mean = np.log(perimeter) / np.log(area)
    
    # Standard error features
    radius_se = np.std(distances)
    texture_se = np.std(tumor_region) / np.sqrt(len(tumor_region))
    perimeter_se = radius_se * 2 * np.pi
    area_se = 2 * radius_mean * radius_se * np.pi
    smoothness_se = np.std(smoothness_mean)
    compactness_se = np.std(compactness_mean)
    concavity_se = np.std(concavity_mean)
    concave_points_se = np.std(concave_points_mean)
    symmetry_se = np.std(symmetry_mean)
    fractal_dimension_se = np.std(fractal_dimension_mean)
    
    # "Worst" features (mean of the three largest values)
    radius_worst = np.mean(np.sort(distances)[-3:])
    texture_worst = np.max(texture_mean)
    perimeter_worst = 2 * np.pi * radius_worst
    area_worst = np.pi * radius_worst ** 2
    smoothness_worst = np.max(smoothness_mean)
    compactness_worst = np.max(compactness_mean)
    concavity_worst = np.max(concavity_mean)
    concave_points_worst = np.max(concave_points_mean)
    symmetry_worst = np.max(symmetry_mean)
    fractal_dimension_worst = np.max(fractal_dimension_mean)
    
    # Create feature vector matching Wisconsin dataset order
    features = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst, compactness_worst,
        concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]
    
    return features

def load_and_preprocess_data():
    """Load and preprocess the breast cancer dataset."""
    # Load the dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_model():
    """Create and return a pipeline with preprocessing and model."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    return pipeline

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model and print performance metrics."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

def plot_feature_importance(model, X):
    """Plot feature importance."""
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_importance = pd.Series(feature_importance, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    feature_importance.head(10).plot(kind='bar')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Creating and training model...")
    model = create_model()
    
    print("Evaluating model...")
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("Plotting feature importance...")
    plot_feature_importance(model, X_train)
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main() 