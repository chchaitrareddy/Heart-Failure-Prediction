# ==============================================================================
# File: train_model.py
# Purpose: Load Framingham Heart Study dataset, clean missing values, train a
#          Random Forest Classifier model, evaluate its performance, and save it
#          using Python's built-in pickle module.
# ==============================================================================

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    print("==========================================")
    print("1. LOADING DATASET")
    print("==========================================")
    
    # Path to the dataset CSV file
    dataset_path = os.path.join("dataset", "heart.csv")
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: Dataset not found at path '{dataset_path}'")
        
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully! Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

    print("\n==========================================")
    print("2. HANDLING MISSING VALUES")
    print("==========================================")
    
    # Print missing values count per column before filling
    print("Missing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Fill numerical column missing values with their respective column median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in column '{col}' with median value: {median_val}")
            
    print("Missing values after cleaning:", df.isnull().sum().sum())

    print("\n==========================================")
    print("3. SEPARATING FEATURES (X) AND TARGET (y)")
    print("==========================================")
    
    # Target variable is 'TenYearCHD' (1 = Risk of Coronary Heart Disease in 10 yrs, 0 = Low Risk)
    target_column = 'TenYearCHD'
    
    # X contains feature columns; y contains the target binary output
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    print("\n==========================================")
    print("4. TRAIN-TEST SPLIT")
    print("==========================================")
    
    # Split dataset: 80% for training the model, 20% for testing model accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    print("\n==========================================")
    print("5. TRAINING RANDOM FOREST CLASSIFIER")
    print("==========================================")
    
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train (fit) the model on the training data
    model.fit(X_train, y_train)
    print("Random Forest Model trained successfully!")

    print("\n==========================================")
    print("6. MODEL EVALUATION")
    print("==========================================")
    
    # Predict outcomes on the test dataset
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n==========================================")
    print("7. SAVING THE MODEL WITH PYTHON PICKLE")
    print("==========================================")
    
    # Ensure model directory exists
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pkl")
    
    # Serialize and save model object to disk using standard pickle module
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Model saved successfully using pickle to '{model_path}'!")
    print("==========================================")

if __name__ == "__main__":
    train_and_save_model()
