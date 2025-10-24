#!/usr/bin/env python3
"""
Local testing script for the Auto Vehicle Pricing project.
This script can run without Azure ML dependencies for basic testing.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("🔍 Testing data loading...")
    try:
        df = pd.read_csv('data/used_cars.csv')
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def test_data_preprocessing(df):
    """Test data preprocessing steps."""
    print("\n🔍 Testing data preprocessing...")
    try:
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"⚠️  Missing values found: {missing_values[missing_values > 0].to_dict()}")
        else:
            print("✅ No missing values found")
        
        # Test categorical encoding
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded['Segment'] = le.fit_transform(df_encoded['Segment'])
        print("✅ Categorical encoding successful")
        
        # Test train/test split
        train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)
        print(f"✅ Train/test split successful: {train_df.shape[0]} train, {test_df.shape[0]} test")
        
        return train_df, test_df
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None, None

def test_model_training(train_df, test_df):
    """Test model training and evaluation."""
    print("\n🔍 Testing model training...")
    try:
        # Prepare features and target
        y_train = train_df['price']
        X_train = train_df.drop(columns=['price'])
        y_test = test_df['price']
        X_test = test_df.drop(columns=['price'])
        
        print(f"   Features: {X_train.shape[1]}, Train samples: {X_train.shape[0]}")
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Model training successful")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model evaluation:")
        print(f"   MSE: {mse:.2f}")
        print(f"   R²: {r2:.2f}")
        
        return model, mse, r2
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None, None

def test_mlflow_integration(model):
    """Test MLflow integration."""
    print("\n🔍 Testing MLflow integration...")
    try:
        # Start MLflow run
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(model, "test_model")
            print("✅ MLflow integration successful")
        return True
    except Exception as e:
        print(f"❌ MLflow integration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Auto Vehicle Pricing Local Tests")
    print("=" * 50)
    
    # Test 1: Data loading
    df = test_data_loading()
    if df is None:
        print("\n❌ Tests failed at data loading stage")
        sys.exit(1)
    
    # Test 2: Data preprocessing
    train_df, test_df = test_data_preprocessing(df)
    if train_df is None or test_df is None:
        print("\n❌ Tests failed at preprocessing stage")
        sys.exit(1)
    
    # Test 3: Model training
    model, mse, r2 = test_model_training(train_df, test_df)
    if model is None:
        print("\n❌ Tests failed at model training stage")
        sys.exit(1)
    
    # Test 4: MLflow integration
    mlflow_success = test_mlflow_integration(model)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Data loading: ✅")
    print(f"   Data preprocessing: ✅")
    print(f"   Model training: ✅ (MSE: {mse:.2f}, R²: {r2:.2f})")
    print(f"   MLflow integration: {'✅' if mlflow_success else '❌'}")
    
    if mlflow_success:
        print("\n🎉 All tests passed! The project is ready for deployment.")
    else:
        print("\n⚠️  Most tests passed, but MLflow integration needs attention.")

if __name__ == "__main__":
    main()
