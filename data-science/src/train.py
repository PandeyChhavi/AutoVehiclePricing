# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set MLflow tracking URI before importing mlflow
import os
os.environ['MLFLOW_TRACKING_URI'] = 'file:///tmp/mlruns'
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train_data
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test_data
    parser.add_argument("--model_output", type=str, help="Path of output model")  # Specify the type for model_output
    parser.add_argument('--n_estimators', type=float, default=100,
                        help='The number of trees in the forest')  # Specify the type and default value for n_estimators
    parser.add_argument('--max_depth', type=float, default=None,
                        help='The maximum depth of the tree')  # Specify the type and default value for max_depth

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from arguments
    train_df = pd.read_csv(Path(args.train_data) / "train_data.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test_data.csv")

    # Split the data into features(X) and target(y) 
    y_train = train_df['price']  # Specify the target column
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a RandomForest Regressor
    # Convert float values to int for RandomForestRegressor
    n_estimators = int(args.n_estimators) if args.n_estimators is not None else 100
    max_depth = int(args.max_depth) if args.max_depth is not None else None
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)  # Provide the arguments for RandomForestRegressor
    model.fit(X_train, y_train)  # Train the model

    # Log model hyperparameters (if MLflow is available)
    try:
        mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        print("✅ MLflow parameters logged successfully")
    except Exception as e:
        print(f"⚠️  MLflow parameter logging failed: {e}")
        print(f"Model: RandomForestRegressor")
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")

    # Predict using the RandomForest Regressor on test data
    yhat_test = model.predict(X_test)  # Predict the test data

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    
    try:
        mlflow.log_metric("MSE", float(mse))  # Log the MSE
        print("✅ MLflow metrics logged successfully")
    except Exception as e:
        print(f"⚠️  MLflow metric logging failed: {e}")

    # Save the model
    try:
        mlflow.sklearn.save_model(sk_model=model, path=args.model_output)  # Save the model
        print("✅ Model saved with MLflow successfully")
    except Exception as e:
        print(f"⚠️  MLflow model saving failed: {e}")
        # Fallback: save model using joblib
        import joblib
        joblib.dump(model, os.path.join(args.model_output, "model.pkl"))
        print("✅ Model saved with joblib as fallback")

if __name__ == "__main__":
    try:
        mlflow.start_run()
        print("✅ MLflow run started successfully")
        mlflow_available = True
    except Exception as e:
        print(f"⚠️  MLflow start_run failed: {e}")
        print("Continuing without MLflow tracking...")
        mlflow_available = False

    # Parse Arguments
    args = parse_args()

    # Convert float values to int for display
    n_estimators = int(args.n_estimators) if args.n_estimators is not None else 100
    max_depth = int(args.max_depth) if args.max_depth is not None else None
    
    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {n_estimators}",
        f"Max Depth: {max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    if mlflow_available:
        try:
            mlflow.end_run()
            print("✅ MLflow run ended successfully")
        except Exception as e:
            print(f"⚠️  MLflow end_run failed: {e}")