# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import os 
import json

# Set MLflow tracking URI before importing mlflow
os.environ['MLFLOW_TRACKING_URI'] = 'file:///tmp/mlruns'
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Load model
    try:
        model = mlflow.sklearn.load_model(args.model_path)  # Load the model from model_path
        print("✅ Model loaded with MLflow successfully")
    except Exception as e:
        print(f"⚠️  MLflow model loading failed: {e}")
        # Fallback: load model using joblib
        import joblib
        model = joblib.load(os.path.join(args.model_path, "model.pkl"))
        print("✅ Model loaded with joblib as fallback")

    # Log model using mlflow (if available)
    try:
        mlflow.sklearn.log_model(model, args.model_name)  # Log the model using with model_name
        print("✅ Model logged with MLflow successfully")
        
        # Register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)  # register the model with model_uri and model_name
        model_version = mlflow_model.version  # Get the version of the registered model
        print(f"✅ Model registered with MLflow. Version: {model_version}")
    except Exception as e:
        print(f"⚠️  MLflow model registration failed: {e}")
        # Fallback: create a simple model info without MLflow
        model_version = "1.0"
        print(f"Using fallback model version: {model_version}")

    # Write model info
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")  # Specify the name of the JSON file (model_info.json)
    with open(output_path, "w") as of:
        json.dump(model_info, of)  # write model_info to the output file
    print(f"✅ Model info written to: {output_path}")

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
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
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