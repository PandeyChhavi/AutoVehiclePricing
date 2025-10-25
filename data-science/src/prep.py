# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Debug: Print arguments
    print(f"Raw data path: {args.raw_data}")
    print(f"Train data path: {args.train_data}")
    print(f"Test data path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    
    # Check if raw data file exists
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # In Azure ML, the data might be mounted differently
    # Check if the path exists as-is first
    if os.path.exists(args.raw_data):
        data_file_path = args.raw_data
        print(f"✅ Found data file at: {data_file_path}")
    else:
        # Try to find the data file in common Azure ML mount locations
        print(f"❌ Data file not found at {args.raw_data}")
        print("Searching for data file in common locations...")
        
        # List all files recursively to find the CSV
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv') and 'used_cars' in file.lower():
                    data_file_path = os.path.join(root, file)
                    print(f"✅ Found data file at: {data_file_path}")
                    break
            else:
                continue
            break
        else:
            print("❌ Could not find used_cars.csv file anywhere")
            raise FileNotFoundError(f"Raw data file not found: {args.raw_data}")

    # Reading Data
    print(f"Reading data from: {data_file_path}")
    df = pd.read_csv(data_file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Encode categorical feature
    le = LabelEncoder()
    df['Segment'] = le.fit_transform(df['Segment'])  # Write code to encode the categorical feature

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)  #  Write code to split the data into train and test datasets

    # Save the train and test data
    os.makedirs(args.train_data, exist_ok=True)  # Create directories for train_data and test_data
    os.makedirs(args.test_data, exist_ok=True)  # Create directories for train_data and test_data
    train_df.to_csv(os.path.join(args.train_data, "train_data.csv"), index=False)  # Specify the name of the train data file
    test_df.to_csv(os.path.join(args.test_data, "test_data.csv"), index=False)  # Specify the name of the test data file

    # log the metrics
    mlflow.log_metric('train size', train_df.shape[0])  # Log the train dataset size
    mlflow.log_metric('test size', test_df.shape[0])  # Log the test dataset size

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()