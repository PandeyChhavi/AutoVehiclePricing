# run_pipeline.py

import os
from pathlib import Path
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import Choice
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# --- 1. Connect to Azure ML Workspace ---
print("Connecting to Azure ML Workspace...")
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["SUBSCRIPTION_ID"],
    resource_group_name=os.environ["RESOURCE_GROUP"],
    workspace_name=os.environ["WORKSPACE_NAME"],
)
print(f"Connected to {ml_client.workspace_name}")

# Get the absolute path to the src directory
src_path = str(Path(__file__).parent / "src")

# --- 2. Setup Required Assets (Compute, Data, Environment) ---

# Create Compute Cluster if it doesn't exist
cpu_compute_target = "cpu-cluster"
try:
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(f"Found existing compute cluster '{cpu_compute_target}', reusing.")
except Exception:
    print(f"Creating a new compute cluster '{cpu_compute_target}'...")
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        type="amlcompute",
        size="Standard_DS11_v2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    ml_client.compute.begin_create_or_update(cpu_cluster).result()
    print("Compute cluster created.")

# Create Data Asset if it doesn't exist
data_asset_name = "used-cars-data"
try:
    data_asset = ml_client.data.get(name=data_asset_name, version="1")
    print(f"Found existing data asset '{data_asset_name}', reusing.")
except Exception:
    print(f"Creating a new data asset '{data_asset_name}'...")
    data_asset = Data(
        path='data/used_cars.csv',
        type=AssetTypes.URI_FILE,
        description="A dataset of used cars for price prediction",
        name=data_asset_name
    )
    ml_client.data.create_or_update(data_asset)
    print("Data asset created.")
    
# Create Environment if it doesn't exist
env_name = "machine_learning_E2E_v2"
try:
    pipeline_env = ml_client.environments.get(name=env_name, label="latest")
    print(f"Found existing environment '{env_name}', reusing.")
except Exception:
    print(f"Creating a new environment '{env_name}'...")
    pipeline_env = Environment(
        name=env_name,
        description="Environment for the Used Car Price Prediction pipeline",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="data-science/environment/train-conda.yml",
    )
    ml_client.environments.create_or_update(pipeline_env)
    print("Environment created.")


# --- 3. Define Pipeline Components ---

# Data Preparation Component
data_prep_component = command(
    name="data_preparation",
    display_name="Data Preparation and Splitting",
    description="Splits the raw data into training and test sets",
    inputs={
        "data": Input(type="uri_file"),
        "test_train_ratio": Input(type="number"),
    },
    outputs={
        "train_data": Output(type="uri_folder"),
        "test_data": Output(type="uri_folder"),
    },
    code=src_path,
    command="python prep.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}",
    environment=f"{env_name}@latest",
)

# Model Training Component
train_component = command(
    name="train_price_prediction_model",
    display_name="Train Car Price Prediction Model",
    description="Trains a Random Forest Regressor model.",
    inputs={
        "train_data": Input(type="uri_folder"),
        "test_data": Input(type="uri_folder"),
        "n_estimators": Input(type="number", default=100),
        "max_depth": Input(type="number", default=5),
    },
    outputs={"model_output": Output(type="mlflow_model")},
    code=src_path,
    command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --model_output ${{outputs.model_output}}",
    environment=f"{env_name}@latest",
)

# Model Registration Component
model_register_component = command(
    name="register_model",
    display_name="Register Best Model",
    description="Registers the best model from the sweep job.",
    inputs={"model": Input(type="mlflow_model")},
    outputs={"registered_model": Output(type="mlflow_model")},
    code=src_path,
    command="python register.py --model_name best_model --model_path ${{inputs.model}} --model_info_output_path ${{outputs.registered_model}}",
    environment=f"{env_name}@latest",
)


# --- 4. Define and Assemble the Full Pipeline ---

@pipeline(
    compute=cpu_compute_target,
    description="End-to-end pipeline for car price prediction",
)
def car_price_pipeline(input_data_uri, test_train_ratio):
    # Step 1: Preprocess Data
    preprocess_step = data_prep_component(
        data=input_data_uri,
        test_train_ratio=test_train_ratio,
    )

    # Step 2: Train and Tune Model using a Sweep Job
    train_step = train_component(
        train_data=preprocess_step.outputs.train_data,
        test_data=preprocess_step.outputs.test_data,
        n_estimators=Choice(values=[10, 20, 50, 100]),
        max_depth=Choice(values=[5, 10, 15, 20]),
    )
    
    # Apply sweep to the training step
    sweep_job = train_step.sweep(
        compute=cpu_compute_target,
        sampling_algorithm="random",
        primary_metric="MSE",
        goal="Minimize",
    )
    sweep_job.set_limits(max_total_trials=10, max_concurrent_trials=2, timeout=7200)

    # Step 3: Register the best model from the sweep
    model_register_step = model_register_component(
        model=sweep_job.outputs.model_output,
    )
    
    return {
        "best_model": model_register_step.outputs.registered_model,
    }

# --- 5. Submit the Pipeline Job ---

print("Instantiating pipeline...")
# Get the latest version of the data asset
latest_data_asset = ml_client.data.get(name=data_asset_name, label="latest")
pipeline_instance = car_price_pipeline(
    input_data_uri=Input(type="uri_file", path=latest_data_asset.path),
    test_train_ratio=0.2,
)

print("Submitting pipeline job...")
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_instance,
    experiment_name=os.environ["EXPERIMENT_NAME"],
)

print(f"Pipeline job '{pipeline_job.name}' submitted. View in Azure ML Studio.")
# Stream the job output
ml_client.jobs.stream(pipeline_job.name)