# run_pipeline.py

import os
from azure.ai.ml import MLClient, load_component, Input, Output
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes
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


# --- 3. Load Pipeline Components from YAML ---
print("Loading components...")
data_prep_component = load_component(source="mlops/azureml/train/data.yml")
train_component = load_component(source="mlops/azureml/train/train.yml")
model_register_component = load_component(source="mlops/azureml/train/newpipeline.yml")


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

    # Step 2: Train Model
    train_step = train_component(
        train_data=preprocess_step.outputs.train_data,
        test_data=preprocess_step.outputs.test_data,
    )

    # Step 3: Register the model
    model_register_step = model_register_component(
        model=train_step.outputs.model_output,
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