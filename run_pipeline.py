# run_pipeline.py

import os
import sys
from azure.ai.ml import MLClient, load_component, Input, Output
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# --- 1. Connect to Azure ML Workspace ---
print("Connecting to Azure ML Workspace...")

# Check if required environment variables are set
required_env_vars = ["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WORKSPACE_NAME", "EXPERIMENT_NAME"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set the following environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
    print("\nFor GitHub Actions, add these as repository secrets:")
    print("  - SUBSCRIPTION_ID")
    print("  - RESOURCE_GROUP") 
    print("  - WORKSPACE_NAME")
    print("  - EXPERIMENT_NAME")
    print("  - AZURE_CREDENTIALS")
    sys.exit(1)

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
    # Use absolute path to ensure Azure ML can find the file
    data_path = os.path.abspath('data/used_cars.csv')
    print(f"Data file path: {data_path}")
    
    data_asset = Data(
        path=data_path,
        type=AssetTypes.URI_FILE,
        description="A dataset of used cars for price prediction",
        name=data_asset_name
    )
    ml_client.data.create_or_update(data_asset)
    print("Data asset created.")
    
# Create Environment if it doesn't exist
env_name = "used_cars_train_env_v2"
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
model_register_component = load_component(source="mlops/azureml/train/register.yml")


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
print(f"Data asset path: {latest_data_asset.path}")
print(f"Data asset type: {latest_data_asset.type}")

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
print(f"RunId: {pipeline_job.name}")
print(f"Web View: https://ml.azure.com/runs/{pipeline_job.name}?wsid=/subscriptions/{os.environ['SUBSCRIPTION_ID']}/resourcegroups/{os.environ['RESOURCE_GROUP']}/workspaces/{os.environ['WORKSPACE_NAME']}")

# Wait for job completion and get status
print("\nWaiting for pipeline completion...")
try:
    # Wait for the job to complete (with timeout)
    ml_client.jobs.stream(pipeline_job.name)
except Exception as e:
    print(f"Pipeline execution completed with status: {pipeline_job.status}")
    print(f"Error details: {e}")
    
    # Try to get more detailed error information
    try:
        job_details = ml_client.jobs.get(pipeline_job.name)
        print(f"Job status: {job_details.status}")
        if hasattr(job_details, 'services') and job_details.services:
            print("Job services:", job_details.services)
    except Exception as detail_error:
        print(f"Could not get job details: {detail_error}")