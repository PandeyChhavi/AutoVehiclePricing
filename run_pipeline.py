# run_pipeline.py
# Submits complete_pipeline.yml (with sweep) to Azure ML

import os
import sys
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

print("="*70)
print("🚀 Azure ML Production Pipeline Submission")
print("="*70)

# --- 1. Check Environment Variables ---
required_env_vars = ["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WORKSPACE_NAME", "EXPERIMENT_NAME"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
    print("\nPlease set the following environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
    print("\nFor GitHub Actions, add these as repository secrets.")
    sys.exit(1)

print("✅ All required environment variables are set")

# --- 2. Connect to Azure ML Workspace ---
print("\n📡 Connecting to Azure ML Workspace...")

try:
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["SUBSCRIPTION_ID"],
        resource_group_name=os.environ["RESOURCE_GROUP"],
        workspace_name=os.environ["WORKSPACE_NAME"],
    )
    print(f"✅ Connected to workspace: {ml_client.workspace_name}")
except Exception as e:
    print(f"❌ Failed to connect to Azure ML workspace: {e}")
    sys.exit(1)

# --- 3. Setup Required Assets ---
print("\n🔧 Setting up required Azure ML assets...")

# Compute Cluster
cpu_compute_target = "cpu-cluster"
try:
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(f"✅ Compute cluster '{cpu_compute_target}' found")
except Exception:
    print(f"⚙️  Creating compute cluster '{cpu_compute_target}'...")
    try:
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size="Standard_DS11_v2",
            min_instances=0,
            max_instances=2,  # Increased for parallel sweep trials
            idle_time_before_scale_down=180,
            tier="Dedicated",
        )
        ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print(f"✅ Compute cluster '{cpu_compute_target}' created")
    except Exception as e:
        print(f"❌ Failed to create compute cluster: {e}")
        sys.exit(1)

# Data Asset
data_asset_name = "used-cars-data"
try:
    data_asset = ml_client.data.get(name=data_asset_name, label="latest")
    print(f"✅ Data asset '{data_asset_name}' found")
except Exception:
    print(f"📊 Creating data asset '{data_asset_name}'...")
    try:
        data_path = os.path.abspath('data/used_cars.csv')
        
        if not os.path.exists(data_path):
            print(f"❌ Data file not found at: {data_path}")
            sys.exit(1)
        
        data_asset = Data(
            path=data_path,
            type=AssetTypes.URI_FILE,
            description="A dataset of used cars for price prediction",
            name=data_asset_name
        )
        ml_client.data.create_or_update(data_asset)
        print(f"✅ Data asset '{data_asset_name}' created")
    except Exception as e:
        print(f"❌ Failed to create data asset: {e}")
        sys.exit(1)

# Environments
env_names = ["used-cars-train-env", "used_cars_train_env_v2"]
for env_name in env_names:
    try:
        pipeline_env = ml_client.environments.get(name=env_name, label="latest")
        print(f"✅ Environment '{env_name}' found")
    except Exception:
        print(f"🐍 Creating environment '{env_name}'...")
        try:
            conda_file = "data-science/environment/train-conda.yml"
            
            if not os.path.exists(conda_file):
                print(f"⚠️  Conda file not found at: {conda_file}")
                print(f"   Skipping environment creation for '{env_name}'")
                continue
            
            pipeline_env = Environment(
                name=env_name,
                description=f"Environment for Used Car Price Prediction pipeline ({env_name})",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file=conda_file,
            )
            ml_client.environments.create_or_update(pipeline_env)
            print(f"✅ Environment '{env_name}' created")
        except Exception as e:
            print(f"⚠️  Failed to create environment '{env_name}': {e}")
            print(f"   Continuing anyway - pipeline may use existing environment...")

# --- 4. Load and Submit Pipeline ---
print("\n" + "="*70)
print("📋 LOADING PRODUCTION PIPELINE (complete_pipeline.yml)")
print("="*70)

pipeline_yaml_path = "mlops/azureml/train/complete_pipeline.yml"

if not os.path.exists(pipeline_yaml_path):
    print(f"❌ Pipeline YAML not found at: {pipeline_yaml_path}")
    print("\n💡 Available files in mlops/azureml/train/:")
    try:
        import glob
        yaml_files = glob.glob("mlops/azureml/train/*.yml")
        for f in yaml_files:
            print(f"   - {f}")
    except:
        pass
    sys.exit(1)

print(f"📂 Loading pipeline from: {pipeline_yaml_path}")

try:
    pipeline_job = load_job(source=pipeline_yaml_path)
    print("✅ Pipeline YAML loaded successfully")
    print(f"\n📊 Pipeline Configuration:")
    print(f"   • Display Name: {pipeline_job.display_name}")
    print(f"   • Description: {pipeline_job.description}")
    if hasattr(pipeline_job, 'jobs'):
        print(f"   • Number of steps: {len(pipeline_job.jobs)}")
except Exception as e:
    print(f"❌ Failed to load pipeline YAML: {e}")
    sys.exit(1)

# Submit Pipeline
print(f"\n🚀 Submitting pipeline to experiment: {os.environ['EXPERIMENT_NAME']}")

try:
    pipeline_run = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=os.environ["EXPERIMENT_NAME"],
    )
    
    print("\n" + "="*70)
    print("✅ PIPELINE SUBMITTED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📋 Pipeline Details:")
    print(f"   • Name: {pipeline_run.name}")
    print(f"   • Status: {pipeline_run.status}")
    print(f"   • Experiment: {os.environ['EXPERIMENT_NAME']}")
    
    print(f"\n🔗 View in Azure ML Studio:")
    print(f"   {pipeline_run.studio_url}")
    
    print(f"\n📊 This pipeline will:")
    print(f"   1. Preprocess data (train/test split)")
    print(f"   2. Run hyperparameter sweep (20 trials)")
    print(f"   3. Select best model based on MSE")
    print(f"   4. Register best model to registry")
    
    print(f"\n⏱️  Estimated time: 15-30 minutes")
    print(f"💡 Monitor progress in Azure ML Studio using the link above")
    
except Exception as e:
    print(f"\n❌ Failed to submit pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ Script completed successfully!")
print("="*70)