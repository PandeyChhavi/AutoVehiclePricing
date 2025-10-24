# Auto Vehicle Pricing

A machine learning pipeline for predicting used car prices using Azure ML and scikit-learn.

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AutoVehiclePricing
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run local tests**
   ```bash
   python test_local.py
   ```

### Azure ML Setup

1. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Azure ML workspace details
   ```

2. **Set up Azure ML workspace**
   - Create an Azure ML workspace
   - Get your subscription ID, resource group, and workspace name
   - Update the `.env` file with your credentials

3. **Run the pipeline**
   ```bash
   python run_pipeline.py
   ```

## 📁 Project Structure

```
AutoVehiclePricing/
├── data/                          # Data files
│   └── used_cars.csv             # Training data
├── data-science/                  # ML code
│   ├── src/                      # Source code
│   │   ├── prep.py              # Data preprocessing
│   │   ├── train.py             # Model training
│   │   └── register.py          # Model registration
│   └── environment/              # Environment configs
│       └── train-conda.yml      # Conda environment
├── mlops/                        # MLOps configurations
│   └── azureml/                  # Azure ML components
│       └── train/               # Training components
├── .github/workflows/            # GitHub Actions
│   └── ci.yml                   # CI/CD pipeline
├── requirements.txt              # Python dependencies
├── run_pipeline.py              # Main pipeline script
└── test_local.py                # Local testing script
```

## 🔧 GitHub Actions

This project includes GitHub Actions for:
- **Testing**: Automated testing on push/PR
- **Linting**: Code quality checks
- **Azure ML**: Automated deployment to Azure ML (on main branch)

### Required GitHub Secrets

For Azure ML deployment, add these secrets to your GitHub repository:

- `AZURE_CREDENTIALS`: Azure service principal credentials
- `SUBSCRIPTION_ID`: Azure subscription ID
- `RESOURCE_GROUP`: Azure resource group name
- `WORKSPACE_NAME`: Azure ML workspace name
- `EXPERIMENT_NAME`: Azure ML experiment name

## 🧪 Testing

### Local Testing
```bash
python test_local.py
```

### GitHub Actions Testing
The CI/CD pipeline automatically runs:
- Data loading tests
- Preprocessing tests
- Model training tests
- MLflow integration tests

## 📊 Data

The dataset contains used car information with the following features:
- `Segment`: Car segment (luxury/non-luxury)
- `Kilometers_Driven`: Total kilometers driven
- `Mileage`: Fuel efficiency
- `Engine`: Engine displacement
- `Power`: Engine power
- `Seats`: Number of seats
- `price`: Target variable (car price)

## 🤖 Model

The pipeline uses a Random Forest Regressor with:
- Hyperparameter tuning via Azure ML sweep
- Cross-validation
- MLflow model tracking and registration

## 🚀 Deployment

### Local Deployment
1. Run `python test_local.py` to verify everything works
2. Configure Azure ML credentials
3. Run `python run_pipeline.py`

### GitHub Deployment
1. Push to main branch
2. GitHub Actions will automatically:
   - Run tests
   - Deploy to Azure ML (if secrets are configured)

## 🛠️ Troubleshooting

### Common Issues

1. **Azure ML Authentication**
   - Ensure your Azure credentials are correctly configured
   - Check that your service principal has the right permissions

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

3. **Data Loading Issues**
   - Verify `data/used_cars.csv` exists
   - Check file permissions

4. **GitHub Actions Failures**
   - Check that all required secrets are set
   - Verify Azure ML workspace is accessible

### Getting Help

- Check the GitHub Actions logs for detailed error messages
- Run `python test_local.py` to identify local issues
- Ensure all environment variables are properly set

## 📝 License

This project is licensed under the MIT License.
