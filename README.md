# MLFlow Iris Classification

This project demonstrates a basic local setup for tracking machine learning experiments using [MLflow](https://mlflow.org/) and scikit-learn. The model trains a `RandomForestClassifier` on the classic Iris dataset.

## Setup

First, ensure you have initialized your Python virtual environment and installed the dependencies in the iris_mlflow folder:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running Training Runs

To execute a training run and log the parameters, metrics, and models to MLflow, run the `train.py` script:

```bash
# Ensure your virtual environment is active
source venv/bin/activate

# Run the training script
python3 train.py
```

Each time you run this script, it will create a new run under the `iris_classification` experiment. All tracked data and model artifacts will be saved locally in an `mlruns/` directory created in this folder.

## Viewing the MLflow UI

To visualize your experiments, compare runs, and look at the logged metrics, use the MLflow tracking UI:

```bash
# Ensure your virtual environment is active
source venv/bin/activate

# Start the MLflow UI server
mlflow ui
```

Once the server starts, open your web browser and navigate to:

**http://127.0.0.1:5000**

From the UI, you can:

- View the `iris_classification` experiment.
- See individual runs and their logged parameters (`n_estimators`, `max_depth`, `random_state`).
- Check the logged metrics (`accuracy`).
- Inspect the saved `random_forest_model` artifacts.

To stop the MLflow UI server, press `Ctrl+C` in the terminal where it is running.
