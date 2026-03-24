import sys
import os
import mlflow

# Force MLflow to read from the exact local path where GitHub downloaded the artifact
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if not tracking_uri:
    tracking_uri = f"file://{os.getcwd()}/mlruns"
mlflow.set_tracking_uri(tracking_uri)

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("Error: model_info.txt not found.")
    sys.exit(1)

print(f"Fetching metrics for Run ID: {run_id}")

try:
    run = mlflow.get_run(run_id)
    accuracy = float(run.data.metrics.get("accuracy", 0.0))
except Exception as e:
    print(f"Error fetching MLflow run: {e}")
    sys.exit(1)

print(f"Detected Model Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Validation FAILED: Accuracy is below the 0.85 threshold.")
    sys.exit(1)
else:
    print("✅ Validation PASSED: Accuracy meets the threshold.")
    sys.exit(0)