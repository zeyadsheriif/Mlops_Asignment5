import os
import mlflow

# Use a portable SQLite database instead of a folder to survive the artifact transfer
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if not tracking_uri:
    tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

target_accuracy = 0.95

print(f"Using Tracking URI: {tracking_uri}")
print("Starting model training...")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "Classifier")
    mlflow.log_metric("accuracy", target_accuracy)
    run_id = run.info.run_id

print(f"Training completed. Accuracy: {target_accuracy}")
print(f"Run ID: {run_id}")

with open("model_info.txt", "w") as f:
    f.write(run_id)