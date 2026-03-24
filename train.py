import os
import mlflow


target_accuracy = float(os.getenv("MOCK_ACCURACY", "0.80"))

print("Starting model training...")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "Classifier")
    mlflow.log_metric("accuracy", target_accuracy)
    run_id = run.info.run_id

print(f"Training completed. Accuracy: {target_accuracy}")
print(f"Run ID: {run_id}")

with open("model_info.txt", "w") as f:
    f.write(run_id)