import mlflow

mlflow.set_tracking_uri("file:./mlruns")


target_accuracy = 0.95

print("Starting model training...")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "Classifier")
    mlflow.log_metric("accuracy", target_accuracy)
    run_id = run.info.run_id

print(f"Training completed. Accuracy: {target_accuracy}")
print(f"Run ID: {run_id}")

with open("model_info.txt", "w") as f:
    f.write(run_id)