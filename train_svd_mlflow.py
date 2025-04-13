import pandas as pd
from surprise import SVD, Dataset, Reader
import joblib
import os
import tempfile
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from svd_wrapper import SurpriseSVDWrapper

# Load ratings data
ratings = pd.read_csv("data/rating.csv")  # Must contain user_id, anime_id, rating

# Train the SVD model
reader = Reader(rating_scale=(1, 10))  # Adjust rating scale if needed
data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)  

# Create input example for model signature
input_example = pd.DataFrame({
    'user_id': [1, 2],  # Example user IDs
    'anime_id': [101, 102]  # Example anime IDs
})

# Log to MLflow
with mlflow.start_run(run_name="SVD_Collaborative_Filtering") as run:
    # Get the current directory where the Python script resides
    current_dir = os.path.dirname(__file__)

    # Define the path to save the model in the same directory as the script
    model_path = os.path.join(current_dir, "svd_model.pkl")

    # Save the model using joblib
    joblib.dump(model, model_path)

    mlflow.pyfunc.log_model(
        artifact_path="svd_model",
        python_model=SurpriseSVDWrapper(),
        artifacts={"model_path": model_path},
        registered_model_name="svd_model",  # Register the model
        input_example=input_example  # Provide input example for signature inference
    )
    
    run_id = run.info.run_id  # Now safely get the run_id after the run is started

# Transition model to 'Production' stage
client = MlflowClient()
model_name = "svd_model"
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

# Transition the latest model to 'Production'
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)

# Delete older versions of the model (if needed)
versions = client.get_latest_versions(model_name, stages=["None"])
for version in versions[1:]:  # Skip the latest version (version 0-indexed)
    client.delete_model_version(model_name, version.version)
    print(f"Deleted model version {version.version}")

print(f"Model logged successfully with run ID: {run_id}")
print(f"Model version {latest_version} has been moved to 'Production'.")
