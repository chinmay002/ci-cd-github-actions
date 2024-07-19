import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import joblib
import boto3
from dotenv import load_dotenv
import os


# Set your S3 bucket details
s3_bucket = "artifacts-mage-to-s3"
s3_artifact_uri = f"s3://{s3_bucket}/mlflow"

# Set the MLflow tracking URI
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


load_dotenv()
access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.amazonaws.com"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Experiment")




# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = LogisticRegression(max_iter=200)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("max_iter", model.max_iter)
    mlflow.log_param("solver", model.solver)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    model_uri = f"runs:/{run.info.run_id}/model"
    model_details = mlflow.register_model(model_uri, "Iris_LogisticRegression_Model")

    print(f"Registered model: {model_details.name} version: {model_details.version}")

    print(f"Logged model with accuracy: {accuracy}")

print("MLflow run completed and logged to S3.")

#download it locally
# Corrected model version URI format
model_version_uri = f"models:/{model_details.name}/{model_details.version}"

local_path = mlflow.artifacts.download_artifacts(model_version_uri)


#load the mdoel
model = mlflow.sklearn.load_model(model_version_uri)

# Save the model as a .pkl file
pkl_path = f"{model_details.name}_v{model_details.version}.pkl"
joblib.dump(model, pkl_path)

# Upload the .pkl file to S3
s3_client = boto3.client("s3")
s3_client.upload_file(pkl_path, s3_bucket, f"models/{pkl_path}")