import numpy as np
from sklearn.linear_model import LogisticRegression


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


    
# Set the MLflow tracking URI to the local server
mlflow.set_tracking_uri("http://34.213.211.223:80")

# Define the experiment name
mlflow.set_experiment('sklearn')


if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print(f"Score: {score}")
    mlflow.log_metric("score", score)
    predictions = lr.predict(X)
    signature = infer_signature(X, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
