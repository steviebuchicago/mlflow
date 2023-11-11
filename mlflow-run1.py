import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
    
# Set the MLflow tracking URI to the local server
mlflow.set_tracking_uri("http://localhost:5000")
    
# Generate some synthetic data for regression
X, y = np.random.rand(100, 10), np.random.rand(100)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the experiment name
mlflow.set_experiment('My_Local_Experiment')

# Start an MLflow run
with mlflow.start_run():

    # Set the MLflow tracking URI to the local server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Log parameters (here just hyperparameters for the model)
    mlflow.log_param("fit_intercept", True)

    # Train a linear regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate and log a metric
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Create a simple artifact text file and log it
    with open("output.txt", "w") as f:
        f.write("This is a simple text artifact.")
    mlflow.log_artifact("output.txt")

print(f"Model trained, MSE: {mse}")
