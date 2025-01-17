import os

import numpy as np
import openai

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec


    
# Set the MLflow tracking URI to the local server
mlflow.set_tracking_uri("http://34.213.211.223:80")

# Define the experiment name
mlflow.set_experiment('openai')

OPENAI_API_KEY= "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"

assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"


print(
    """
# ******************************************************************************
# Text embeddings
# ******************************************************************************
"""
)

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-embedding-ada-002",
        task=openai.Embedding,
        artifact_path="model",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["hello", "world"]))


print(
    """
# ******************************************************************************
# Text embeddings with batch_size parameter
# ******************************************************************************
"""
)

with mlflow.start_run():
    mlflow.openai.log_model(
        model="text-embedding-ada-002",
        task=openai.Embedding,
        artifact_path="model",
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
            params=ParamSchema([ParamSpec(name="batch_size", dtype="long", default=1024)]),
        ),
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["hello", "world"], params={"batch_size": 16}))
