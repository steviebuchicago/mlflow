import random
from collections import namedtuple
from unittest import mock

import numpy as np
import pytest
import sklearn.neighbors as knn
from sklearn import datasets

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import add_libraries_to_model
from mlflow.models.utils import (
    _enforce_array,
    _enforce_datatype,
    _enforce_object,
    _enforce_property,
    get_model_version_from_model_uri,
)
from mlflow.types import DataType
from mlflow.types.schema import Array, Object, Property

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


def test_adding_libraries_to_model_default(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    wheeled_model_info = add_libraries_to_model(model_uri)
    assert wheeled_model_info.run_id == run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_run(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(model_uri)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_run_id_passed(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        pass

    wheeled_model_info = add_libraries_to_model(model_uri, run_id=wheeled_run_id)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_model_name(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    wheeled_model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{wheeled_model_name}/1"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        new_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(
            model_uri, registered_model_name=wheeled_model_name
        )
    assert wheeled_model_info.run_id == new_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == new_run_id
    assert wheeled_model_version.name == wheeled_model_name
    assert wheeled_model_name != model_name


def test_adding_libraries_to_model_when_version_source_None(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    model_version_without_source = ModelVersion(name=model_name, version=1, creation_timestamp=124)
    assert model_version_without_source.run_id is None
    with mock.patch.object(
        MlflowClient, "get_model_version", return_value=model_version_without_source
    ) as mlflow_client_mock:
        wheeled_model_info = add_libraries_to_model(model_uri)
        assert wheeled_model_info.run_id is not None
        assert wheeled_model_info.run_id != original_run_id
        mlflow_client_mock.assert_called_once_with(model_name, "1")


@pytest.mark.parametrize(
    ("data", "data_type"),
    [
        ("string", DataType.string),
        (np.int32(1), DataType.integer),
        (np.int32(1), DataType.long),
        (np.int32(1), DataType.double),
        (True, DataType.boolean),
        (1.0, DataType.double),
        (np.float32(0.1), DataType.float),
        (np.float32(0.1), DataType.double),
        (np.int64(100), DataType.long),
        (np.datetime64("2023-10-13 00:00:00"), DataType.datetime),
    ],
)
def test_enforce_datatype(data, data_type):
    assert _enforce_datatype(data, data_type) == data


def test_enforce_datatype_with_errors():
    with pytest.raises(MlflowException, match=r"Expected dtype to be DataType, got str"):
        _enforce_datatype("string", "string")

    with pytest.raises(
        MlflowException, match=r"Failed to enforce schema of data `123` with dtype `string`"
    ):
        _enforce_datatype(123, DataType.string)


def test_enforce_object():
    data = {
        "a": "some_sentence",
        "b": b"some_bytes",
        "c": ["sentence1", "sentence2"],
        "d": {"str": "value", "arr": [0.1, 0.2]},
    }
    obj = Object(
        [
            Property("a", DataType.string),
            Property("b", DataType.binary, required=False),
            Property("c", Array(DataType.string)),
            Property(
                "d",
                Object(
                    [
                        Property("str", DataType.string),
                        Property("arr", Array(DataType.double), required=False),
                    ]
                ),
            ),
        ]
    )
    assert _enforce_object(data, obj) == data

    data = {"a": "some_sentence", "c": ["sentence1", "sentence2"], "d": {"str": "some_value"}}
    assert _enforce_object(data, obj) == data


def test_enforce_object_with_errors():
    with pytest.raises(MlflowException, match=r"Expected data to be dictionary, got list"):
        _enforce_object(["some_sentence"], Object([Property("a", DataType.string)]))

    with pytest.raises(MlflowException, match=r"Expected obj to be Object, got Property"):
        _enforce_object({"a": "some_sentence"}, Property("a", DataType.string))

    obj = Object([Property("a", DataType.string), Property("b", DataType.string, required=False)])
    with pytest.raises(MlflowException, match=r"Missing required properties: {'a'}"):
        _enforce_object({}, obj)

    with pytest.raises(
        MlflowException, match=r"Invalid properties not defined in the schema found: {'c'}"
    ):
        _enforce_object({"a": "some_sentence", "c": "some_sentence"}, obj)

    with pytest.raises(MlflowException, match=r"Failed to enforce schema for key `a`"):
        _enforce_object({"a": 1}, obj)


def test_enforce_property():
    data = "some_sentence"
    prop = Property("a", DataType.string)
    assert _enforce_property(data, prop) == data

    data = ["some_sentence1", "some_sentence2"]
    prop = Property("a", Array(DataType.string))
    assert _enforce_property(data, prop) == data

    prop = Property("a", Array(DataType.binary))
    assert _enforce_property(data, prop) == ["some_sentence1", "some_sentence2"]

    data = {
        "a": "some_sentence",
        "b": b"some_bytes",
        "c": ["sentence1", "sentence2"],
        "d": {"str": "value", "arr": [0.1, 0.2]},
    }
    prop = Property(
        "any_name",
        Object(
            [
                Property("a", DataType.string),
                Property("b", DataType.binary, required=False),
                Property("c", Array(DataType.string), required=False),
                Property(
                    "d",
                    Object(
                        [
                            Property("str", DataType.string),
                            Property("arr", Array(DataType.double), required=False),
                        ]
                    ),
                ),
            ]
        ),
    )
    assert _enforce_property(data, prop) == data
    data = {"a": "some_sentence", "d": {"str": "some_value"}}
    assert _enforce_property(data, prop) == data


def test_enforce_property_with_errors():
    with pytest.raises(
        MlflowException, match=r"Failed to enforce schema of data `123` with dtype `string`"
    ):
        _enforce_property(123, Property("a", DataType.string))

    with pytest.raises(MlflowException, match=r"Expected data to be list, got ndarray"):
        _enforce_property(
            np.array(["some_sentence1", "some_sentence2"]), Property("a", Array(DataType.string))
        )

    with pytest.raises(MlflowException, match=r"Missing required properties: {'a'}"):
        _enforce_property(
            {"b": ["some_sentence1", "some_sentence2"]},
            Property(
                "any_name",
                Object([Property("a", DataType.string), Property("b", Array(DataType.string))]),
            ),
        )

    with pytest.raises(MlflowException, match=r"Failed to enforce schema for key `a`"):
        _enforce_property(
            {"a": ["some_sentence1", "some_sentence2"]},
            Property("any_name", Object([Property("a", DataType.string)])),
        )


def test_enforce_array():
    data = ["some_sentence1", "some_sentence2"]
    arr = Array(DataType.string)
    assert _enforce_array(data, arr) == data

    data = [
        {"a": "some_sentence1", "b": "some_sentence2"},
        {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
    ]
    arr = Array(
        Object(
            [
                Property("a", DataType.string),
                Property("b", DataType.string, required=False),
                Property("c", Array(DataType.string), required=False),
            ]
        )
    )
    assert _enforce_array(data, arr) == data


def test_enforce_array_with_errors():
    with pytest.raises(
        MlflowException, match=r"Failed to enforce schema of data `123` with dtype `string`"
    ):
        _enforce_array([123, 456, 789], Array(DataType.string))

    with pytest.raises(MlflowException, match=r"Expected data to be list, got ndarray"):
        _enforce_array(np.array(["some_sentence1", "some_sentence2"]), Array(DataType.string))

    with pytest.raises(MlflowException, match=r"Missing required properties: {'b'}"):
        _enforce_array(
            [
                {"a": "some_sentence1", "b": "some_sentence2"},
                {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
            ],
            Array(Object([Property("a", DataType.string), Property("b", DataType.string)])),
        )

    with pytest.raises(
        MlflowException, match=r"Invalid properties not defined in the schema found: {'c'}"
    ):
        _enforce_array(
            [
                {"a": "some_sentence1", "b": "some_sentence2"},
                {"a": "some_sentence3", "c": ["some_sentence4", "some_sentence5"]},
            ],
            Array(
                Object(
                    [Property("a", DataType.string), Property("b", DataType.string, required=False)]
                )
            ),
        )
