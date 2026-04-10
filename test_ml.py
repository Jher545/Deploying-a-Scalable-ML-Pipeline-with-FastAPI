import pytest
import numpy as np
import pandas as pd
import os
from ml.data import process_data
from ml.model import train_model, inference, load_model
# TODO: implement the first test. Change the function name and input as needed
#testing small batch of data
@pytest.fixture
def data():
    project_path = "/workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    data_path = os.path.join(project_path, "data", "census.csv")
    return pd.read_csv(data_path).iloc[:100]
def test_one(data):
    """
    Testing process data
    #check y is binary
    """
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    X, y, encoder, lb = process_data(data, cat_features, label="salary",training=True)
    #Checking X array and # of rows
    assert isinstance(X, np.ndarray)
    assert len(X) == len(data)
    #check y is binary
    assert set(np.unique(y)).issubset({0,1})
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two(data):
    """
    Testing train_model
    #check function returns model with predict method
    """
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    X, y, _, _ = process_data(data, cat_features, label="salary", training = True)
    model = train_model(X, y)
    #check function returns model with predict method
    assert hasattr(model, "predict")
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three(data):
    """
    #testing test_inference
    # add description for the third test
    """
    cat_features =["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    X, y, encoder, lb = process_data(data, cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)

    #Checking exactly 1 prediction per input row
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)    
    pass
