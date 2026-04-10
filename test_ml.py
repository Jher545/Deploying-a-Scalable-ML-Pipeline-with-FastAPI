import pytest
import numpy as np
import pandas as pd
import os
from ml.data import process_data
from ml.model import train_model, inference, load_model
# TODO: implement the first test. Change the function name and input as needed
#testing small batch of data
current_dir = os.path.dirname(os.path.abspath(__file__))
@pytest.fixture
def data():
    df = pd.DataFrame({
        "age": [39, 50, 38],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private"],
        "fnlgt": [77513, 83311, 215646],
        "education": ["Bachelors", "Bachelors", "HS-grad"],
        "education-num": [13, 13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family"],
        "race": ["White", "White", "White"],
        "sex": ["Male", "Male", "Male"],
        "capital-gain": [2174, 0, 0],
        "capital-loss": [0, 0, 0],
        "hours-per-week": [40, 13, 40],
        "native-country": ["United-States", "United-States", "United-States"],
        "salary": ["<=50K", "<=50K", "<=50K"]
    })
    return df
def test_one(data):
    """
    Testing process data
    #check process data produces shapes and binary labels
    """
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    X, y, _, _ =process_data(
        data, categorical_features=cat_features,label="salary",training=True
    )
    assert isinstance(X, np.ndarray)
    assert len(X) == len(data)
    #check y is binary
    assert set(np.unique(y)).issubset({0,1})


# TODO: implement the second test. Change the function name and input as needed
def test_two(data):
    """
    Testing train_model
    #Check the train_model returns model using predict method
    """
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    
    X, y, _, _ = process_data(data, cat_features, label="salary", training = True)
    model = train_model(X, y)
    #check function returns model with predict method
    assert hasattr(model, "predict")
    
# TODO: implement the third test. Change the function name and input as needed
def test_three(data):
    """
    #testing inference produces # of predictions correctly
    """
    cat_features =["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    X, y, _, _ =process_data(data, cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)

    #Checking exactly 1 prediction per input row
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)
