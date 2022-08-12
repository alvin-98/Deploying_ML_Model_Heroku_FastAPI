import pytest
import pandas as pd
import numpy as np
import joblib
import logging


logging.basicConfig(
    filename='./logs/pipeline_logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def clean_df():
    filename = "data/clean_data.csv"
    input_df = pd.read_csv(filename)
    return input_df


def test_clean_data(clean_df: pd.DataFrame):
    """
    We test if the function clean_data generates valid dataframes
    """
    try:
        assert clean_df.shape[0] > 0
        assert clean_df.shape[1] > 0
        logging.info("Testing clean_data: SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing clean_data: The clean dataframe doesn't appear to have rows and columns"
        )
        raise err

    
def test_process_data(clean_df: pd.DataFrame):
    """
    We test if the function process_data generates valid outputs
    """
    try:
        X_train = np.load('data/X_processed_train.npy')
        y_train = np.load('data/y_processed_train.npy')
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert len(y_train) > 0
        logging.info("Testing process_data: SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing process_data: The processed file doesn't appear to have rows and columns"
        )
        raise err


def test_column_names(clean_df: pd.DataFrame):
    """
    We test whether all columns are present in our dataset
    """
    expected_columns = ['age',
                        'workclass',
                        'fnlgt',
                        'education',
                        'education-num',
                        'marital-status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'capital-gain',
                        'capital-loss',
                        'hours-per-week',
                        'native-country',
                        'salary'
                        ]
    try:
        assert list(clean_df.columns) == expected_columns
        logging.info("Testing for completeness of column names: SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing for completeness of column names: The columns in data doesn't match the expected columns"
        )
        raise err


def test_model(clean_df: pd.DataFrame):
    """
    We test whether the train_model produces a trained random forest classifier
    """
    X_train = np.load('data/X_processed_train.npy')
    y_train = np.load('data/y_processed_train.npy')
    try:
        filename = "model/rf_model.sav"
        rf_model= joblib.load(filename)
        preds = rf_model.predict(X_train)
        assert len(preds) == len(y_train)
        logging.info("Testing inference: SUCCESS")

    except AssertionError as err:
        logging.error(
        "Testing train_model: The trained model doesn't work as expected"
        )
        raise err