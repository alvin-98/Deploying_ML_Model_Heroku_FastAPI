import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def get_cat_list(raw_data, label=None):
    raw_data.columns = [col.strip() for col in raw_data.columns]
    cat_features = list(set(raw_data.columns)- set(raw_data.describe().columns))
    cat_features.remove(label)
    return cat_features


def clean_data(raw_data, categorical_features=[]):
    """
    Cleans input dataframe by remove unknown values and additional spaces.

    Inputs
    ------
    raw_data : pd.DataFrame
        raw dataframe as read from the input csv file
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])

    Returns
    -------
    clean_df : pd.DataFrame
        clean dataframe after removing unknown values and additional spaces
    """
    
    clean_df = pd.DataFrame()
    clean_df = raw_data.copy(deep=True)
    clean_df.columns = [col.strip() for col in raw_data.columns]
    clean_df[categorical_features + ['salary']] = clean_df[categorical_features + ['salary']].applymap(str.strip)
    clean_df.replace('?', np.NaN, inplace=True)
    clean_df.dropna(inplace=True)
    clean_df.to_csv('data/clean_data.csv', index=False)

    return clean_df


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    np.save(f"data/X_processed_{'train' if training else 'test'}.npy", X)
    if training:
        np.save(f"data/y_processed_train.npy", y)
    return X, y, encoder, lb
