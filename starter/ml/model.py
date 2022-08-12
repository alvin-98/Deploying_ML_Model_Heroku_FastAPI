from xml.etree.ElementPath import prepare_predicate
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import yaml
from yaml import CLoader as Loader
import os
import json
from starter.ml.data import process_data
# from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf_config = os.path.abspath("rf_config.json")

    with open("starter/ml/params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)

    random_Forest = RandomForestClassifier(**params)
    random_Forest.fit(X_train, y_train)

    return random_Forest



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def model_inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_model_slice_performance(model, cat_features, X):
    """
    Validated the model performance on categorical slices using precision, recall and F1.

    Inputs
    ------
    model: sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    
    cat_features: list
        List of categorical features.

    X : np.array
        Data used for prediction.
    Returns
    -------
    performance: dict
        model performance on slices of categorical features.
    """
    performance = {}

    for category in cat_features:
        performance[category] = {}
        unique_cat = X[category].unique()
        for cl in unique_cat:
            X_class = X[X[category] == cl]
            X_test, y_test, encoder, lb = process_data(
                                        X_class, categorical_features=cat_features,training=False
                                    )
            preds = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            performance[category][cl] = {'precision': precision,
                                        'recall': recall,
                                        'f1': fbeta}
    
    with open('performance.json', 'w') as fp:
        json.dump(performance, fp,  indent=4)