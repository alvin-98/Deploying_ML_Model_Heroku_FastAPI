# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.data import get_cat_list
from ml.model import train_model
from ml.data import clean_data
from ml.model import compute_model_slice_performance
import joblib
# Add the necessary imports for the starter code.

# Add code to load in the data.
filename = "data/census.csv"
raw_data = pd.read_csv(filename)

cat_features = get_cat_list(raw_data, label="salary")

clean_df = clean_data(raw_data, cat_features)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(clean_df, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True, save=True
)

# Process the test data with the process_data function.
X_test, _, _, _ = process_data(
    test, categorical_features=cat_features, training=False, encoder=encoder, lb=lb, save=True
)

# Train and save a model.
trained_model = train_model(X_train, y_train)

filename = "model/rf_model.sav"
joblib.dump(trained_model, filename)

filename = "model/encoder.sav"
joblib.dump(encoder, filename)

filename = "model/lb.sav"
joblib.dump(lb, filename)

slice_performance = compute_model_slice_performance(trained_model, cat_features, test, encoder, lb)