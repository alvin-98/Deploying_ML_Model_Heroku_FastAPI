# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Made by: Alvin Vinod
- Model Type: Random Forest Model
- Hyperparameters: 
    - max_depth = 5
    - n_jobs = -1
    - random_state = 42

## Intended Use
Model is to be used to predict whether the income of an individual is above or below USD 50,000. The features mentioned in the [link](https://archive.ics.uci.edu/ml/datasets/census+income) are used to make the prediction.

## Training Data
- 80% of the available data was used for training.
- Extraction was done by Barry Becker from the 1994 Census database. More details available in the source website [here](https://archive.ics.uci.edu/ml/datasets/census+income).
- The features are one hot encoded using the sklearn OneHotEncoder with sparse=False and handle_unknown="ignore", while the target variable is encoded using a default LabelBinarizer.

## Evaluation Data
- 20% of the total data is used for evaluation.
- The features have been one hot encoded using the sklearn OneHotEncoder with sparse=False and handle_unknown="ignore".

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- Precision: 0.80
- Recall: 0.60
- F1 Score: 0.68

## Ethical Considerations
The dataset has race and sex as predictors. This may cause a negative feedback loop and bias the model towards common stereotypes.

## Caveats and Recommendations
- The data is imbalanced. More data collection is required with a balance of various classes in mind.
- More complex models such as XGBoost may give better results.
- Model may be biased towards certain classes.