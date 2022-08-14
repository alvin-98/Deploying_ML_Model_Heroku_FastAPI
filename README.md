Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories

* Create a directory for the project and initialize Git and DVC.
* Connect your local Git repository to GitHub.

## GitHub Actions

* GitHub Actions are setup on this repository. Pre-made "Python Application" has been setup with the python version 3.8 which runs pytest and flake8 on push and requires both to pass without error.

## Data

* Download census.csv from the data folder in the starter repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* This data is messy, with additional spaces and NaN values.
* The current pipeline, remove all additional spaces and also removes rows with NaN values.
* After cleaning, you can commit this modified data to DVC under a new name.

## Model

* A Random Forest Model has been trained on the cleaned version with the following hyperparameters: 
   - max_depth = 5
   - n_jobs = -1
   - random_state = 42 (for both the Random forest model and train_test_split)
* `compute_model_slice_performance` function outputs the performance of the model on all categorical slices of the data. This is saved in the file 'slice_output.json'.

## API Creation & Deployment

* A FastAPI RESTful API has been built and deployed on [Heroku](https://project3-api-development.herokuapp.com/):
   * GET on the root: Displays a welcome message.
   * POST on the root: Provides model inference.
* There are 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).