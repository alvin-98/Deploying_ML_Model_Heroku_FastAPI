# Put the code for your API here.
from fileinput import filename
from fastapi import FastAPI
import joblib
from starter.ml.model import model_inference
from starter.ml.data import process_data
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pandas as pd
import os


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Instantiate the app.
app = FastAPI()

class Person(BaseModel):
    age: int
    workclass: Literal["Private",
                    "Self-emp-not-inc",
                    "Self-emp-inc",
                    "Federal-gov",
                    "Local-gov",
                    "State-gov",
                    "Without-pay",
                    "Never-worked"]
    fnlwgt: int
    education: Literal["Bachelors",
                    "Some-college",
                    "11th",
                    "HS-grad",
                    "Prof-school",
                    "Assoc-acdm",
                    "Assoc-voc",
                    "9th",
                    "7th-8th",
                    "12th",
                    "Masters",
                    "1st-4th",
                    "10th",
                    "Doctorate",
                    "5th-6th",
                    "Preschool"]
    educationNum: int = Field(None, alias='education-num')
    maritalStatus: Literal["Married-civ-spouse",
                        "Divorced",
                        "Never-married",
                        "Separated",
                        "Widowed",
                        "Married-spouse-absent",
                        "Married-AF-spouse"] = Field(None, alias='marital-status')
    occupation: Literal["Tech-support",
                    "Craft-repair",
                    "Other-service",
                    "Sales",
                    "Exec-managerial",
                    "Prof-specialty",
                    "Handlers-cleaners",
                    "Machine-op-inspct",
                    "Adm-clerical",
                    "Farming-fishing",
                    "Transport-moving",
                    "Priv-house-serv",
                    "Protective-serv",
                    "Armed-Forces"]
    relationship: Literal["Wife",
                    "Own-child",
                    "Husband",
                    "Not-in-family",
                    "Other-relative",
                    "Unmarried"]
    race: Literal["White",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other",
                "Black"]
    sex: Literal["Female", "Male"]
    capitalGain: int = Field(None, alias='capital-gain')
    capitalLoss: int = Field(None, alias='capital-loss')
    hoursPerWeek: int = Field(None, alias='hours-per-week')
    nativeCountry: Literal["United-States",
                        "Cambodia",
                        "England",
                        "Puerto-Rico",
                        "Canada",
                        "Germany",
                        "Outlying-US(Guam-USVI-etc)",
                        "India",
                        "Japan",
                        "Greece",
                        "South",
                        "China",
                        "Cuba",
                        "Iran",
                        "Honduras",
                        "Philippines",
                        "Italy",
                        "Poland",
                        "Jamaica",
                        "Vietnam",
                        "Mexico",
                        "Portugal",
                        "Ireland",
                        "France",
                        "Dominican-Republic",
                        "Laos",
                        "Ecuador",
                        "Taiwan",
                        "Haiti",
                        "Columbia",
                        "Hungary",
                        "Guatemala",
                        "Nicaragua",
                        "Scotland",
                        "Thailand",
                        "Yugoslavia",
                        "El-Salvador",
                        "Trinadad&Tobago",
                        "Peru",
                        "Hong",
                        "Holand-Netherlands"]  = Field(None, alias='native-country')
    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": "Private",
                "fnlwgt": 500,
                "education": "Bachelors",
                "education-num": 10,
                "marital-status": "Divorced",
                "occupation": "Sales",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital-gain": 500,
                "capital-loss": 20,
                "hours-per-week": 50,
                "native-country": "India",
            }
        }


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"Welcome Message": "Welcome to Salary Prediction Model!"}

@app.post("/")
async def inference(person: Person):
    filename = "model/rf_model.sav"
    rf_model= joblib.load(filename)

    filename = "model/encoder.sav"
    encoder = joblib.load(filename)

    filename = "model/lb.sav"
    lb = joblib.load(filename)

    # input_data = np.array([[person.age,
    #                 person.workclass,
    #                 person.fnlwgt,
    #                 person.education,
    #                 person.educationNum,
    #                 person.maritalStatus,
    #                 person.occupation,
    #                 person.relationship,
    #                 person.race,
    #                 person.sex,
    #                 person.capitalGain,
    #                 person.capitalLoss,
    #                 person.hoursPerWeek,
    #                 person.nativeCountry]])
    data = pd.DataFrame(person.dict())
    # data = pd.DataFrame(data=input_data, 
    #                     columns=["age",
    #                     "workclass",
    #                     "fnlwgt",
    #                     "education",
    #                     "education-num",
    #                     "marital-status",
    #                     "occupation",
    #                     "relationship",
    #                     "race",
    #                     "sex",
    #                     "capital-gain",
    #                     "capital-loss",
    #                     "hours-per-week",
    #                     "native-country"])
    X, y, _, _ = process_data(
                        data, 
                        categorical_features=[
                                                "workclass",
                                                "education",
                                                "marital-status",
                                                "occupation",
                                                "relationship",
                                                "race",
                                                "sex",
                                                "native-country",
                                            ],
                        training=False,
                        encoder=encoder
                        )

    pred = model_inference(rf_model, X)
    y_pred = lb.inverse_transform(pred[0])
    return {"prediction": y_pred[0]}