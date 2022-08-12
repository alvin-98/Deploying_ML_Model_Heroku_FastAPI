import json
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture(scope="session")
def test_client():
    client = TestClient(app)
    return client


def get_test(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome Message": "Welcome to Salary Prediction Model!"}


def post_test_gt50k(client):
    response = client.post("/", json={
        "age": 55,
        "workclass": "Private",
        "fnlwgt": 500,
        "education": "Doctorate",
        "educationNum": 10,
        "maritalStatus": "Divorced",
        "occupation": "Sales",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capitalGain": 50000,
        "capitalLoss": 20,
        "hoursPerWeek": 50,
        "nativeCountry": "United-States",
    })
    assert response.status_code == 200
    assert response.json == {"prediction": ">50K"}


def post_test_ls50k(client):
    response = client.post("/", json={
        "age": 55,
        "workclass": "Private",
        "fnlwgt": 500,
        "education": "Bachelors",
        "educationNum": 1,
        "maritalStatus": "Divorced",
        "occupation": "Sales",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capitalGain": 100,
        "capitalLoss": 50,
        "hoursPerWeek": 30,
        "nativeCountry": "India",
    })
    assert response.status_code == 200
    assert response.json == {"prediction": "<=50K"}