import json
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture
def client():
    app_client = TestClient(app)
    return app_client


def test_get(client):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome Message": "Welcome to Salary Prediction Model!"}


def test_post_gt50k(client):
    client = TestClient(app)
    response = client.post("/", json={
        "age": 55,
        "workclass": "Private",
        "fnlwgt": 500,
        "education": "Doctorate",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Sales",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capital-gain": 50000,
        "capital-loss": 20,
        "hours-per-week": 50,
        "native-country": "United-States",
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_post_ls50k(client):
    client = TestClient(app)
    response = client.post("/", json={
        "age": 55,
        "workclass": "Private",
        "fnlwgt": 500,
        "education": "Bachelors",
        "education-num": 1,
        "marital-status": "Divorced",
        "occupation": "Sales",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 100,
        "capital-loss": 50,
        "hours-per-week": 30,
        "native-country": "India",
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}