import requests, json

data_1 = {
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
    }
response = requests.post('https://project3-api-development.herokuapp.com/', data=json.dumps(data_1))

print(data_1)
print(response.status_code)
print(response.json())

print('-----------------------')
data_2 = {
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
response = requests.post('https://project3-api-development.herokuapp.com/', data=json.dumps(data_2))

print(data_2)
print(response.status_code)
print(response.json())