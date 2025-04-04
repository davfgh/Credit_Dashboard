import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def test_predict_endpoint():
    """ Vérifie que le endpoint /predict fonctionne bien et retourne une réponse correcte. """
    print(f"Testing API at: {API_URL}")

    sample_data = {
        "AMT_ANNUITY": 15000,
        "AMT_CREDIT": 200000,
        "AMT_GOODS_PRICE": 180000,
        "CREDIT_TERM": 60,
        "DAYS_BIRTH": -12000,
        "DAYS_ID_PUBLISH": -3000,
        "DAYS_REGISTRATION": -4000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.6,
        "DEBT_CREDIT_RATIO": 0.3,
        "ANNUITY_BIRTH_RATIO": 0.02,
        "ANNUITY_INCOME_PERCENT": 0.1,
        "CREDIT_GOODS_RATIO": 1.1,
        "INSTA_AMT_PAYMENT": 10000,
        "INSTA_NUM_INSTALMENT_VERSION": 3,
        "POS_CNT_INSTALMENT_FUTURE": 2,
        "PREV_CNT_PAYMENT": 12
    }

    response = requests.post(f"{API_URL}/predict", json=sample_data)
    print("Response:", response.json())
    assert response.status_code == 200, f"Erreur API : {response.status_code}"
    response_data = response.json()
    assert "prediction" in response_data
    assert "probability_class_1" in response_data

def test_shap_values_endpoint():
    """ Vérifie que le endpoint /shap_values retourne bien les valeurs SHAP. """
    response = requests.get(f"{API_URL}/shap_values")
    assert response.status_code == 200, f"Erreur API : {response.status_code}"
    response_data = response.json()
    assert "shap_values" in response_data
    assert "features_names" in response_data
