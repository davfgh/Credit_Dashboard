# Documentation de l'API et de l'Interface Streamlit pour le Scoring Client

## 1️. Introduction

Cette web application permet d'effectuer des **prédictions sur la fiabilité d'un client** en fonction de ses caractéristiques financières.

Elle repose sur **deux composants principaux** :
- **Une API Flask**, exposant un modèle de machine learning, permettant d'obtenir un score de risque et des valeurs explicatives via SHAP.
- **Une interface utilisateur Streamlit**, permettant d’explorer les prédictions et d’interagir avec le modèle de manière intuitive.

L'API peut être utilisée **en local** ou **via le déploiement sur le cloud**.

---

## 2️. **Architecture**

### API Flask
- Chargement du modèle stocké dans un fichier pickle.
- Vérification des données d'entrée.
- Prédiction du risque client via le modèle.
- Retour des résultats sous forme de réponse JSON.

### Interface Streamlit
- Affichage des **informations client**.
- Comparaison des caractéristiques avec des moyennes globales.
- Génération d’une **prédiction et ajustement d’un seuil dynamique**.
- Visualisation des **valeurs SHAP locales** pour comprendre l'impact des variables sur la décision.

---

## 3️. **Modélisation**

### **Notebook de Modélisation**
Un notebook dédié à la modélisation est disponible à la racine du projet :
`Farizon_David_notebook_modelisation_012025.ipynb`

Ce notebook comprend les étapes suivantes :
- **Feature extraction** à partir d'un notebook issu de la compétition Kaggle **Home Credit Default Risk**.
- **Analyse exploratoire rapide** des features extraites.
- **Modélisation avec 3 modèles** :
  - Dummy Classifier
  - Régression Logistique
  - LightGBM
- **Optimisation du seuil de probabilités** sur le meilleur modèle sélectionné.
- **Dernier entraînement du modèle sélectionné** sur le jeu d'entraînement.
- **Export du modèle final** au format **pickle**.

### **Suivi des Expériences avec MLFlow**
Le projet intègre **MLFlow** pour suivre et enregistrer les expériences de modélisation :
- Stockage des modèles et des hyperparamètres.
- Enregistrement automatique des métriques de performance.
- Visualisation des runs d'entraînement.

Les expériences sont stockées dans :
`models/` *(Dossier MLFlow)*
`mlruns.db` *(Base de données MLFlow)*

Pour visualiser les expériences MLFlow, lancez la commande suivante :
```bash
mlflow ui --backend-store-uri mlruns/
```

Puis, ouvrez le navigateur à l'adresse :
**http://127.0.0.1:5000**

### **Rapport Evidently sur le Data Drift**
Un rapport Evidently est généré pour analyser le **data drift**, disponible à l'emplacement suivant :
`evidently/report/`
`Farizon_David_4_Tableau_HTML_data_drift_evidently_012025.html`

Ce rapport permet de surveiller et d'évaluer les variations des distributions de données entre les différentes périodes d'entraînement et de prédiction.

---

## 4. **Installation et Exécution**

### **4.1 Prérequis**
Avant de démarrer l’API et Streamlit en local, assurez-vous d’avoir installé les dépendances nécessaires :
```bash
pip install -r requirements.txt
```

### **4.2 Lancement de l’API en local**
Si l’API est déjà en cours d’exécution, il faut d’abord la stopper avant de la relancer :
```bash
taskkill /F /IM python.exe
```

Ensuite, exécutez les commandes suivantes pour démarrer l'API :
```bash
cd api
python app.py
```

Elle s’exécutera sur `http://127.0.0.1:5000/`.

**Note** : Cette étape n'est pas nécessaire pour la version déployée sur le web.

### **4.3 Lancement de l'Interface Streamlit en local**
Après avoir démarré l’API, lancez Streamlit avec la commande suivante :
```bash
cd streamlit_app
streamlit run streamlit_app.py
```

Par défaut, Streamlit sera accessible à l'adresse suivante :
- **Sur la machine locale** : `http://localhost:8501`
- **Sur le réseau local (si configuré)** : `http://<IP_LOCAL>:8501`

---

## 5. **Déploiement Automatique sur le Cloud**
L’API est déployée sur **Azure** et accessible à l'adresse suivante :
**https://prediction-api.azurewebsites.net**

**Déploiement automatique**
Ce projet utilise le centre de déploiement Azure, assurant un **déploiement continu**.
Le fichier `.github/workflows/api-deployment_prediction.yml` est issu du déploiement automatique généré par Azure.
Une étape supplémentaire a été ajoutée dans la section `build` pour exécuter les **tests unitaires** avant le déploiement.

**Durée du processus** : Le déploiement peut prendre jusqu'à **30 minutes**.

---

## 6. **Endpoints de l’API**

### **6.1 Endpoint de prédiction (`/predict`)**
- **URL en local** : `http://127.0.0.1:5000/predict`
- **URL après déploiement** : `https://prediction-api.azurewebsites.net/predict`
- **Méthode** : `POST`
- **Description** : Prend en entrée un dictionnaire JSON contenant les caractéristiques d'un client et retourne la prédiction du modèle.

#### **Exemple de requête**
```json
{
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
```

#### **Exemple de réponse**
```json
{
    "lower_bound": 0.47000000000000003,
    "margin": 0,
    "optimal_threshold": 0.47000000000000003,
    "prediction": "Classe_0 (fiable)",
    "probability_class_0": 0.7965365906433127,
    "probability_class_1": 0.20346340935668727,
    "upper_bound": 0.47000000000000003
}
```

### **6.2 Endpoint des valeurs SHAP (`/shap_values`)**
- **URL en local** : `http://127.0.0.1:5000/shap_values`
- **URL après déploiement** : `https://prediction-api.azurewebsites.net/shap_values`
- **Méthode** : `GET`
- **Description** : Retourne les valeurs SHAP pour expliquer la décision du modèle sur un client donné.

#### **Exemple de réponse**
```json
{
    "base_values": -0.7438573109528505,
    "features_names": [
        "AMT_ANNUITY",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "ANNUITY_INCOME_PERCENT",
        "CREDIT_GOODS_RATIO",
        "CREDIT_TERM",
        "DAYS_BIRTH",
        "DAYS_ID_PUBLISH",
        "DAYS_REGISTRATION",
        "DEBT_CREDIT_RATIO",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "INSTA_AMT_PAYMENT",
        "INSTA_NUM_INSTALMENT_VERSION",
        "POS_CNT_INSTALMENT_FUTURE",
        "PREV_CNT_PAYMENT"
    ],
    "sample_values": [
        [
            0.8909397409758836,
            0.08973204731496785,
            0.32072747706903165,
            0.4984264892130457,
            -0.20075803566475942,
            1.070873783523731,
            0.9541051485718979,
            -0.8382782531774322,
            1.5035750809363868,
            1.187847533926258,
            -1.4975092984482719,
            1.3295106903291087,
            -1.3488443099480953,
            0.0521144846901881,
            -0.8374556177651016,
            0.08371349049402538,
            -1.2720999017458583
        ]
    ],
    "shap_values": [
        [
            -0.010641224200687333,
            -0.0008106912275832734,
            0.002393538705481381,
            0.004372882501110571,
            -0.04106824406961996,
            -0.008260239906941412,
            0.035637388022227594,
            0.001301685287086701,
            0.0002388982934175701,
            0.09724037024202416,
            0.19869268928905112,
            -0.32390777184358877,
            0.6972116554472935,
            0.01674885687250339,
            0.015972481375106144,
            -0.0035605964245221527,
            -0.004908864388482032
        ]
    ]
}
```

---

## 7. **Fichier OpenAPI pour tests sur Postman**
Le fichier `openapi.yml` situé dans le dossier `api/` permet de tester facilement les endpoints via **Postman** ou **Swagger UI**.

---

## 8. **Tests de l'API**
Les **tests unitaires** sont exécutés automatiquement lors du **déploiement sur Azure**.
Ils permettent de s'assurer que les endpoints `/predict` et `/shap_values` fonctionnent correctement.

Pour tester l'API à la fois en local et après déploiement, vous pouvez utiliser le notebook :
📂 `api/Farizon_David_5_notebook_test_API_0125.ipynb`

---

## 9. **Structure des fichiers**
```bash
📂 Credit_Dashboard
│── 📂 api
│   │── app.py                # Script principal de l'API Flask
│   │── openapi.yml            # Spécification OpenAPI pour tester l'API via Postman
│── 📂 models
│   │── lgbm_final_model.pkl  # Modèle final utilisé par l'API
│── 📂 models                # Dossier MLFlow
│── 📂 features
│   │── app_test_features.csv # Données de test (features)
│── 📂 streamlit
│   │── streamlit_app.py      # Interface utilisateur Streamlit
│── .venv                     # Environement Python
│── .gitignore
│── requirements.txt          # Dépendances utilisées dans le .venv
│── requirements-prod.txt     # Dépendances utilisées pour le déploiement de l'API
│── README.md
│── 📂 .github
│   │── workflows
│   │   │── api-deployment_prediction.yml  # CI/CD utilisé pour le déploiement de l'API
```

**Bonne utilisation de l’API et de l’interface Streamlit !**
