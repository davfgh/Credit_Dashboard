import os
import pickle
import pandas as pd

def test_model_and_data_loading():
    """
    Vérifie que les fichiers essentiels (modèle et données) sont bien présents et exploitables.

    :return: Aucun – génère une erreur si un fichier est manquant ou inutilisable.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model_path = os.path.join(base_dir, "models", "lgbm_final_model.pkl")
    features_path = os.path.join(base_dir, "features", "app_test_features.csv")
    train_features_path = os.path.join(base_dir, "features", "app_train_features.csv")

    # Vérification des chemins
    assert os.path.exists(model_path), f"Le fichier modèle est introuvable à {model_path}"
    assert os.path.exists(features_path), f"Le fichier app_test_features.csv est introuvable à {features_path}"
    assert os.path.exists(train_features_path), f"Le fichier app_train_features.csv est introuvable à {train_features_path}"

    # Chargement du modèle
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    assert "model" in model_data
    assert "features" in model_data
    assert "optimal_threshold" in model_data

    # Chargement des données
    df_test = pd.read_csv(features_path)
    df_train = pd.read_csv(train_features_path)

    assert not df_test.empty, "Le jeu de test est vide"
    assert not df_train.empty, "Le jeu d'entraînement est vide"
