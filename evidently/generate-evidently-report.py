import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 📂 Définition des chemins mis à jour
base_dir = "D:/Pro/OpenClassrooms/Projet_7/3_dossier_code_012025"
features_train_path = os.path.join(base_dir, "features", "app_train_features.csv")
features_test_path = os.path.join(base_dir, "features", "app_test_features.csv")
report_dir = os.path.join(base_dir, "evidently", "report")
nom = os.getenv("NOM", "Nom")
prenom = os.getenv("PRENOM", "Prenom")
filename = f"{nom}_{prenom}_4_Tableau_HTML_data_drift_evidently_012025.html"
report_path = os.path.join(report_dir, filename)

# 📌 Vérification et création des dossiers si besoin
os.makedirs(report_dir, exist_ok=True)

# 📌 Chargement des données
print("📌 Chargement des données d'entraînement et de test...")
train_data = pd.read_csv(features_train_path)
test_data = pd.read_csv(features_test_path)

# 📌 Suppression des colonnes non pertinentes si besoin
id_columns = ["SK_ID_CURR"]  # Exclusion des identifiants
train_data = train_data.drop(columns=id_columns, errors='ignore')
test_data = test_data.drop(columns=id_columns, errors='ignore')

# 📌 Vérification des colonnes communes
common_columns = list(set(train_data.columns) & set(test_data.columns))
train_data = train_data[common_columns]
test_data = test_data[common_columns]

# 📊 Création du rapport Evidently
print("📌 Génération du rapport Evidently sur le Data Drift...")
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=train_data, current_data=test_data)

# 📥 Sauvegarde du rapport HTML
print(f"📁 Sauvegarde du rapport Evidently dans : {report_path}")
data_drift_report.save_html(report_path)

print("✅ Rapport Evidently généré avec succès !")

# 📂 Chemin du fichier HTML généré
html_report_path = report_path

# 📝 Lire le contenu du fichier HTML
with open(html_report_path, "r", encoding="utf-8") as file:
    html_content = file.read()

# 🔄 Ajout de (train) et (test) après "Reference Distribution" et "Current Distribution"
html_content = html_content.replace("Reference Distribution", "Reference Distribution (app. train features data set)")
html_content = html_content.replace("Current Distribution", "Current Distribution     (app. test features data set)")

# 💾 Sauvegarder les modifications
with open(html_report_path, "w", encoding="utf-8") as file:
    file.write(html_content)

print("✅ Modifications effectuées dans le rapport Evidently.")
