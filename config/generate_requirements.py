import os
import re
import subprocess

# Chemins
raw_requirements_file = "D://Pro//OpenClassrooms//Projet_7//3_dossier_code_012025/config/raw_requirements.txt"
cleaned_requirements_file = "D://Pro//OpenClassrooms//Projet_7//3_dossier_code_012025/requirements.txt"
pip_path = "D://Pro//OpenClassrooms//Projet_7//.venv//Scripts//pip.exe"

# Repo
os.makedirs(os.path.dirname(raw_requirements_file), exist_ok=True)

# Génération du fichier raw_requirements.txt
try:
    with open(raw_requirements_file, "w") as f:
        result = subprocess.run([pip_path, "freeze"], stdout=f, text=True, check=True)
    print(f"Fichier raw_requirements.txt généré à {raw_requirements_file}")
except subprocess.CalledProcessError as e:
    print(f"Erreur lors de l'exécution de pip freeze : {e}")
    exit(1)
except Exception as e:
    print(f"Une erreur inattendue s'est produite : {e}")
    exit(1)

# Nettoyage du fichier requirements
try:
    with open(raw_requirements_file, "r") as raw_f, open(cleaned_requirements_file, "w") as cleaned_f:
        for line in raw_f:
            # Nettoyage des lignes
            if "@ file://" in line:
                line = re.sub(r"@ file://.*", "", line)
            cleaned_f.write(line)
    print(f"Fichier requirements.txt nettoyé généré à {cleaned_requirements_file}")

    # Ajout manuel de gunicorn pour le déploiement sur Azure
    with open(cleaned_requirements_file, "a") as cleaned_f:
        cleaned_f.write("\ngunicorn==21.2.0\n")  # Ajoute Gunicorn pour le serveur
    print("✅ Gunicorn ajouté au requirements.txt")

except Exception as e:
    print(f"Une erreur inattendue s'est produite lors du nettoyage : {e}")
    exit(1)
