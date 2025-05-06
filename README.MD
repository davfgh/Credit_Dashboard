# Documentation de l'Interface Streamlit pour le Scoring Client
Cette web application permet d'effectuer des **prédictions sur la fiabilité d'un client** en fonction de ses caractéristiques financières.

Elle repose sur **deux composants principaux** :
- **Une API**, exposant un modèle de machine learning, permettant d'obtenir un score de risque et des valeurs explicatives via SHAP.
- **Une interface utilisateur Streamlit**, permettant d’explorer les prédictions et d’interagir avec le modèle de manière intuitive.

🌐 **Dashboard déployé** :  
https://scoring-dashboard-a7ehg6bkghgvffdf.francecentral-01.azurewebsites.net

---

## Lancement en local

> **Note** : Le démarrage de l’API n’est nécessaire **qu’en local**. En production (version déployée), elle est déjà disponible.

1. **Démarrer l’API Flask**  
   ```bash
   cd api
   python app.py
  ```

2. **Lancer l’interface Streamlit**  
   ```bash
  cd streamlit_app
  streamlit run dashboard_032025.py
  ```

**Bonne utilisation du dashboard !**
