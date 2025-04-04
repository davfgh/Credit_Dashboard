import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import requests
import shap
import pickle
import os
import warnings

# 🔧 Configuration des logs
logging.basicConfig(level=logging.DEBUG, format="DEBUG:%(message)s")
warnings.simplefilter("always")  # Activer tous les warnings

# 📂 Définition des chemins
base_dir = "D:/Pro/OpenClassrooms/Projet_7/3_dossier_code_012025"
model_path = os.path.join(base_dir, "models", "lgbm_final_model.pkl")
features_path = os.path.join(base_dir, "features", "app_test_features.csv")
train_features_path = os.path.join(base_dir, "features", "app_train_features.csv")


# 🎯 Initialisation des états de session
if "selected_client" not in st.session_state:
    st.session_state.selected_client = None
if "margin" not in st.session_state:
    st.session_state.margin = 0.00
if "input_data" not in st.session_state:
    st.session_state.input_data = {feat: 0.0 for feat in range(100)}  # Temp init
if "previous_input_data" not in st.session_state:
    st.session_state.previous_input_data = None
if "mode" not in st.session_state:
    st.session_state.mode = "manuel"
if "group_filter" not in st.session_state:
    st.session_state.group_filter = None

@st.cache_data(show_spinner=False)
def build_comparison_table(dataframe, client_data, filter_column=None, filter_value=None):
    """
    Construit un tableau de comparaison entre les données du client et un sous-ensemble de clients.

    :param dataframe: DataFrame contenant les données complètes
    :param client_data: Dictionnaire des valeurs du client à comparer
    :param filter_column: Colonne sur laquelle appliquer un filtre (ex: 'TARGET')
    :param filter_value: Valeur à filtrer (ex: 0 ou 1)
    :return: DataFrame formaté avec comparaisons statistiques
    """
    if filter_column and filter_column in dataframe.columns:
        subset = dataframe[dataframe[filter_column] == filter_value]
    else:
        subset = dataframe
    subset = subset.dropna()
    mean_std = subset[client_data.keys()].agg(["mean", "std"])
    rows_list = []
    for feature in client_data:
        val_client = client_data[feature]
        mean_val = mean_std.loc["mean", feature]
        std_val = mean_std.loc["std", feature]
        lower_bound = mean_val - std_val
        upper_bound = mean_val + std_val

        # Comparaison à la moyenne
        if val_client < mean_val:
            pos_moy = "<"
        elif val_client > mean_val:
            pos_moy = ">"
        else:
            pos_moy = "≈"  # approximativement égal

        # Intervalle
        if lower_bound <= val_client <= upper_bound:
            intervalle = "✅"
        else:
            intervalle = "❌"

        rows_list.append({
            "Feature": feature,
            "Valeur Client": val_client,
            "Moyenne": mean_val,
            "Écart-Type": std_val,
            "Δ Moyenne": pos_moy,
            "Intervalle": intervalle
        })
    return pd.DataFrame(rows_list)

def display_group_details(filter_type, client_value, delta, subset_size=None, dataset_label="", target_value=None):
    """
    Affiche les détails du groupe de clients utilisé pour la comparaison dans l'interface Streamlit.

    :param filter_type: Type de filtre appliqué (ex: 'DAYS_BIRTH' ou 'AMT_CREDIT')
    :param client_value: Valeur de la variable du client pour le filtre (ex: âge ou crédit)
    :param delta: Marge de variation autour de la valeur du client
    :param subset_size: Taille du groupe de comparaison
    :param dataset_label: Libellé du jeu de données utilisé (ex: 'de test')
    :param target_value: Classe cible prédite (0 ou 1)
    :return: Aucun (affiche directement dans l'application Streamlit)
    """
    with st.expander("ℹ️ Détails du groupe de comparaison"):
        if filter_type == "DAYS_BIRTH":
            birth_min = int((client_value - delta) / -365.25)
            birth_max = int((client_value + delta) / -365.25)
            st.markdown(f"📆 **Tranche d'âge sélectionnée** : {birth_min} - {birth_max} ans")
        elif filter_type == "AMT_CREDIT":
            credit_min = round(client_value - delta)
            credit_max = round(client_value + delta)
            st.markdown(f"💸 **Tranche de crédit sélectionnée** : {credit_min:,}".replace(",", " ") + f" € - {credit_max:,}".replace(",", " ") + " €")
        elif filter_type is None and target_value is None:
            st.markdown(f"📊 **Aucun filtre appliqué** : l'ensemble des clients du jeu de données {dataset_label} est utilisé pour la comparaison.")

        if target_value is not None:
            label = "Classe_0 (Fiable)" if target_value == 0 else "Classe_1 (Risqué)"
            st.markdown(f"🔹 **Filtre cible appliqué** : {label}")

        if subset_size is not None:
            st.markdown(f"👥 **Nombre de clients dans le groupe comparé** : {subset_size:,}".replace(",", " "))


def display_explanation():
    """
    Affiche une infobulle expliquant les colonnes du tableau de comparaison des variables client.

    :return: Aucun (affiche directement une explication dans l'interface utilisateur Streamlit)
    """
    with st.expander("ℹ️ Explication des colonnes du tableau"):
        st.markdown("""
        - **Δ Moyenne** : comparaison directe entre la valeur du client et la moyenne :
            - `>` : supérieur à la moyenne
            - `<` : inférieur à la moyenne
            - `≈` : environ égal à la moyenne
        - **Intervalle** :
            - ✅ : la valeur est dans l'intervalle moyenne ± écart-type
            - ❌ : la valeur est hors de cet intervalle
        """)

st.set_page_config(page_title="Credit Dashboard", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #1F2937;'>📊 Credit Dashboard - Analyse & Décision Client</h1>",
    unsafe_allow_html=True
)

# 📌 1. Chargement du Modèle et des Données
st.header("📌 1. Chargement")
try:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    features_names = model_data["features"]
    optimal_threshold = model_data["optimal_threshold"]
    st.session_state.input_data = {feat: 0.0 for feat in features_names}
    st.success("✅ Modèle chargé avec succès !")

    try:
        data = pd.read_csv(features_path)
        train_data = pd.read_csv(train_features_path)
        st.session_state.train_data_loaded = True
        st.success("✅ Données chargées avec succès !")
    except Exception as e:
        train_data = pd.DataFrame()
        st.warning(f"⚠️ Impossible de charger les données. {e}")

    st.write(f"🔹 **Nombre total de clients dans le dataset d'entraînement** : {train_data.shape[0]:,}".replace(",", " "))
    st.write(f"🔹 **Nombre total de clients dans le dataset de test** : {data.shape[0]:,}".replace(",", " "))
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des fichiers : {e}")

# 📌 2. Sélection du Client Aléatoire et Comparaison
st.header("📌 2. Sélection d'un client")

with st.expander("ℹ️ Modes de sélection disponibles"):
    st.markdown("""
    Trois méthodes de sélection de client sont disponibles :

    - 🔍 **Saisie d'un ID client** : entrez manuellement un identifiant pour charger un client spécifique.
    - 🎲 **Client aléatoire** : sélection automatique d'un client parmi les données disponibles.
    - 📝 **Saisie manuelle** : remplissage des variables sans données pré-remplies.
    """)

if st.session_state.get("mode") == "auto":
    with st.expander("ℹ️ Informations sur le mode automatique"):
        st.markdown("""
        - 📌 **Client sélectionné automatiquement** : les champs sont affichés en lecture seule.
        - 🧹 Cliquez sur le bouton de droite pour passer en **saisie manuelle**.
        - 🎲 Vous pouvez aussi sélectionner un autre client aléatoirement ou utiliser le champ ID pour une recherche ciblée.
        """)

# col1, col2 = st.columns([1, 1])
col_id, col_auto, col_manual = st.columns([1, 1, 1], vertical_alignment="bottom")

# 🎲 Pré-remplir avec un client aléatoire

with col_id:
    client_id_input = st.text_input("🔍 Rechercher par ID client", value="", placeholder="Entrez l'ID du client ici")
    if client_id_input:
        try:
            client_id_input = int(client_id_input)
            if client_id_input in data.index:
                st.session_state.selected_client = data.loc[[client_id_input]]
                st.session_state.mode = "auto"
                st.session_state.input_data = {
                    feat: float(st.session_state.selected_client[feat].values[0]) for feat in features_names
                }
                st.session_state.previous_input_data = st.session_state.input_data.copy()
                if "shap_values_data" in st.session_state:
                    del st.session_state.shap_values_data
                st.success(f"✅ Client ID {client_id_input} sélectionné.")
            else:
                st.warning("⚠️ ID client non trouvé dans le jeu de données.")
        except ValueError:
            st.warning("⚠️ Veuillez entrer un ID valide (nombre entier).")

with col_auto:
    if st.button("🎲 Selection d'un client aléatoire", key="btn_auto"):
        data_clean = data.dropna()
        st.session_state.selected_client = data_clean.sample(1, random_state=np.random.randint(1000))
        st.session_state.mode = "auto"
        st.session_state.input_data = {
            feat: float(st.session_state.selected_client[feat].values[0]) for feat in features_names
        }
        st.session_state.previous_input_data = st.session_state.input_data.copy()
        if "shap_values_data" in st.session_state:
            del st.session_state.shap_values_data

# Replir manuellement le formumaire
with col_manual:
    if st.button("🧹 Réinitialiser le formulaire (passer en mode manuel)", key="btn_manual"):
        st.session_state.mode = "manuel"
        st.session_state.input_data = {feat: 0.0 for feat in features_names}
        st.session_state.selected_client = None
        if "shap_values_data" in st.session_state:
            del st.session_state.shap_values_data

# 💡 Custom CSS
st.markdown("""
    <style>
    button[kind="secondary"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 0.5rem;
        margin-bottom: 1em;
    }
    button[aria-label="🧹 Réinitialiser le formulaire (manuel)"] {
        background-color: #F97316 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

try:
    # 🔎 Filtrer les clients sans valeurs manquantes
    data_clean = data.dropna()

    if data_clean.empty:
        st.warning("⚠️ Aucun client sans valeurs manquantes trouvé.")
    else:
        if "selected_client" not in st.session_state:
            st.session_state.selected_client = None

        if st.session_state.selected_client is None:
            st.session_state.selected_client = data_clean.sample(1, random_state=np.random.randint(1000))

        random_client = st.session_state.selected_client
        client_id = random_client.index[0]

        # ✅ Affichage fusionné du mode actif
        if st.session_state.get("mode") == "manuel":
            st.markdown(
                "<div style='background-color:#2C3E50; padding:10px; border-radius:8px; font-size:1.1em;'>"
                "✍️ <strong>Mode actif : Saisie manuelle</strong>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            client_id_display = st.session_state.selected_client.index[0] if st.session_state.selected_client is not None else "?"
            st.markdown(
                f"<div style='background-color:#2C3E50; padding:10px; border-radius:8px; font-size:1.1em;'>"
                f"👤 <strong>Mode actif : Client sélectionné (ID : {client_id_display})</strong>"
                "</div>",
                unsafe_allow_html=True
            )

        # 📋 Modifier ou saisir les données du client
        st.header("📋 Saisie / modification des données du client")

        # Initialisation des données si non présentes
        if "input_data" not in st.session_state:
            st.session_state.input_data = {feat: 0.0 for feat in features_names}

        if st.session_state.mode == "manuel":
            for feat in features_names:
                st.session_state.input_data[feat] = st.number_input(
                    f"{feat}", value=st.session_state.input_data.get(feat, 0.0), key=feat
                )
        else:
            with st.expander("ℹ️ Client sélectionné : les champs ne sont pas modifiables."):
                st.markdown("""
                - 🧹 Cliquez sur le bouton de droite pour passer en **saisie manuelle**.
                - 🎲 Vous pouvez aussi sélectionner un autre client aléatoirement ou utiliser le champ ID pour une recherche ciblée.
                """)


        # 📊 **Comparaison aux clients semblables
        st.subheader("📊 Comparaison avec les groupes de clients")

        # Selectbox
        group_filter_selection = st.selectbox(
            "Choisissez une variable de regroupement :",
            [None, "DAYS_BIRTH", "AMT_CREDIT"],
            index=[None, "DAYS_BIRTH", "AMT_CREDIT"].index(st.session_state.group_filter),
            format_func=lambda x: "Aucun" if x is None else x,
            key="group_filter_selector"
        )
        st.session_state.group_filter = group_filter_selection

        if group_filter_selection == "DAYS_BIRTH":
            st.markdown("🧓 Cette option permet de comparer un client à des clients de la **même tranche d'âge** (en années).")
        elif group_filter_selection == "AMT_CREDIT":
            st.markdown("💰 Cette option compare un client à d'autres ayant des **montants de crédit similaires**.")

        # ✅ 🎯 Filtrage dynamique selon la sélection
        subset_data = data_clean.copy()  # par défaut : tous les clients

        if group_filter_selection == "DAYS_BIRTH":
            client_birth = st.session_state.input_data.get("DAYS_BIRTH", None)
            if client_birth is not None:
                delta = 1825  # plage de +/- 1825 jours (+ ou - 5 ans)
                subset_data = subset_data[
                    (subset_data["DAYS_BIRTH"] >= client_birth - delta) &
                    (subset_data["DAYS_BIRTH"] <= client_birth + delta)
                ]

        elif group_filter_selection == "AMT_CREDIT":
            client_credit = st.session_state.input_data.get("AMT_CREDIT", None)
            if client_credit is not None:
                delta = 50000  # plage de +/- 50k
                subset_data = subset_data[
                    (subset_data["AMT_CREDIT"] >= client_credit - delta) &
                    (subset_data["AMT_CREDIT"] <= client_credit + delta)
                ]

        if group_filter_selection == "DAYS_BIRTH":
            client_value = st.session_state.input_data.get("DAYS_BIRTH", None)
            delta = 1825
        elif group_filter_selection == "AMT_CREDIT":
            client_value = st.session_state.input_data.get("AMT_CREDIT", None)
            delta = 50000
        else:
            client_value = None
            delta = None

        if client_value is not None and delta is not None:
            display_group_details(
                filter_type=group_filter_selection,
                client_value=client_value,
                delta=delta,
                subset_size=subset_data.shape[0]
            )
        else:
            display_group_details(
                filter_type=None,
                client_value=None,
                delta=None,
                subset_size=subset_data.shape[0],
                dataset_label="de test"
            )

        # ✅ Appel à la fonction sans filtre (comparaison sur tous les clients)
        df_no_target = build_comparison_table(subset_data, st.session_state.input_data)
        st.dataframe(
            df_no_target.style.format({"Valeur Client": "{:.2f}", "Moyenne": "{:.2f}", "Écart-Type": "{:.2f}"}),
            use_container_width=True
        )
        display_explanation()

        # 🔍 Visualisation bivariée : Âge vs Montant du crédit (hexbin)
        st.subheader("📉 Analyse bivariée : Âge vs Montant du crédit")

        try:
            # Préparation des données
            data_age_credit = train_data[["DAYS_BIRTH", "AMT_CREDIT"]].copy()
            data_age_credit["AGE"] = (-data_age_credit["DAYS_BIRTH"] / 365.25)

            client_age = -st.session_state.input_data["DAYS_BIRTH"] / 365.25
            client_credit = st.session_state.input_data["AMT_CREDIT"]

            fig, ax = plt.subplots(figsize=(8, 6))
            hb = ax.hexbin(
                data_age_credit["AGE"], data_age_credit["AMT_CREDIT"],
                gridsize=50, cmap="Blues", mincnt=1, linewidths=0.2, edgecolors='grey'
            )

            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Densité de clients")

            # Client actuel en surbrillance
            ax.scatter(
                client_age,
                client_credit,
                color="darkred",
                s=80,
                marker="o",
                label="Client sélectionné",
                zorder=10
            )

            ax.set_xlabel("Âge du client (années)")
            ax.set_ylabel("Montant du crédit (€)")
            ax.set_title("Hexbin - Répartition des clients")
            ax.legend()

            st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ Impossible d'afficher la visualisation hexbin : {e}")

        st.markdown("""
            🔍 **Figure : Analyse densité - Âge vs Montant du crédit**

            - **⬢ Zones hexagonales** : regroupement de clients selon densité.
            - **🔴 Rond foncé** : client actuellement sélectionné.
            """)

        st.markdown("ℹ️ Cette figure représente la densité de clients selon leur âge et leur montant de crédit. Le point rouge représente le client sélectionné.")


except Exception as e:
    st.error(f"❌ Erreur lors de la sélection du client : {e}")

# 📌 3. Prédiction et Réglage de la Zone Grise
st.header("📌 3. Prédiction")

try:
    # 📌 Préparation des données pour la prédiction
    # input_data = random_client[features_names].to_dict(orient='records')[0]
    input_data = st.session_state.input_data

    # 🔗 URL de l'API (endpoint predict)
    # api_url = "http://127.0.0.1:5000/predict"
    api_url= "https://prediction-api.azurewebsites.net/predict"

    # 🚀 Debugging avant l'appel API
    print(f"🔍 Vérification - Envoi de la requête API avec les données suivantes : {input_data}")
    print(f"🔍 API URL: {api_url}")

    # 🚀 Envoi de la requête à l'API
    response = requests.post(api_url, json=input_data)

    if response.status_code == 200:
        prediction = response.json()
        st.session_state.prediction = prediction

        if "prediction" in st.session_state and st.session_state.prediction.get("probability_class_1") is not None:

            prediction = st.session_state.prediction
            probability_class_1 = prediction["probability_class_1"]
            score_pct = int(probability_class_1 * 100)

            # Définir la couleur selon le score
            if score_pct < 40:
                color = "green"
            elif score_pct < 70:
                color = "gold"
            else:
                color = "red"

            # 📊 Affichage clair du score
            st.markdown("### 🎯 Score de probabilité (risque de défaut)")

            with st.expander("ℹ️ Qu'est-ce que ce score ?"):
                st.markdown("""
                🔍 Cette probabilité représente le **risque que le client ne rembourse pas son crédit**.
                Elle est calculée par un modèle de machine learning entraîné sur des données historiques.
                Plus ce score est élevé, plus le client est considéré comme risqué.
                """)

            st.markdown(
                f"""
                <div style="background-color:#1F2937; padding:15px; border-radius:8px; color:white; font-size:1em; margin-bottom: 10px;">
                    📊 <strong>Score actuel</strong> : {score_pct}% de probabilité d'être risqué
                    <div style="background-color:#e0e0e0; border-radius:6px; overflow:hidden; margin-top:8px;">
                        <div style="width:{score_pct}%; background-color:{color}; padding:6px 0;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"🔹 **Seuil optimal** : {optimal_threshold:.3f}")

            diff_pct = (probability_class_1 - optimal_threshold) * 100
            position = "au-dessus" if diff_pct > 0 else "en-dessous" if diff_pct < 0 else "égal au"
            diff_formatted = f"{diff_pct:+.1f}%"

            st.markdown(
                f"""
                <div style='background-color: #1F2937; padding: 8px 12px; border-radius: 8px; color: white; font-size: 0.95em;'>
                📏 <strong>Écart </strong> : {diff_formatted} → Le client est <strong>{position}</strong> du seuil optimal.
                </div>
                """,
                unsafe_allow_html=True
            )

            # ℹ️ Infobulle sur le seuil optimal
            with st.expander("ℹ️ **Explication**"):
                st.write(
                    "🔹 Le **seuil optimal** est la probabilité à partir de laquelle un client est "
                    "considéré comme **risqué**. Il a été optimisé selon des **critères métier**.\n\n"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # 📌 Verdict simple
            verdict = "Classe_1 (Risqué)" if probability_class_1 >= optimal_threshold else "Classe_0 (Fiable)"
            verdict_color = "#FFCCCB" if verdict == "Classe_1 (Risqué)" else "lightgreen"

            st.markdown(
                f'<div style="background-color: {verdict_color}; padding: 15px; border-radius: 10px;">'
                f'<h3 style="text-align: center; color: black;">🔮 {verdict}</h3>'
                '</div>',
                unsafe_allow_html=True
            )

            with st.expander("ℹ️ Explication de la décision du modèle"):
                st.markdown("""
                - **Classe_0 (Fiable)** : Le modèle estime que le client est capable de rembourser son crédit.
                - **Classe_1 (Risqué)** : Le modèle estime que le client présente un risque élevé de non-remboursement.
                Ces résultats ne sont pas définitifs mais basés sur les données disponibles et les critères appris.
                """)

        else:
            st.warning("⚠️ Aucune prédiction disponible. Veuillez sélectionner un client pour lancer la prédiction.")

        # 📊 **Comparaison aux clients semblables
        st.subheader("📊 Comparaison avec les clients de la même classe prédite")
        predicted_class = 1 if probability_class_1 > optimal_threshold else 0

        if "train_data_loaded" in st.session_state and st.session_state.train_data_loaded:
            subset_by_target = train_data[train_data["TARGET"] == predicted_class]
            df_filtered = build_comparison_table(
                subset_by_target,
                st.session_state.input_data,
                filter_column=None  # Déjà filtré
            )
            display_group_details(
                filter_type=None,
                client_value=None,
                delta=None,
                subset_size=subset_by_target.shape[0],  # ✅ Nombre de clients correct
                dataset_label="d'entraînement",
                target_value=predicted_class
            )
            st.dataframe(
                df_filtered.style.format({"Valeur Client": "{:.3f}", "Moyenne": "{:.3f}", "Écart-Type": "{:.3f}"}),
                use_container_width=True
            )
            display_explanation()
        else:
            st.info("ℹ️ Comparaison avec les clients de la même classe non disponible : données d'entraînement manquantes.")


except Exception as e:
    st.error(f"❌ Erreur lors de la requête à l'API : {e}")

# 📌 4. Feature Importance Locale (SHAP)
st.header("📌 4. Feature Importance Locale")

st.markdown(
    "ℹ️ **Pourquoi cette analyse ?**\n"
    "Cette section montre **les principales variables qui influencent** la prédiction du modèle **pour ce client spécifique**."
)

# ℹ️ Infobulle sur le SHAP Waterfall Plot
with st.expander("ℹ️ **Comment lire ce graphique ?**"):
    st.write(
        "- 🟥 **Facteurs augmentant la probabilité d'être risqué** : Ces features poussent la prédiction vers un risque élevé.\n"
        "- 🟦 **Facteurs réduisant le risque** : Ces features diminuent la probabilité que le client soit risqué."
    )

# 📌 Endpoint de l'API pour récupérer les SHAP values
# api_shap_url = "http://127.0.0.1:5000/shap_values"
api_shap_url = "https://prediction-api.azurewebsites.net/shap_values"

# 📌 Vérification et récupération des données SHAP avec mise en cache
if "shap_values_data" not in st.session_state:
    try:
        response = requests.get(api_shap_url)

        if response.status_code == 200:
            st.session_state.shap_values_data = response.json()
        else:
            st.error(f"❌ Erreur API SHAP : {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion à l'API SHAP : {e}")

# 📌 Utilisation des données SHAP en cache si disponibles
if "shap_values_data" in st.session_state:
    shap_data = st.session_state.shap_values_data

    # 🔍 Extraction des données de l'API
    shap_values = np.array(shap_data["shap_values"]).reshape(1, -1)  # Assurer (1, N)
    feature_names = shap_data["features_names"]
    sample_values = np.array(shap_data["sample_values"]).reshape(1, -1)  # Même format (1, N)
    base_values = shap_data["base_values"]

    # 📌 Vérification des dimensions après correction
    print(f"📌 SHAP values shape : {shap_values.shape}")
    print(f"📌 Feature names count : {len(feature_names)}")
    print(f"📌 Sample values shape : {sample_values.shape}")

    # 📌 Création d'un objet SHAP Explanation pour afficher la figure waterfall
    explainer = shap.Explanation(
        values=shap_values[0],
        base_values=base_values,
        data=sample_values[0],  # Correspondance avec les features
        feature_names=feature_names
    )

    # 📊 Génération et affichage du Waterfall Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explainer, max_display=11, show=False)
    plt.title(f"Impact des principales features sur la prédiction")
    st.pyplot(fig)

    st.markdown("🔍 **Figure : SHAP Waterfall Plot des principales features**")

else:
    st.error("❌ Les données SHAP n'ont pas pu être récupérées.")
