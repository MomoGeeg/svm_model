"""
Application de prédiction d'un vahicule 
"""
import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# configuration de la page
st.set_page_config(
  page_title=" Prédiction du type de véhicue",
  page_icon="📱",
  layout="wide"
)
DESCRIPTION = (
  "Ce modèle de machine learning permet de prédir le type de véhicule (Venant ou Occasion) en fonction : "
  "de la marque, du prix, de l'année, le quartier et de la transmission"
)

# Chargement des artéfacts

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (adresse, marque)
    uniques = jb.load("uniques.joblib")     # valeurs uniques
    scaler = jb.load("scaler.joblib")       # normaliseur
    svm = jb.load("svm_model.joblib")        # modèle
    return encoders, uniques, scaler, svm


encoders, uniques, scaler, svm = load_artifacts()

# nom des classe de la target
target_class = uniques[3]

# fonction de prédiction
def pred_func(marque, annee, transmission, quartier, prix):
  quartier = quartier.strip().title()
  #encodage des nouvelles données
  # encode transmission : LabelEncoding
  transmission_encoded = encoders['Transmission'].transform([transmission])[0]

  # Encoder Marque (OneHotEncoder séparé)
  ohe_marque = encoders['Marque']
  marque_encoded = ohe_marque.transform([[marque]])
  colonnes_marque = ohe_marque.get_feature_names_out(['Marque'])
  df_marque = pd.DataFrame(marque_encoded, columns=colonnes_marque)

  # Encoder Quartier (OneHotEncoder séparé)
  ohe_quartier = encoders['Quartier']
  quartier_encoded = ohe_quartier.transform([[quartier]])
  colonnes_quartier = ohe_quartier.get_feature_names_out(['Quartier'])
  df_quartier = pd.DataFrame(quartier_encoded, columns=colonnes_quartier)

  #Normaliser Prix et Année
  df_continu = pd.DataFrame([[annee, prix]], columns=['Année', 'Prix'])
  df_continu_scaled = pd.DataFrame(scaler.transform(df_continu), columns=['Année', 'Prix'])

  # Transmission dans un DataFrame
  df_transmission = pd.DataFrame([[transmission_encoded]], columns=['Transmission'])

  # Assembler toutes les colonnes
  X_input = pd.concat([df_continu_scaled, df_transmission, df_marque, df_quartier], axis=1)

  # Forcer exactement l'ordre et les colonnes du modèle entraîné
  colonnes_modele = ['Année', 'Transmission', 'Prix'] + colonnes_marque.tolist() + colonnes_quartier.tolist()
  X_input = X_input[colonnes_modele]

  # Prédiction
  prediction = svm.predict(X_input)[0]
  classe_predite = target_class[prediction]

  return classe_predite

import os
# Fonction de prediction multiple
def pred_func_csv(file):
    df = pd.read_csv(file)
    prediction = []
    erreurs = []

    for i, row in enumerate(df.iloc[:, :].values):
        try:
            y_pred = pred_func(row[0], row[1], row[2], row[3], row[4])
            prediction.append(y_pred)
        except Exception as e:
            erreurs.append(f"Ligne {i} ({row}) → {e}")
            prediction.append(None)

    df['Etat'] = prediction

    if erreurs:
        st.warning("Certaines lignes n'ont pas pu être prédites :\n" + "\n".join(erreurs))

    return df


# Interface

st.title("📱 Prédiction de l'état d'un véhicule vendu à Dakar")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'état d'un vahicule vendu à Dakar avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, cal2 = st.columns([1, 2])
        with col1:
            marque = st.selectbox("Marque", options=list(uniques[0]))
            annee = st.number_input("Année")
            transmission = st.selectbox("Transmission", options=list(uniques[1]))
            quartier = st.selectbox("Quartier", options=list(uniques[2]))
            prix = st.number_input("Prix", value=0.0, step=0.1, format="%.1f")

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = pred_func(marque, annee, transmission, quartier, prix)
            st.success(f"**Etat du Véhicule  :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire l'état du véhicule avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "marque, année, transmission, quartier, prix, etat"
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")