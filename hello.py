from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Arbre de Decision": "decision_tree.pkl",
    "Foret Aleatoire": "random_forest.pkl",
    "Regression Logistique": "logistic_regression.pkl",
    "Modele simple": "model_simple.pkl",
}

SCALER_FILES = {
    "Modele simple": "scaler_simple.pkl",
    "default": "scaler.pkl",
}

MODEL_METRICS = {
    "Arbre de Decision": {"Accuracy": "0.9352", "F1": "0.90", "AUC": "0.7944"},
    "Foret Aleatoire": {"Accuracy": "0.9414", "F1": "0.92", "AUC": "0.7434"},
    "Regression Logistique": {"Accuracy": "0.9383", "F1": "0.92", "AUC": "0.8446"},
    "Modele simple": {"Accuracy": "0.9321", "F1": "N/A", "AUC": "N/A"},
}

PROVINCES = [
    "Bubanza",
    "Bujumbura Rural",
    "Bururi",
    "Cankuzo",
    "Cibitoke",
    "Gitega",
    "Kayanza",
    "Kirundo",
    "Makamba",
    "Muramvya",
    "Muyinga",
    "Mwaro",
    "Ngozi",
    "Rutana",
    "Ruyigi",
]

CULTURES = ["Bananier", "Haricot", "Manioc", "Mais", "Patate douce", "Sorgho"]


def inject_custom_style():
    st.markdown(
        """
        <style>
            #MainMenu,
            footer,
            header {
                visibility: hidden;
            }

            .stApp {
                background: linear-gradient(135deg, #f5fbf7 0%, #eef7f9 52%, #fff8ed 100%);
                color: #17352a;
            }

            [data-testid="stSidebar"] {
                background: #12372a;
            }

            [data-testid="stSidebar"] * {
                color: #f7fff9 !important;
            }

            h1, h2, h3 {
                color: #164734;
            }

            .stCaptionContainer,
            [data-testid="stCaptionContainer"] {
                color: #4d6b60;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(25, 92, 68, 0.12);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 8px 24px rgba(20, 72, 54, 0.08);
            }

            .stButton > button {
                background: #1f7a4f;
                color: #ffffff;
                border: 1px solid #1f7a4f;
                border-radius: 8px;
                font-weight: 700;
            }

            .stButton > button:hover {
                background: #17613e;
                border-color: #17613e;
                color: #ffffff;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInput"] input {
                border-color: rgba(31, 122, 79, 0.28);
                border-radius: 8px;
            }

            div[role="radiogroup"] label {
                background: rgba(255, 255, 255, 0.65);
                border-radius: 8px;
                padding: 4px 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    models = {
        name: joblib.load(MODELS_DIR / file_name)
        for name, file_name in MODEL_FILES.items()
        if (MODELS_DIR / file_name).exists()
    }
    for model in models.values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    scalers = {
        "default": joblib.load(MODELS_DIR / SCALER_FILES["default"]),
        "Modele simple": joblib.load(MODELS_DIR / SCALER_FILES["Modele simple"]),
    }
    return models, scalers


def get_feature_names(model_name, model, scalers):
    scaler = scalers["Modele simple"] if model_name == "Modele simple" else scalers["default"]
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [f"feature_{index}" for index in range(model.n_features_in_)]


def make_input_frame(values, feature_names):
    row = {feature: 0.0 for feature in feature_names}

    direct_values = {
        "annee": values["annee"],
        "altitude_m": values["altitude_m"],
        "pluviometrie_mm": values["pluviometrie_mm"],
        "temperature_moy_C": values["temperature_moy_C"],
        "superficie_ha": values["superficie_ha"],
        "utilisation_engrais": values["utilisation_engrais"],
        "acces_irrigation": values["acces_irrigation"],
        "nb_menages": values["nb_menages"],
    }

    for key, value in direct_values.items():
        if key in row:
            row[key] = float(value)

    if values["saison"] == "B" and "saison_B" in row:
        row["saison_B"] = 1.0

    province_key = f"province_{values['province']}"
    if province_key in row:
        row[province_key] = 1.0

    culture_candidates = [
        f"culture_{values['culture']}",
        "culture_Maļs" if values["culture"] == "Mais" else None,
        "culture_Maïs" if values["culture"] == "Mais" else None,
    ]
    for culture_key in culture_candidates:
        if culture_key in row:
            row[culture_key] = 1.0
            break

    return pd.DataFrame([row], columns=feature_names)


def predict(model, scaler, input_frame):
    scaled_values = scaler.transform(input_frame)
    prediction = int(model.predict(scaled_values)[0])

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        positive_index = classes.index(1.0) if 1.0 in classes else classes.index(1)
        probability = float(model.predict_proba(scaled_values)[0][positive_index])
    else:
        probability = np.nan

    return prediction, probability


def render_importance(model_name, model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        title = "Importance des variables"
    elif hasattr(model, "coef_"):
        importance = model.coef_[0]
        title = "Coefficients de la regression logistique"
    else:
        return

    importance_df = pd.DataFrame(
        {"Variable": feature_names, "Importance": importance}
    ).sort_values("Importance", key=lambda values: values.abs(), ascending=False)

    st.subheader(title)
    st.bar_chart(importance_df.head(10).set_index("Variable"))


def main():
    st.set_page_config(
        page_title="Prediction des recoltes au Burundi",
        page_icon="AG",
        layout="wide",
    )
    inject_custom_style()

    st.title("Prediction des recoltes au Burundi")
    st.caption("Application Streamlit de prediction Bonne recolte / Mauvaise recolte.")

    models, scalers = load_artifacts()
    if not models:
        st.error("Aucun modele n'a ete trouve dans le dossier models.")
        st.stop()

    with st.sidebar:
        st.header("Modele")
        model_name = st.selectbox("Choisir le modele", list(models.keys()))
        st.divider()
        st.header("Metriques globales")
        metrics = MODEL_METRICS.get(model_name, {})
        metric_cols = st.columns(3)
        for col, metric_name in zip(metric_cols, ["Accuracy", "F1", "AUC"]):
            col.metric(metric_name, metrics.get(metric_name, "A renseigner"))

    model = models[model_name]
    scaler = scalers["Modele simple"] if model_name == "Modele simple" else scalers["default"]
    feature_names = get_feature_names(model_name, model, scalers)

    st.subheader("Caracteristiques de la parcelle")
    col1, col2, col3 = st.columns(3)

    with col1:
        annee = st.number_input("Annee", min_value=2015, max_value=2035, value=2024, step=1)
        saison = st.selectbox("Saison", ["A", "B"])
        province = st.selectbox("Province", PROVINCES, index=PROVINCES.index("Gitega"))
        culture = st.selectbox("Culture", CULTURES, index=CULTURES.index("Haricot"))

    with col2:
        altitude_m = st.number_input("Altitude (m)", min_value=0.0, value=1720.0, step=10.0)
        pluviometrie_mm = st.number_input("Pluviometrie (mm)", min_value=0.0, value=430.0, step=10.0)
        temperature_moy_C = st.number_input("Temperature moyenne (C)", min_value=0.0, value=18.2, step=0.1)
        superficie_ha = st.number_input("Superficie (ha)", min_value=0.1, value=2.5, step=0.1)

    with col3:
        utilisation_engrais = st.radio("Utilisation d'engrais", ["Non", "Oui"], horizontal=True)
        acces_irrigation = st.radio("Acces a l'irrigation", ["Non", "Oui"], horizontal=True)
        nb_menages = st.number_input("Nombre de menages", min_value=1, value=100, step=1)

    values = {
        "annee": annee,
        "saison": saison,
        "province": province,
        "culture": culture,
        "altitude_m": altitude_m,
        "pluviometrie_mm": pluviometrie_mm,
        "temperature_moy_C": temperature_moy_C,
        "superficie_ha": superficie_ha,
        "utilisation_engrais": 1 if utilisation_engrais == "Oui" else 0,
        "acces_irrigation": 1 if acces_irrigation == "Oui" else 0,
        "nb_menages": nb_menages,
    }

    input_frame = make_input_frame(values, feature_names)

    if st.button("Predire la recolte", type="primary", use_container_width=True):
        prediction, probability = predict(model, scaler, input_frame)
        label = "Bonne recolte" if prediction == 1 else "Mauvaise recolte"
        confidence = probability if prediction == 1 else 1 - probability

        result_col, prob_col = st.columns([2, 1])
        with result_col:
            if prediction == 1:
                st.success(f"Prediction : {label}")
            else:
                st.error(f"Prediction : {label}")
        with prob_col:
            st.metric("Probabilite associee", f"{confidence * 100:.1f}%")

        st.progress(max(0.0, min(1.0, probability)))
        st.caption(f"Probabilite d'une bonne recolte : {probability * 100:.1f}%")

        with st.expander("Voir les donnees envoyees au modele"):
            st.dataframe(input_frame, use_container_width=True)

    render_importance(model_name, model, feature_names)


if __name__ == "__main__":
    main()
