import streamlit as st
import pandas as pd
import pickle, os, datetime
from sklearn.preprocessing import LabelEncoder

# Configuration 
st.set_page_config(page_title="Patient Churn", page_icon="🏥", layout="wide")

MODEL_PATH = "model_dump.pkl"
SPECIALTIES = ["Cardiology","Family Medicine","General Practice",
               "Internal Medicine","Neurology","Orthopedics","Pediatrics"]
STATES      = ["CA","FL","GA","IL","MI","NC","NY","OH","PA","TX"]
INSURANCES  = ["Medicaid","Private","Self-Pay"]

# Chargement du modèle 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):                    # Vérifie la présence du fichier
        return None, f"Fichier '{MODEL_PATH}' introuvable."
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), None                   # Désérialisation via pickle
    except Exception as e:
        return None, str(e)

# Frequency encoding depuis le dataset d'entraînement
@st.cache_data
def get_freq_maps():
    for csv in ["patient_churn_dataset.csv", "Patient_churn_dataset.csv"]:
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            n  = len(df)
            return {                                       # Fréquence relative de chaque modalité
                "specialty": (df.groupby("Specialty").size() / n).to_dict(),
                "insurance": (df.groupby("Insurance_Type").size() / n).to_dict(),
                "state":     (df.groupby("State").size() / n).to_dict(),
            }
    # Fallback : fréquences uniformes si le CSV est absent
    return {"specialty": {s: 1/7 for s in SPECIALTIES},
            "insurance": {i: 1/3 for i in INSURANCES},
            "state":     {s: 1/10 for s in STATES}}

# Prétraitement (réplique exacte du notebook) 
def preprocess(inputs):
    fm  = get_freq_maps()
    le  = LabelEncoder().fit(["Female","Male"])
    dt  = pd.to_datetime(inputs["last_date"], errors="coerce")  # Décomposition de la date
    return pd.DataFrame([{
        "Age":                        inputs["age"],
        "Tenure_Months":              inputs["tenure"],
        "Visits_Last_Year":           inputs["visits"],
        "Missed_Appointments":        inputs["missed"],
        "Days_Since_Last_Visit":      inputs["days_since"],
        "Overall_Satisfaction":       inputs["satisfaction"],
        "Wait_Time_Satisfaction":     inputs["wait_sat"],
        "Staff_Satisfaction":         inputs["staff_sat"],
        "Provider_Rating":            inputs["provider_rating"],
        "Avg_Out_Of_Pocket_Cost":     inputs["out_of_pocket"],
        "Billing_Issues":             inputs["billing"],
        "Portal_Usage":               inputs["portal"],
        "Referrals_Made":             inputs["referrals"],
        "Distance_To_Facility_Miles": inputs["distance"],
        "Gender_encoded":   int(le.transform([inputs["gender"]])[0]),           # Female=0, Male=1
        "Specialty_encoded": fm["specialty"].get(inputs["specialty"], 1/7),     # Frequency encoding
        "Insurance_encoded": fm["insurance"].get(inputs["insurance"], 1/3),     # Frequency encoding
        "State_encoded":     fm["state"].get(inputs["state"], 1/10),            # Frequency encoding
        "year":        dt.year, "month": dt.month,
        "day":         dt.day,  "day_of_week": dt.dayofweek,                    # Composantes temporelles
    }])

# Interface
st.title("🏥 Patient Churn Predictor")
st.caption("AdaBoostClassifier · PyCaret · SADO KAMGA Miguel Wesley")

model, err = load_model()

# Sidebar — saisie des features
with st.sidebar:
    st.header("⚙️ Données Patient")
    st.markdown("**Démographie**")
    age    = st.number_input("Âge",         18, 90,  52)
    gender = st.selectbox("Genre",          ["Female","Male"])
    state  = st.selectbox("État",           STATES)

    st.markdown("**Médical**")
    specialty = st.selectbox("Spécialité",   SPECIALTIES, index=2)
    insurance = st.selectbox("Assurance",    INSURANCES,  index=2)

    st.markdown("**Historique**")
    tenure    = st.number_input("Ancienneté (mois)",          1,  120, 60)
    visits    = st.number_input("Visites/an",                 0,   50,  5)
    missed    = st.number_input("RDV manqués",                0,   20,  1)
    days_since= st.number_input("Jours depuis visite",        0,  500, 30)
    last_date = st.date_input("Dernière interaction",
                              datetime.date(2024, 6, 15))

    st.markdown("**Satisfaction (1–5)**")
    satisfaction   = st.slider("Globale",          1.0, 5.0, 3.5, 0.1)
    wait_sat       = st.slider("Temps d'attente",  1.0, 5.0, 3.0, 0.1)
    staff_sat      = st.slider("Personnel",        1.0, 5.0, 3.5, 0.1)
    provider_rating= st.slider("Prestataire",      1.0, 5.0, 4.0, 0.1)

    st.markdown("**Finances & Usage**")
    out_of_pocket = st.number_input("Coût patient ($)", 0, 10000, 500, 50)
    billing       = st.number_input("Pb facturation",   0,    10,   0)
    portal        = st.number_input("Sessions portail", 0,    50,   5)
    referrals     = st.number_input("Références",       0,     3,   1)
    distance      = st.number_input("Distance (miles)", 0.5, 50.0, 25.0, 0.5)

    predict_btn = st.button("🔮 Prédire", use_container_width=True)

# Zone principale — résultat
col1, col2 = st.columns(2)

with col1:                                                        # Récapitulatif des données saisies
    st.subheader("📋 Récapitulatif")
    st.dataframe(pd.DataFrame({
        "Variable": ["Âge","Genre","État","Spécialité","Assurance","Ancienneté","Visites/an",
                     "RDV manqués","Jours/visite","Satisfaction","Sat. attente",
                     "Sat. personnel","Note prestataire","Coût ($)","Pb fact.","Portail","Réf.","Distance"],
        "Valeur":   [age, gender, state, specialty, insurance, tenure, visits,
                     missed, days_since, satisfaction, wait_sat,
                     staff_sat, provider_rating, out_of_pocket, billing, portal, referrals, distance]
    }), hide_index=True, use_container_width=True)

with col2:                                                        # Zone de prédiction
    st.subheader("🎯 Résultat")

    if err:                                                       # Erreur chargement modèle
        st.error(f"⚠️ {err}\n\nPlacez `modele.pkl` dans le même dossier que `app.py`.")

    elif predict_btn:
        try:
            X = preprocess({                                      # Construit le vecteur d'entrée
                "age": age, "gender": gender, "state": state, "tenure": tenure,
                "specialty": specialty, "insurance": insurance, "visits": visits,
                "missed": missed, "days_since": days_since, "last_date": str(last_date),
                "satisfaction": satisfaction, "wait_sat": wait_sat, "staff_sat": staff_sat,
                "provider_rating": provider_rating, "out_of_pocket": out_of_pocket,
                "billing": billing, "portal": portal, "referrals": referrals, "distance": distance,
            })

            pred = int(model.predict(X)[0])                      # Classe prédite : 0 ou 1

            if pred == 1:                                         # Patient à risque de churn
                st.error("🚨 **CHURN PROBABLE** — Ce patient risque de partir.")
            else:                                                 # Patient fidèle
                st.success("✅ **PATIENT FIDÈLE** — Ce patient devrait rester.")

            if hasattr(model, "predict_proba"):                   # Probabilités si disponibles
                proba = model.predict_proba(X)[0]
                st.metric("Probabilité de churn",  f"{proba[1]*100:.1f}%")
                st.metric("Probabilité de rétention", f"{proba[0]*100:.1f}%")
                st.progress(int(proba[1] * 100))                  # Jauge visuelle du risque

        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")

    else:
        st.info("👈 Renseignez les données puis cliquez sur **Prédire**.")
        st.balloons()