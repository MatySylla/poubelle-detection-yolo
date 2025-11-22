# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import datetime
import base64

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "runs/detect/model_poubelle2/weights/best.pt"  # ton modèle
DEMO_IMAGE = "/mnt/data/A_photograph_captures_a_cylindrical_trash_bin_made.png"  # image de demo
LOG_CSV = "predictions_log.csv"

# -------------------------
# REQUIRE CHECKS
# -------------------------
st.set_page_config(page_title="Poubelle IA – Dashboard", page_icon="🗑️", layout="wide")

# -------------------------
# UTIL: load model (cached)
# -------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Modèle introuvable: {path}. Place `best.pt` dans ce chemin.")
        return None
    return YOLO(path)

model = load_model()

# -------------------------
# UTIL: log prédiction
# -------------------------
def log_prediction(img_name, detections):
    ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    records = []
    if len(detections) == 0:
        records.append({"timestamp": ts, "image": img_name, "class": "none", "conf": 0.0})
    else:
        for d in detections:
            records.append({"timestamp": ts, "image": img_name, "class": d["class"], "conf": d["conf"]})
    df = pd.DataFrame(records)
    if os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)

# -------------------------
# UTIL: download link for dataframe
# -------------------------
def get_table_download_link(df: pd.DataFrame, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Télécharger les logs (.csv)</a>'
    return href

# -------------------------
# THEME SWITCH
# -------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def set_dark_mode(val):
    st.session_state.dark_mode = val

LIGHT_CSS = """
:root {
  --bg: #FFFFFF;
  --card: #F8FAFC;
  --text: #0B2545;
}
"""
DARK_CSS = """
:root {
  --bg: #0B1220;
  --card: #0F1724;
  --text: #E6EEF8;
}
"""

def inject_css():
    css = DARK_CSS if st.session_state.dark_mode else LIGHT_CSS
    st.markdown(f"<style>{css} body {{ background: var(--bg); color: var(--text); }}</style>", unsafe_allow_html=True)

inject_css()

# -------------------------
# SIDEBAR: navigation + settings
# -------------------------
st.sidebar.title("🗂 Menu")
page = st.sidebar.radio("Aller à", ["Accueil", "Statistiques", "À propos", "Paramètres"])

st.sidebar.markdown("---")
st.sidebar.write("Theme")
st.sidebar.checkbox("Mode sombre", value=st.session_state.dark_mode, on_change=set_dark_mode, args=(not st.session_state.dark_mode,))
st.sidebar.markdown("---")
st.sidebar.write("Démo image :")
if os.path.exists(DEMO_IMAGE):
    st.sidebar.image(DEMO_IMAGE, use_column_width=True)
st.sidebar.markdown("---")
if model is not None:
    with open(MODEL_PATH, "rb") as f:
        st.sidebar.download_button("⬇️ Télécharger le modèle", f, "best.pt")

# -------------------------
# PLACEHOLDERS pour éléments dynamiques
# -------------------------
uploaded_img_placeholder = st.empty()
annotated_img_placeholder = st.empty()
results_placeholder = st.empty()

# -------------------------
# PAGE: Accueil
# -------------------------
if page == "Accueil":
    st.markdown("<h1 style='text-align:center;'>🗑️ Détection de Poubelles (YOLOv8)</h1>", unsafe_allow_html=True)
    st.markdown("**Uploader une image** pour détecter si une poubelle est **pleine** ou **vide**.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded_file = st.file_uploader("📤 Charger une image", type=["jpg", "jpeg", "png"])
        st.markdown("Ou utiliser l'image de démonstration ci-dessous :")
        if st.button("🔎 Utiliser l'image de démo"):
            if os.path.exists(DEMO_IMAGE):
                with open(DEMO_IMAGE, "rb") as f:
                    uploaded_bytes = f.read()
                    uploaded_file = io.BytesIO(uploaded_bytes)
            else:
                st.error("Image de démo introuvable.")

    with col2:
        st.info("Paramètres d'inférence")
        conf_th = st.slider("Seuil de confiance", 0.7, 0.80, 0.75, 0.01)
        max_det = st.number_input("Max détections à afficher", min_value=1, max_value=20, value=5, step=1)

    if uploaded_file:
        # Lecture image
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Impossible de lire le fichier image.")
            st.stop()

        uploaded_img_placeholder.image(img, caption="Image chargée", use_container_width=True)

        if model is None:
            st.error("Modèle non chargé.")
            st.stop()

        with st.spinner("🔍 Analyse en cours ..."):
            results = model.predict(np.array(img), conf=conf_th, max_det=max_det)

        # Annotated image
        ann = results[0].plot()
        annotated_img_placeholder.image(ann, caption="Image annotée", use_container_width=True)

        # Extraire les détections
        boxes = results[0].boxes
        detections = []
        if boxes is None or len(boxes) == 0:
            results_placeholder.warning("❌ Aucune poubelle détectée.")
            log_prediction("uploaded_image", [])
        else:
            results_placeholder.success(f"✅ {len(boxes)} détection(s) trouvée(s)")
            for i, b in enumerate(boxes):
                cls_id = int(b.cls[0])
                cls_name = model.names[cls_id]
                conf = float(b.conf[0]) * 100
                results_placeholder.markdown(f"**{i+1}.** `{cls_name}` — confiance: {conf:.2f}%")
                detections.append({"class": cls_name, "conf": conf})
            log_prediction("uploaded_image", detections)

# -------------------------
# PAGE: Statistiques
# -------------------------
elif page == "Statistiques":
    st.markdown("<h2>📊 Statistiques des prédictions</h2>", unsafe_allow_html=True)
    if os.path.exists(LOG_CSV):
        df = pd.read_csv(LOG_CSV)
        st.dataframe(df.sort_values("timestamp", ascending=False).head(200))
        st.markdown("### Résumé")
        summary = df.groupby("class").agg(count=("class","size"), avg_conf=("conf","mean")).reset_index()
        st.table(summary)
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    else:
        st.info("Aucun log trouvé. Effectuer des prédictions sur la page Accueil.")

# -------------------------
# PAGE: À propos
# -------------------------
elif page == "À propos":
    st.markdown("<h2>📚 À propos</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Projet :** Détection de poubelles pleines/vides  
    **Auteur :** Maty Sylla  
    **Technos :** YOLO (Ultralytics), Streamlit  
    **But :** Démonstration d'un pipeline de vision par ordinateur pour la gestion intelligente des déchets.
    """)
    st.markdown("### Documentation courte")
    st.markdown("""
    - Télécharge le modèle via la sidebar.  
    - Uploade une image et ajuste le seuil de confiance.  
    - Visualise les résultats et consulte les statistiques.
    """)
    st.markdown("### Chemin de l'image de demo (locale):")
    st.code(DEMO_IMAGE)

# -------------------------
# PAGE: Paramètres
# -------------------------
elif page == "Paramètres":
    st.markdown("<h2>⚙️ Paramètres</h2>", unsafe_allow_html=True)
    st.markdown("Modifier le chemin du modèle ou réinitialiser les logs.")
    model_path_input = st.text_input("Chemin du modèle YOLO", MODEL_PATH)
    if st.button("🔁 Recharger le modèle"):
        if os.path.exists(model_path_input):
            st.session_state.clear()
            st.experimental_rerun()
        else:
            st.error("Chemin introuvable.")
    if st.button("🧹 Réinitialiser logs"):
        if os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)
            st.success("Logs supprimés.")
        else:
            st.info("Aucun log à supprimer.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Développé par Maty Sylla • Projet IA – 2025</p>", unsafe_allow_html=True)
