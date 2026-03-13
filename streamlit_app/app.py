""" Streamlit interface for PCB diagnostic with YOLO """

import sys
import streamlit as st
from pathlib import Path
from streamlit_app.components.diagnostics import run_diagnostic

# -----

# Ensure project root imports work when running `streamlit run streamlit_app/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="PCB Diagnostic", page_icon="🧪", layout="wide")
st.title("Diagnostic PCB - Detection des composants")
st.write("Charge une image de carte PCB, puis lance la détection des composants avec YOLO.")

uploaded_file = st.file_uploader(
    "Image PCB",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

conf_threshold = st.slider(
    "Seuil de confiance",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05,
)

run_prediction = st.button("Lancer le diagnostic", type="primary")

if run_prediction:
    if uploaded_file is None:
        st.warning("Ajoute une image avant de lancer le diagnostic.")
    else:
        with st.spinner("Prediction en cours..."):
            try:
                payload = uploaded_file.getvalue()
                result = run_diagnostic(payload, param_conf_threshold=conf_threshold)
            except Exception as exc:
                st.error(f"Erreur pendant la prediction: {exc}")
            else:
                left_col, right_col = st.columns([3, 2])

                with left_col:
                    st.image(result["annotated_image"], caption="Image annotee")

                with right_col:
                    st.success(result["summary"])
                    if result["rows"]:
                        st.dataframe(result["rows"], use_container_width=True)
                    else:
                        st.info("Aucune detection a afficher.")
