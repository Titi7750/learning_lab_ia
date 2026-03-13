""" Helpers for running YOLO inference and building diagnostics """

import cv2
import numpy as np
from typing import Any
from pathlib import Path
from ultralytics import YOLO
from config import MODEL_PATH

# -----

def _format_box(param_box: Any, param_names: dict[int, str]) -> dict[str, Any]:
    """ Convert one YOLO box to a simple serializable diagnostic row """

    cls_idx = int(param_box.cls.item())
    confidence = float(param_box.conf.item())

    return {
        "classe": param_names.get(cls_idx, str(cls_idx)),
        "confiance": round(confidence, 4)
    }

# -----

def run_diagnostic(param_image_bytes: bytes, param_conf_threshold: float = 0.4) -> dict[str, Any]:
    """ Run prediction and return image + text diagnostics for Streamlit """

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Modele introuvable: {MODEL_PATH}")

    image_array = np.frombuffer(param_image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Impossible de decoder l'image envoyee.")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=image, conf=param_conf_threshold, verbose=False)

    first_result = results[0]
    plotted = first_result.plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    rows = [_format_box(box, model.names) for box in first_result.boxes]
    detected_classes = sorted({row["classe"] for row in rows})

    if rows:
        summary = (
            f"{len(rows)} detection(s) - classes: {', '.join(detected_classes)}"
        )
    else:
        summary = "Aucun composant détecté avec le seuil actuel."

    return {
        "summary": summary,
        "rows": rows,
        "annotated_image": plotted_rgb,
    }
