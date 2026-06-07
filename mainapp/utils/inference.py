import os
import joblib
import numpy as np
from django.conf import settings

MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models", "fertility_model.pkl")

_model = None
def load_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            _model = None
    return _model

def predict_from_features(features: dict):
    """
    features keys: ph,n,p,k,oc,ec,moisture
    returns: {'label': 'LOW'|'MEDIUM'|'HIGH', 'confidence': float, 'probs': {...}}
    """
    model = load_model()
    # prepare vector in expected order: ph,n,p,k,oc,ec,moisture
    import numpy as np
    X = np.array([features.get('ph') or 6.5,
                  features.get('n') or 40,
                  features.get('p') or 20,
                  features.get('k') or 150,
                  features.get('oc') or 0.5,
                  features.get('ec') or 0.5,
                  features.get('moisture') or 15.0]).reshape(1, -1)

    if model is None:
        # fallback simple scoring
        score = 0.0
        ph = features.get('ph') or 6.5
        n = features.get('n') or 40
        oc = features.get('oc') or 0.5
        if 6 <= ph <= 7.5: score += 0.3
        if n >= 50: score += 0.3
        if oc >= 0.8: score += 0.2
        label = 'HIGH' if score >= 0.6 else ('MEDIUM' if score >= 0.3 else 'LOW')
        return {'label': label, 'confidence': float(score), 'probs': {}}

    try:
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probs))
        label = classes[best_idx]
        confidence = float(probs[best_idx])
        prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
        return {'label': label, 'confidence': confidence, 'probs': prob_map}
    except Exception as e:
        return {'label': 'UNKNOWN', 'confidence': 0.0, 'probs': {}}


def manual_predict_model(values):
    model = load_model()
    if model is None:
        return {"label": "Model Not Found", "confidence": 0}

    X = np.array([
        values["N"],
        values["P"],
        values["K"],
        values["temperature"],
        values["humidity"],
        values["ph"],
        values["rainfall"],
    ]).reshape(1, -1)

    try:
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probs))
        label = classes[best_idx]
        confidence = float(probs[best_idx])
        return {"label": label, "confidence": confidence}

    except:
        return {"label": "Error", "confidence": 0}