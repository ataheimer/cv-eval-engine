import numpy as np
import joblib
from .features import FEATURES, build_features

try:
    import shap
except Exception:
    shap = None


def explain_top_k(cv_feat_dict: dict, job_dict: dict, k: int = 3):
    if shap is None:
        return None
    model = joblib.load("models/xgb_fit_model.bin")
    feat = build_features(cv_feat_dict, job_dict)
    x = np.array([[feat[f] for f in FEATURES]])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x)[0]
    ranked = sorted(zip(FEATURES, sv), key=lambda t: -abs(t[1]))
    return ranked[:k]
