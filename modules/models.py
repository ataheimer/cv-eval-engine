import os, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from .features import FEATURES, build_features, load_jobs

# İsteğe bağlı importlar
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
META_PATH = MODEL_DIR / "metadata.json"
DATA_PATH = Path("data/labeled_pairs.csv")


# ========= Weak-supervision: etiket üretimi (biraz daha esnetildi) =========
def _generate_dummy_labels():
    jobs = load_jobs()
    rows = []

    cvs = [
        ({"skills_found":["python","pandas","numpy","scikit-learn","xgboost"],"years_of_experience":1,"education_level":"BSc"}),
        ({"skills_found":["python","sql","pandas","numpy","lightgbm"],"years_of_experience":2,"education_level":"MSc"}),
        ({"skills_found":["python","catboost","airflow","docker"],"years_of_experience":3,"education_level":"BSc"}),
        ({"skills_found":["python","sql"],"years_of_experience":0,"education_level":"Other"}),
        ({"skills_found":["python","xgboost","lightgbm","catboost","mlflow"],"years_of_experience":4,"education_level":"MSc"}),
        ({"skills_found":["python","django","flask"],"years_of_experience":2,"education_level":"BSc"}),
        ({"skills_found":["python","spark","aws"],"years_of_experience":3,"education_level":"BSc"}),
        ({"skills_found":["python","xgboost","power bi"],"years_of_experience":1,"education_level":"BSc"}),
        ({"skills_found":["python","catboost","lightgbm","git","linux"],"years_of_experience":2,"education_level":"BSc"}),
        ({"skills_found":["python","pytorch","tensorflow"],"years_of_experience":3,"education_level":"MSc"}),
        ({"skills_found":["python","mlflow","airflow"],"years_of_experience":1,"education_level":"BSc"}),
        ({"skills_found":["python"],"years_of_experience":0,"education_level":"Other"}),
    ]

    for j_id, job in jobs.items():
        for idx, cv in enumerate(cvs):
            feat = build_features(cv, job)
            # Esnetilmiş kural:
            mand_cov = feat["mandatory_coverage_ratio"]
            pref_cnt = feat["preferred_overlap_count"]
            ok_cov = (mand_cov >= 0.5) or (mand_cov >= 0.34 and pref_cnt >= 1)
            ok_year = cv["years_of_experience"] >= job.get("min_years", 0)
            label = 1 if (ok_cov and ok_year) else 0
            rows.append({"job_id": j_id, "cv_id": idx, **feat, "label": label})

    df = pd.DataFrame(rows)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def _load_training_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = _generate_dummy_labels()
    X = df[FEATURES].values
    y = df["label"].values.astype(int)
    return X, y


# ===================== CV değerlendirme araçları (AUC & CV) =================
def _cv_auc(estimator, X, y, fit_params=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in skf.split(X, y):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        if fit_params:
            estimator.fit(X_tr, y_tr, **fit_params)
        else:
            estimator.fit(X_tr, y_tr)
        if hasattr(estimator, "predict_proba"):
            p = estimator.predict_proba(X_va)[:, 1]
        else:
            p = estimator.predict(X_va)
            if p.ndim == 1:
                p = p.astype(float)
        oof[va] = p
    return float(roc_auc_score(y, oof)), oof


def _pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg / max(1, pos))


# ====================== Tekil modelleri eğit (weighted) ======================

def _train_xgb(X, y):
    if xgb is None:
        return None
    spw = _pos_weight(y)
    # FEATURES sırasına göre artan kısıt: +1 = arttıkça olasılık artsın
    # [skill_overlap_count, mandatory_coverage_ratio, preferred_overlap_count, years, edu]
    mono = "(+1,+1,+1,+1,+1)"
    mdl = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
        scale_pos_weight=spw, tree_method="hist",
        eval_metric="auc", random_state=42,
        monotone_constraints=mono
    )
    auc, oof = _cv_auc(mdl, X, y)
    mdl.fit(X, y)
    return ("xgboost", mdl, auc, oof)


def _train_lgb(X, y):
    if lgb is None:
        return None
    spw = _pos_weight(y)
    # LightGBM’de liste halinde: +1 artan; 0 kısıtsız
    mono = [+1, +1, +1, +1, +1]
    mdl = lgb.LGBMClassifier(
        n_estimators=700, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
        objective="binary", random_state=42,
        scale_pos_weight=spw,
        monotone_constraints=mono
    )
    auc, oof = _cv_auc(mdl, X, y)
    mdl.fit(X, y)
    return ("lightgbm", mdl, auc, oof)



def _train_cat(X, y):
    if CatBoostClassifier is None:
        return None
    # CatBoost class_weights ile dengeleme
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    cw = [1.0, float(neg / max(1, pos))] if pos > 0 else [1.0, 1.0]
    mdl = CatBoostClassifier(
        iterations=900, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=False,
        class_weights=cw
    )
    auc, oof = _cv_auc(mdl, X, y)
    mdl.fit(X, y, verbose=False)
    return ("catboost", mdl, auc, oof)


# ============================ Ensemble wrapper ===============================
class AvgEnsemble:
    def __init__(self, models):
        self.models = models  # list of (name, fitted_model)

    def predict_proba(self, X):
        probs = []
        for _, m in self.models:
            p = m.predict_proba(X)[:, 1]
            probs.append(p)
        p_mean = np.mean(probs, axis=0)
        # Sklearn style: 2-column proba döndür
        return np.vstack([1 - p_mean, p_mean]).T


# ========================= Kalibrasyon + Kaydetme ============================
def _calibrate(model, X, y, method="isotonic"):
    """
    Modeli olasılık açısından kalibre eder.
    cv=3 ile CalibratedClassifierCV kullanıyoruz (küçük veri için stabil).
    """
    try:
        calib = CalibratedClassifierCV(model, method=method, cv=3)
        calib.fit(X, y)
        return calib
    except Exception:
        # kalibrasyon başarısızsa orijinal modeli döndür
        return model


def train_all_and_select():
    """
    XGBoost, LightGBM, CatBoost ve (varsa) Ensemble'ı 5-fold ROC-AUC ile karşılaştır.
    En iyiyi seç, kalibre et ve kaydet.
    """
    X, y = _load_training_data()

    candidates = []
    for trainer in (_train_xgb, _train_lgb, _train_cat):
        out = trainer(X, y)
        if out is not None:
            candidates.append(out)  # (name, model, auc, oof)

    if not candidates:
        raise RuntimeError("xgboost/lightgbm/catboost modellerinden hiçbiri kurulu değil.")

    # Ensemble (mevcut tüm modellerin ortalaması)
    if len(candidates) >= 2:
        # OOF ortalama AUC
        oof_stack = np.mean([c[3] for c in candidates], axis=0)
        ens_auc = float(roc_auc_score(y, oof_stack))
        ens_model = AvgEnsemble([(c[0], c[1]) for c in candidates])
        candidates.append(("ensemble_avg", ens_model, ens_auc, oof_stack))

    # En iyiyi seç
    best_name, best_model, best_auc, _ = max(candidates, key=lambda t: t[2])

    # Tüm veride kalibrasyon
    best_model_cal = _calibrate(best_model, X, y, method="isotonic")

    # Kaydet
    joblib.dump(best_model_cal, BEST_MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "best_model": best_name,
        "cv_auc": {name: float(auc) for name, _, auc, _ in candidates},
        "features": FEATURES,
        "calibrated": True,
        "calibration": "isotonic",
    }, indent=2), encoding="utf-8")

    return best_name, float(best_auc), {name: float(auc) for name, _, auc, _ in candidates}


def ensure_model():
    if not BEST_MODEL_PATH.exists() or not META_PATH.exists():
        train_all_and_select()


def _load_best():
    if not BEST_MODEL_PATH.exists() or not META_PATH.exists():
        ensure_model()
    model = joblib.load(BEST_MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def predict_fit(cv_feat_dict: dict, job_dict: dict):
    model, meta = _load_best()
    feat = build_features(cv_feat_dict, job_dict)
    X = np.array([[feat[k] for k in meta.get("features", FEATURES)]], dtype=float)
    proba = float(model.predict_proba(X)[:, 1][0])
    return proba, feat


def quick_ablation_report():
    import itertools
    X, y = _load_training_data()
    from sklearn.linear_model import LogisticRegression
    base_feats = FEATURES
    # küçük ve hızlı bir modelle kaba fikir (logreg + 5-fold)
    def cv_auc_lr(cols):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        Xc = pd.DataFrame(X, columns=base_feats)[list(cols)].values
        oof = np.zeros_like(y, dtype=float)
        for tr, va in skf.split(Xc, y):
            m = LogisticRegression(max_iter=200).fit(Xc[tr], y[tr])
            oof[va] = m.predict_proba(Xc[va])[:,1]
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, oof))
    base_auc = cv_auc_lr(base_feats)
    print("BASE AUC:", base_auc)
    for f in base_feats:
        cols = [c for c in base_feats if c != f]
        auc = cv_auc_lr(cols)
        print(f"- drop {f:>25}  -> AUC {auc:.4f} (Δ {auc-base_auc:+.4f})")
