# modules/career.py
import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd

from .features import FEATURES, build_features, load_jobs

# İsteğe bağlı modeller
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

MODEL_DIR = Path("models")
CAREER_MODEL_PATH = MODEL_DIR / "career_best.pkl"
CAREER_META_PATH = MODEL_DIR / "career_meta.json"
DATA_PATH = Path("data/labeled_pairs.csv")   # mevcut zayıf etiket datasını yeniden kullanacağız

# ---- Seviyeler ve eşikler (0-100 skor aralığı) ----
STAGES = [
    ("intern",   0,  20),
    ("junior",  20, 40),
    ("mid",     40, 60),
    ("senior",  60, 80),
    ("expert",  80, 101),
]

def stage_from_score(s: float) -> str:
    for name, lo, hi in STAGES:
        if lo <= s < hi:
            return name
    return "intern"

def _target_rule(feat: dict) -> float:
    """
    Weak-supervision 'level_score' (0-100).
    Zorunlu kapsama + prefered + yıllar + eğitim birlikte puanlar.
    """
    cov = float(feat.get("mandatory_coverage_ratio", 0.0))         # 0..1
    pref = float(feat.get("preferred_overlap_count", 0.0))
    years = float(feat.get("years_of_experience", 0.0))
    edu = int(feat.get("edu_level_ord", 0))                         # 0/1/2/3

    # ağırlıklar: coverage > years > preferred > education
    score = (
        55.0 * cov +                     # %55 etki
        20.0 * min(years, 10) / 10.0 +   # %20 (10 yılda tavan)
        15.0 * min(pref, 3) / 3.0 +      # %15 (3 tercihli beceride tavan)
        10.0 * (edu / 3.0)               # %10
    )
    return float(max(0.0, min(100.0, score)))

def _load_or_build_training_frame():
    # Eğitim için mevcut weak-supervision tablosunu kullanıyoruz; yoksa diğer modül üretecek.
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        # Çok nadir olur; en azından boş dönmeyelim
        raise RuntimeError("labeled_pairs.csv bulunamadı. Önce ana modeli eğit (ensure_model).")
    # Bu tabloda FEATURES var; buradan hedefi kural ile hesaplıyoruz.
    targets = []
    for _, row in df.iterrows():
        feat = {k: row[k] for k in FEATURES}
        targets.append(_target_rule(feat))
    X = df[FEATURES].values
    y = np.array(targets, dtype=float)
    return X, y

def _cv_rmse(estimator, X, y):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(X):
        est = estimator
        est.fit(X[tr], y[tr])
        p = est.predict(X[va])
        oof[va] = p
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    return rmse

def _train_xgb(X, y):
    if xgb is None: return None
    mdl = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
        tree_method="hist", random_state=42
    )
    rmse = _cv_rmse(mdl, X, y)
    mdl.fit(X, y)
    return ("xgboost_reg", mdl, rmse)

def _train_lgb(X, y):
    if lgb is None: return None
    mdl = lgb.LGBMRegressor(
        n_estimators=600, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42
    )
    rmse = _cv_rmse(mdl, X, y)
    mdl.fit(X, y)
    return ("lightgbm_reg", mdl, rmse)

def _train_cat(X, y):
    if CatBoostRegressor is None: return None
    mdl = CatBoostRegressor(
        iterations=700, depth=6, learning_rate=0.05,
        loss_function="RMSE", random_seed=42, verbose=False
    )
    rmse = _cv_rmse(mdl, X, y)
    mdl.fit(X, y, verbose=False)
    return ("catboost_reg", mdl, rmse)

def train_career_and_select():
    X, y = _load_or_build_training_frame()
    MODEL_DIR.mkdir(exist_ok=True)

    cands = []
    for trainer in (_train_xgb, _train_lgb, _train_cat):
        out = trainer(X, y)
        if out: cands.append(out)
    if not cands:
        raise RuntimeError("Regresyon için xgboost/lightgbm/catboost bulunamadı.")

    # RMSE en küçük olan en iyi
    best = min(cands, key=lambda t: t[2])
    name, model, rmse = best
    joblib.dump(model, CAREER_MODEL_PATH)
    CAREER_META_PATH.write_text(json.dumps({
        "best_reg": name,
        "rmse_cv": {n: float(r) for n,_,r in cands},
        "features": FEATURES,
        "stage_bins": STAGES
    }, indent=2), encoding="utf-8")
    return name, float(rmse)

def ensure_career_model():
    if not CAREER_MODEL_PATH.exists() or not CAREER_META_PATH.exists():
        train_career_and_select()

def _load_model_meta():
    ensure_career_model()
    model = joblib.load(CAREER_MODEL_PATH)
    meta = json.loads(CAREER_META_PATH.read_text(encoding="utf-8"))
    return model, meta

def predict_current_stage(cv_feats: dict, job_dict: dict):
    """Şu anki 'level_score' ve stage."""
    model, meta = _load_model_meta()
    feat = build_features(cv_feats, job_dict)
    X = np.array([[feat[k] for k in meta.get("features", FEATURES)]], dtype=float)
    score = float(model.predict(X)[0])
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "stage": stage_from_score(score)
    }

def predict_future_trajectory(cv_feats: dict, job_dict: dict, years_ahead=(2,5)):
    """
    Basit bir 'progression' modeli: gelecekte coverage biraz artar ve yıl artar varsayımı.
    (İstersen daha sofistike bir progresyon fonksiyonu yazabiliriz.)
    """
    base_feat = build_features(cv_feats, job_dict)
    outputs = []
    for ya in years_ahead:
        sim = dict(base_feat)
        sim["years_of_experience"] = float(sim["years_of_experience"]) + ya
        # coverage/overlap’ta küçük bir kazanım varsayımı (0.05 * ya ile sınıra kadar)
        sim["mandatory_coverage_ratio"] = min(1.0, sim["mandatory_coverage_ratio"] + 0.05*ya)
        sim["preferred_overlap_count"] = min(sim["preferred_overlap_count"] + int(ya>0), 5)
        # edu_level aynı kalır
        model, meta = _load_model_meta()
        X = np.array([[sim[k] for k in meta.get("features", FEATURES)]], dtype=float)
        score = float(model.predict(X)[0])
        score = max(0.0, min(100.0, score))
        outputs.append({
            "years_ahead": ya,
            "score": round(score, 1),
            "stage": stage_from_score(score)
        })
    return outputs
