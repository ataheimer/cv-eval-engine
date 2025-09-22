import json
from pathlib import Path

FEATURES = [
    "skill_overlap_count",
    "mandatory_coverage_ratio",
    "preferred_overlap_count",
    "years_of_experience",
    "edu_level_ord",
]

_DEF_JOBS_PATH = Path("data/jobs_sample.json")


def load_jobs():
    if _DEF_JOBS_PATH.exists():
        return json.loads(_DEF_JOBS_PATH.read_text(encoding="utf-8"))
    # Fallback: tek bir demo iş
    return {
        "demo": {
            "title": "Python Developer (Predictive Modelling & Gradient Boosting)",
            "mandatory": ["python","xgboost","lightgbm","catboost","pandas","numpy","scikit-learn"],
            "preferred": ["mlflow","airflow","docker"],
            "min_years": 0,
            "edu_min": "BSc"
        }
    }


def build_features(cv_feats: dict, job: dict):
    s_cv = set(map(str.lower, cv_feats["skills_found"]))
    s_mand = set(map(str.lower, job.get("mandatory", [])))
    s_pref = set(map(str.lower, job.get("preferred", [])))

    feat = {
        "skill_overlap_count": len(s_cv & (s_mand | s_pref)),
        "mandatory_coverage_ratio": (len(s_cv & s_mand) / max(1, len(s_mand))),
        "preferred_overlap_count": len(s_cv & s_pref),
        "years_of_experience": cv_feats["years_of_experience"],
        "edu_level_ord": {"Other":0,"BSc":1,"MSc":2,"PhD":3}.get(cv_feats["education_level"],0),
    }
    return feat

