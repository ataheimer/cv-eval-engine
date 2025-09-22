import re, json
from pathlib import Path
import pdfplumber

SKILL_LEXICON = [
    "python","sql","pandas","numpy","scikit-learn","tensorflow","pytorch",
    "xgboost","lightgbm","catboost","docker","airflow","spark","aws","gcp",
    "azure","tableau","power bi","powerbi","fastapi","flask","django","git","linux",
    "mlflow","kubernetes","openshift","grafana","prometheus","kafka","elasticsearch"
]

DEGREE_ORDER = {"phd":3,"m.sc":2,"msc":2,"master":2,"b.sc":1,"bsc":1,"bachelor":1}


def _clean_text(t: str) -> str:
    t = t or ""
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_text_from_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        parts = []
        for p in pdf.pages:
            parts.append(p.extract_text() or "")
    return "\n".join(parts)


def extract_skills(text: str):
    t = _clean_text(text)
    found = []
    for sk in SKILL_LEXICON:
        pat = r"\b" + re.escape(sk).replace("\\ ", r"\\s*") + r"\b"
        if re.search(pat, t):
            found.append(sk.replace("powerbi","power bi"))
    return sorted(set(found))


def extract_education_level(t: str) -> str:
    t = _clean_text(t)
    for k,v in DEGREE_ORDER.items():
        if k in t:
            return {3:"PhD",2:"MSc",1:"BSc"}[v]
    return "Other"


def extract_years_experience(t: str) -> int:
    t = _clean_text(t)
    # 1) "X years" kalıbı
    m = re.findall(r"(\d+)\s+years?", t)
    cand = max([int(x) for x in m], default=0)
    # 2) Yıllara bakarak kaba tahmin (opsiyonel geliştirilebilir)
    return min(cand, 30)


def extract_from_file(path: Path):
    text = ""
    if path.suffix.lower() == ".pdf":
        text = _extract_text_from_pdf(path)
    else:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")

    skills = extract_skills(text)
    years = extract_years_experience(text)
    edu = extract_education_level(text)

    return {
        "raw_text": text[:15000],  # güvenlik için kes
        "skills_found": skills,
        "years_of_experience": years,
        "education_level": edu,
    }
