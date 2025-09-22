import json, re
from pathlib import Path
import pdfplumber

from modules.llm_enrich import enrich_job_with_llm, _normalize_skills

# (opsiyonel) basit keyword fallback
SKILLS = ["python","xgboost","lightgbm","catboost","pandas","numpy","scikit-learn",
          "flask","fastapi","django","react","postgresql","mysql","mlflow","airflow",
          "kubernetes","spark","kafka","elasticsearch","llm","rag","langchain",
          "power bi","tableau","git","docker","aws","gcp","azure"]

def heuristic_extract(text: str):
    t = text.lower()
    found = [s for s in SKILLS if re.search(r"\b"+re.escape(s).replace(" ", r"\s*")+r"\b", t)]
    found = _normalize_skills(found)
    # kaba bir bölüşüm: ilk 6 zorunlu, sonraki 6 tercihli
    return found[:6], found[6:12]

def extract_text_from_pdf(path: Path) -> str:
    with pdfplumber.open(path) as p:
        return "\n".join([pg.extract_text() or "" for pg in p.pages])

def main():
    out = {}
    in_dir = Path("data/jobs_pdfs")
    for pdf in in_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf)
        llm = enrich_job_with_llm(text)

        mand_h, pref_h = heuristic_extract(text)
        # hibrit birleştirme (LLM öncelikli, sonra heuristik ekle)
        mand = list(dict.fromkeys((llm.get("mandatory") or []) + mand_h))[:10]
        pref = list(dict.fromkeys((llm.get("preferred") or []) + pref_h))[:10]

        title = llm.get("title") or pdf.stem
        job_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

        out[job_id] = {
            "title": title,
            "mandatory": mand,
            "preferred": pref,
            "min_years": llm.get("min_years", 0),
            "edu_min": llm.get("edu_min", "BSc")
        }

    Path("data").mkdir(exist_ok=True)
    Path("data/jobs_sample.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("✔ yazıldı: data/jobs_sample.json (%d ilan)" % len(out))

if __name__ == "__main__":
    main()
