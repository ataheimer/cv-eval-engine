from typing import Dict, List
import os, re, json
import requests

# Normalize için kanonik beceri sözlüğü
CANON_SKILLS = {
    "python":"python","sql":"sql","pandas":"pandas","numpy":"numpy","scikit-learn":"scikit-learn",
    "sklearn":"scikit-learn","tensorflow":"tensorflow","pytorch":"pytorch","xgboost":"xgboost",
    "lightgbm":"lightgbm","catboost":"catboost","docker":"docker","airflow":"airflow","spark":"spark",
    "aws":"aws","gcp":"gcp","azure":"azure","tableau":"tableau","power bi":"power bi","powerbi":"power bi",
    "fastapi":"fastapi","flask":"flask","django":"django","git":"git","linux":"linux","mlflow":"mlflow",
    "kubernetes":"kubernetes","openshift":"openshift","kafka":"kafka","elasticsearch":"elasticsearch",
    "react":"react","postgres":"postgresql","postgresql":"postgresql","mysql":"mysql","next.js":"next.js",
    "supabase":"supabase","llm":"llm","rag":"rag","langchain":"langchain"
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ESKİ:
# PROMPT = """ ... {cv_text} ... """

# YENİ (özel yer tutucu kullan):
PROMPT = """
You are an expert HR data scientist. Given the raw CV text below, infer the candidate's likely skills (even if not explicitly listed), years of experience, highest education level, and links (GitHub/Google Scholar/papers). Return **strict JSON** with keys:
{
  "inferred_skills": ["python", "sql", ...],
  "years_of_experience_inferred": number,
  "education_level": "PhD|MSc|BSc|Other",
  "links": ["https://..."]
}
Rules:
- Use reasonable priors: Computer Engineering implies basic programming + data structures; AI company role names imply ML skills, etc.
- Do not hallucinate specific company names or project details that are not in the text.
- If unsure, omit the skill.
CV TEXT START
---
<<CV_TEXT>>
---
CV TEXT END
"""


def _ollama_generate(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama -> {"response": "..."}
    return data.get("response", "")

def _extract_json_block(s: str) -> str:
    # 1) ```json ... ``` code-fence
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) İlk {...} bloğu
    m = re.search(r"\{\s*[\s\S]*\}\s*", s)
    if m:
        return m.group(0).strip()
    return ""

def _safe_json_load(s: str):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def _normalize_skills(skills) -> List[str]:
    # LLM bazen "python, sql" gibi tek string döndürebilir
    if isinstance(skills, str):
        skills = [p.strip() for p in skills.split(",") if p.strip()]
    if not isinstance(skills, list):
        skills = []
    out = []
    for sk in skills:
        key = str(sk).strip().lower()
        if key in CANON_SKILLS:
            out.append(CANON_SKILLS[key])
    # unique
    seen, norm = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); norm.append(s)
    return norm

def _with_defaults(js: Dict) -> Dict:
    if not isinstance(js, dict):
        js = {}
    js.setdefault("inferred_skills", [])
    js.setdefault("years_of_experience_inferred", 0)
    js.setdefault("education_level", None)
    js.setdefault("links", [])
    # normalize edilmiş alan
    js["inferred_skills_norm"] = _normalize_skills(js.get("inferred_skills"))
    return js

def enrich_with_llm(cv_text: str, baseline: Dict) -> Dict:
    prompt = PROMPT.replace("<<CV_TEXT>>", cv_text[:12000])
    raw = _ollama_generate(prompt)
    js_block = _extract_json_block(raw)
    js = _safe_json_load(js_block)
    return _with_defaults(js)


def merge_skillsets(skill_list: List[str], inferred: List[str]) -> List[str]:
    base = [str(s).lower() for s in (skill_list or [])]
    add  = [str(s).lower() for s in (inferred or [])]
    seen, merged = set(), []
    for s in base + add:
        if s and s not in seen:
            seen.add(s); merged.append(s)
    return merged

# ESKİ:
# JOB_PROMPT = """ ... {job_text} ... """

# YENİ:
JOB_PROMPT = """
You are a technical recruiter. From the JOB TEXT, extract structured requirements.
Return STRICT JSON with keys:
{
  "title": "string",
  "mandatory": ["python","sql",...],
  "preferred": ["docker","airflow",...],
  "min_years": 0,
  "edu_min": "PhD|MSc|BSc|Other"
}
Rules:
- Prefer concrete, verifiable skills (frameworks, libs, tools). Avoid soft skills.
- If the text implies ranges (e.g. 2+ years), pick a reasonable integer (e.g. 2).
- If unsure about education, choose "BSc" for standard software roles.
- Normalize to lowercase tokens (e.g., "Power BI" -> "power bi", "PostgreSQL" -> "postgresql").
JOB TEXT START
---
<<JOB_TEXT>>
---
JOB TEXT END
"""
def enrich_job_with_llm(job_text: str) -> Dict:
    prompt = JOB_PROMPT.replace("<<JOB_TEXT>>", job_text[:12000])
    raw = _ollama_generate(prompt)
    js_block = _extract_json_block(raw)
    js = _safe_json_load(js_block)
    if not isinstance(js, dict):
        js = {}
    js.setdefault("title", None)
    js.setdefault("mandatory", [])
    js.setdefault("preferred", [])
    js.setdefault("min_years", 0)
    js.setdefault("edu_min", "BSc")
    js["mandatory"] = _normalize_skills(js.get("mandatory"))
    js["preferred"] = _normalize_skills(js.get("preferred"))
    try:
        js["min_years"] = int(js.get("min_years") or 0)
    except Exception:
        js["min_years"] = 0
    if js.get("edu_min") not in {"PhD","MSc","BSc","Other"}:
        js["edu_min"] = "BSc"
    return js
