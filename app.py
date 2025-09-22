from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path
from modules.career import ensure_career_model, predict_current_stage, predict_future_trajectory
from modules.extract import extract_from_file
from modules.scoring import total_score
from modules.features import load_jobs
from modules.models import predict_fit, ensure_model
from modules.explain import explain_top_k
from modules.llm_enrich import enrich_with_llm, merge_skillsets

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "cv-eval-secret"

@app.route("/")
def index():
    jobs = load_jobs()
    return render_template("index.html", jobs=jobs)

@app.post("/evaluate")
def evaluate():
    file = request.files.get("cv_file")
    job_id = request.form.get("job_id")
    use_llm = (request.form.get("use_llm") == "on")
    show_shap = (request.form.get("show_shap") == "on")

    if not file or file.filename.strip() == "":
        flash("Please upload a CV file (PDF or TXT).", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    cv_path = UPLOAD_DIR / (str(uuid.uuid4()) + "_" + filename)
    file.save(cv_path)

    # 1) Parse + baseline extraction
    cv_feats = extract_from_file(cv_path)

    # 1b) (Optional) Enrichment with LLM
    llm = None
    if use_llm:
        try:
            llm = enrich_with_llm(cv_feats["raw_text"], cv_feats) or {}
            inferred_list = llm.get("inferred_skills_norm") or llm.get("inferred_skills") or []
            cv_feats["skills_found"] = merge_skillsets(cv_feats.get("skills_found", []), inferred_list)

            llm_years = llm.get("years_of_experience_inferred")
            if isinstance(llm_years, (int, float)):
                cv_feats["years_of_experience"] = max(cv_feats.get("years_of_experience", 0), int(llm_years))

            if (llm.get("education_level") in {"PhD","MSc","BSc","Other"}) and cv_feats.get("education_level") == "Other":
                cv_feats["education_level"] = llm["education_level"]
        except Exception as e:
            flash(f"LLM enrichment failed: {e}", "error")
            llm = None

    # 2) Basic scoring (0-100)
    base_score = total_score(
        cv_feats["skills_found"],
        cv_feats["years_of_experience"],
        cv_feats["education_level"]
    )

    # 3) Job matching (fit score)
    jobs = load_jobs()
    selected_job = None
    fit_output = None
    shap_top = None
    missing_mandatory = []
    hybrid_score = None  # hybrid score (0–100)

    career_now, career_future = None, None

    if job_id and job_id in jobs:
        selected_job = jobs[job_id]
        ensure_model()
        fit_prob, feat_dict = predict_fit(cv_feats, selected_job)  # 0..1
        fit_output = {
            "probability": round(float(fit_prob), 4),
            "features": feat_dict,
        }

        # Hybrid score: 0.7 * model_proba + 0.3 * mandatory_coverage_ratio
        coverage = float(feat_dict.get("mandatory_coverage_ratio", 0.0))
        hybrid = 0.7 * fit_prob + 0.3 * coverage
        hybrid_score = round(hybrid * 100.0, 1)

        # Missing mandatory skills
        mand = set(map(str.lower, selected_job.get("mandatory", [])))
        have = set(map(str.lower, cv_feats["skills_found"]))
        missing_mandatory = sorted(list(mand - have))

        # SHAP (optional)
        if show_shap:
            shap_top = explain_top_k(cv_feats, selected_job, k=3)
        
        # Career stage and trajectory
        try:
            ensure_career_model()
            career_now = predict_current_stage(cv_feats, selected_job)
            career_future = predict_future_trajectory(cv_feats, selected_job, years_ahead=(2,5))
        except Exception as e:
            flash(f'Career prediction failed: {e}', 'error')

    return render_template(
        "result.html",
        cv=cv_feats,
        base_score=base_score,
        fit_output=fit_output,
        shap_top=shap_top,
        job=selected_job,
        missing_mandatory=missing_mandatory,
        llm=llm,
        used_llm=use_llm,
        used_shap=show_shap,
        hybrid_score=hybrid_score,
        career_now=career_now,
        career_future=career_future,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
