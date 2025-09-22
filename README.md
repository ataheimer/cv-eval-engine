# CV Evaluation Engine (Flask + Gradient Boosting + Career Prediction)

## Overview
This project is a **CV Evaluation Engine** designed as a take-home assignment demo.  
It evaluates candidate CVs, scores them against job postings, and provides insights using **gradient boosting models**.  
Additionally, it includes an **innovative career trajectory predictor** that estimates the candidate’s current level (junior, mid, senior, etc.) and projects future growth.

Main features:
- CV text extraction (PDF/TXT).
- Basic scoring (skills, experience, education).
- Job fit probability using **XGBoost/LightGBM/CatBoost**.
- SHAP explainability (top-k feature contributions).
- Hybrid score (weighted combination of model probability + coverage).
- Career stage estimation and **future trajectory prediction**.
- **Optional LLM-based enrichment** (via [Ollama](https://ollama.ai/)) to infer missing skills, education, or experience from CV text.

---

## Tech Stack
- **Backend**: Flask  
- **ML Models**: XGBoost, LightGBM, CatBoost  
- **Explainability**: SHAP  
- **Frontend**: HTML, CSS (custom minimal UI)  
- **LLM Enrichment**: Ollama (local inference with models like `llama2`, `mistral`, `gemma`)  
- **Data**: Weakly-supervised CV–Job pairs generated from job descriptions  

---

## Installation

### 1. Clone and Environment
```bash
git clone https://github.com/yourname/cv-eval-engine-flask.git
cd cv-eval-engine-flask

# Create and activate virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install Ollama for LLM Enrichment

LLM-based enrichment is optional but recommended to infer missing skills, education, and experience from CV text.

* Download Ollama from: [https://ollama.ai/download](https://ollama.ai/download)
* After installing, pull a model (example: `llama2`):

```bash
ollama pull llama2
```

* Verify it works:

```bash
ollama run llama2 "Hello, world!"
```

### 4. Pre-train the Base Models

```bash
python - <<'PY'
from modules.models import ensure_model
from modules.career import ensure_career_model
ensure_model()
ensure_career_model()
PY
```

---

## Usage

Run the Flask app:

```bash
export FLASK_APP=app.py   # Windows: set FLASK_APP=app.py
flask run -p 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

Steps:

1. Upload a CV (`.pdf` or `.txt`).
2. Select a job posting (optional, enables fit prediction).
3. Optionally enable:

   * **LLM enrichment** (Ollama must be running).
   * **SHAP explanations**.
4. View results:

   * Base score (0–100).
   * Fit probability & hybrid score.
   * Missing mandatory skills.
   * SHAP explanations.
   * Career stage & future trajectory.

---

## Data & Models

* Job postings are defined in `data/jobs_sample.json`.
* If no labeled training data exists, the system generates **weakly supervised labels** for model training.
* Models supported:

  * **XGBoost** (default).
  * **LightGBM**.
  * **CatBoost**.
* Career predictor is a lightweight gradient boosting model estimating current and future levels.

---

## Example Output

* Base Score: `78/100`
* Fit Probability: `62.3%`
* Hybrid Score: `68.5%`
* Missing mandatory skills: `mlflow, airflow`
* Career trajectory:

  * Now → Mid-level (score 55)
  * +2 years → Senior (score 72)
  * +5 years → Expert (score 88)

---

## Docker Usage

### 1. Build the image

```bash
docker build -t cv-eval .
```

### 2. Run the container (with volumes and Ollama support)

#### PowerShell (Windows)

```powershell
docker run --name cv-eval -p 8000:8000 `
  -e OLLAMA_URL=http://host.docker.internal:11434 `
  -v "${PWD}\uploads:/app/uploads" `
  -v "${PWD}\models:/app/models" `
  -v "${PWD}\data:/app/data" `
  cv-eval
```

#### Git Bash / Linux / macOS

```bash
docker run --name cv-eval -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  cv-eval
```

### 3. Access the app

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Ollama for LLM Enrichment

If you want to enable **LLM-based enrichment**:

* Install Ollama on your host: [https://ollama.ai/download](https://ollama.ai/download)
* Pull a model (e.g., `llama2`):

  ```bash
  ollama pull llama2
  ```
* Ensure the container can reach Ollama. By default, the environment variable `OLLAMA_URL=http://host.docker.internal:11434` is passed.

### 5. Volumes

* Uploaded CVs → `./uploads/`
* Trained models → `./models/`
* Generated/parsed data → `./data/`

These directories are mounted as volumes so your data persists across container rebuilds.

### 6. Logs & Maintenance

* View logs:

  ```bash
  docker logs -f cv-eval
  ```
* Stop / Start:

  ```bash
  docker stop cv-eval
  docker start cv-eval
  ```
* Remove:

  ```bash
  docker rm -f cv-eval
  ```

---

## Future Work

* Improve dataset realism with actual CV-job matches.
* Add salary prediction and career path clustering.
* Expand LLM enrichment (parse GitHub links, publications).
* Enhance frontend with charts/visualizations.

---

## License

This project is provided for **educational and demonstration purposes only**.