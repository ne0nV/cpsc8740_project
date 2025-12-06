# Recommender App Scaffold

This repo implements the **AI Tools Setup Summary**: Python 3.11, Jupyter, pandas/numpy, scikit-learn, optional collaborative filtering (Implicit + LightFM or either as available), PyTorch (optional), a **FastAPI** service with **CORS**, SQLite/Parquet-friendly IO, and a tiny static web page that calls the API locally.

> If you're on Apple Silicon and see build issues for `lightfm` or `implicit`, install via conda-forge _or_ skip them initially and use the baseline model. (They are in `requirements-optional.txt`.)

## Quickstart (pip + venv)

```bash
# 1) Python 3.11 recommended
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install core requirements
pip install --upgrade pip wheel
pip install -r requirements-core.txt

# 3) (Optional) If you want LightFM / Implicit and have C build tools set up:
pip install -r requirements-optional.txt  # may require a compiler

# 4) Run a sanity notebook (Jupyter will open in browser)
jupyter notebook
# open notebooks/00_sanity_check.ipynb and run all cells

# 5) (Optional) Train baseline and save artifacts
python scripts/train_baseline.py

# 6) Start the API
uvicorn api.main:app --reload --port 8000

# 7) Open the tiny static page (no npm needed)
# In another terminal:
python -m http.server 5173 -d web
# Then visit http://localhost:5173/static/index.html
```

## Conda (recommended for Apple Silicon)

```bash
conda create -n recoapp python=3.11
conda activate recoapp
pip install -r requirements-core.txt

# Optional (may be easier with conda-forge):
# conda install -c conda-forge lightfm
# conda install -c conda-forge implicit

# Then continue with steps 4–7 above.
```

## Notes
- **Artifacts** (trained models / popularity tables) are stored in `./artifacts/` and referenced via `src/utils/config.py`.
- **CORS** is enabled for `http://localhost:5173` by default, matching the tiny static page port.
- A small **synthetic dataset** is included in `data/sample_interactions.csv` for end-to-end sanity.
- For a full dataset, the scripts include a MovieLens-100k downloader; run: `python scripts/download_movielens.py`.
- If LightFM or Implicit isn't available, the API gracefully falls back to **popularity baseline** with cold-start handling.
- PyTorch is optional (kept as a future path).

## Project layout

```
.
├── api/
│   ├── main.py                 # FastAPI app + CORS
│   └── recommender_service.py  # Loads artifacts, serves recommendations
├── artifacts/                  # Saved models (created after training)
├── data/
│   └── sample_interactions.csv # tiny synthetic dataset
├── notebooks/
│   ├── 00_sanity_check.ipynb
│   └── 01_train_baselines.ipynb
├── scripts/
│   ├── train_baseline.py
│   └── download_movielens.py
├── src/
│   ├── models/
│   │   ├── baselines.py
│   │   ├── implicit_als.py
│   │   ├── lightfm_model.py
│   │   └── metrics.py
│   └── utils/
│       ├── config.py
│       └── io.py
├── tests/
│   └── test_metrics.py
├── web/
│   └── static/index.html       # Calls the API without a JS build step
├── .env.example
├── Makefile
├── requirements-core.txt
├── requirements-optional.txt
└── README.md
```

## React + Vite (optional)
If you prefer a React app instead of the tiny static page, scaffold it next to this folder:
```bash
npm create vite@latest web -- --template react
cd web
npm install
npm run dev  # defaults to port 5173
```
Ensure CORS in `api/main.py` matches the origin (http://localhost:5173).
