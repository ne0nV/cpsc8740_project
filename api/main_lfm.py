from __future__ import annotations
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
from api.recommender_service_lfm import RecommenderServiceLFM

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="Reco API (LightFM-first)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_service = RecommenderServiceLFM()
_service.load()

@app.get("/health")
def health():
    return {"status": "ok", "lfm_loaded": bool(_service.lfm)}

@app.get("/recommend")
def recommend(user_id: Optional[int] = Query(None), k: int = Query(10, ge=1, le=50)) -> List[Dict[str, Any]]:
    return _service.recommend(user_id=user_id, k=k)
