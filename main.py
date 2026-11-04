import os
import io
import json
import xgboost as xgb
import sqlite3
import asyncio
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our core pipeline modules (Chain: news -> generator -> evaluator)
from generator import generate_ad_variations
from evaluator import (
    evaluate_ads,
    extract_metadata_for_custom_ad,
    embedding_model,
    XGB_PATH,
    ANN_PATH
)

load_dotenv()

MLOPS_DB = "mlops_master.db"

# =====================================================================
# 1. AUTOPILOT DEMOGRAPHIC -> INTEREST ENGINE (From Kaggle CSV Analysis)
# =====================================================================
# Mapped from interest_gender_age_breakdown.csv (1,143 records across 40 interests)
AUTOPILOT_INTEREST_MAP = {
    "M": {
        "30-34": {"primary": 20, "secondary": 36, "note": "Interest 20 (63.3% Male, 12 records in 30-34) & Interest 36 (85.7% Male dominant)"},
        "35-39": {"primary": 29, "secondary": 23, "note": "Interest 29 (58.4% Male, 11 records in 35-39) & Interest 23 (65.2% Male dominant)"},
        "40-44": {"primary": 20, "secondary": 2,  "note": "Interest 20 (63.3% Male) & Interest 2 (68.0% Male dominant, peak in 40-44)"},
        "45-49": {"primary": 20, "secondary": 29, "note": "Interest 20 (63.3% Male, 8 records in 45-49) & Interest 29 (9 records)"}
    },
    "F": {
        "30-34": {"primary": 27, "secondary": 64, "note": "Interest 27 (61.7% Female, 12 records in 30-34) & Interest 64 (56.2% Female, 12 records)"},
        "35-39": {"primary": 27, "secondary": 63, "note": "Interest 27 (61.7% Female, 9 records in 35-39) & Interest 63 (60.9% Female, 9 records)"},
        "40-44": {"primary": 27, "secondary": 10, "note": "Interest 27 (61.7% Female, 8 records in 40-44) & Interest 10 (12 Female records)"},
        "45-49": {"primary": 27, "secondary": 26, "note": "Interest 27 (61.7% Female, 8 records in 45-49) & Interest 26 (56.1% Female, 8 records)"}
    },
    "BOTH": {
        "30-34": {"primary": 16, "secondary": 10, "note": "Interest 16 (Balanced 51.4% M / 48.6% F, 47 total records in 30-34)"},
        "35-39": {"primary": 16, "secondary": 10, "note": "Interest 16 (Balanced 51.4% M / 48.6% F, 30 total records in 35-39)"},
        "40-44": {"primary": 16, "secondary": 10, "note": "Interest 16 (Balanced 51.4% M / 48.6% F, 28 total records in 40-44)"},
        "45-49": {"primary": 16, "secondary": 10, "note": "Interest 16 (Balanced 51.4% M / 48.6% F, 35 total records in 45-49)"}
    }
}

def resolve_autopilot_audience(age: str, gender: str, daily_budget: float = 50.0):
    """Automatically selects the strongest interest ID based on Gender and Age Group."""
    clean_gender = gender.strip().upper()
    if clean_gender not in ["M", "F", "BOTH"]:
        clean_gender = "BOTH"
        
    clean_age = age.strip()
    if clean_age not in ["30-34", "35-39", "40-44", "45-49"]:
        clean_age = "30-34"
        
    mapping = AUTOPILOT_INTEREST_MAP[clean_gender][clean_age]
    return {
        "age_encoded": clean_age,
        "gender_encoded": clean_gender,
        "interest": mapping["primary"],
        "secondary_interest": mapping["secondary"],
        "spent": float(daily_budget),
        "autopilot_explanation": mapping["note"]
    }

# =====================================================================
# 2. SEASONAL TAGGING & UNIFIED MLOps SQLITE DATABASE
# =====================================================================
SEASONAL_KEYWORDS = {
    "Diwali": ["diwali", "deepavali", "festive", "dhanteras"],
    "Valentine": ["valentine", "love", "couple", "february"],
    "IPL": ["ipl", "cricket", "t20", "match", "stadium"],
    "Holi": ["holi", "colors", "gulal", "spring"],
    "Independence_Day": ["independence", "15 august", "freedom", "tricolor"]
}

SEASON_MONTH_MAP = {
    10: "Diwali", 11: "Diwali",
    2: "Valentine",
    3: "Holi",
    4: "IPL", 5: "IPL",
    8: "Independence_Day"
}

def detect_season_tag(ad_text: str) -> str:
    text_lower = str(ad_text).lower()
    for season, kw_list in SEASONAL_KEYWORDS.items():
        if any(kw in text_lower for kw in kw_list):
            return season
    return "None"

def init_mlops_db():
    conn = sqlite3.connect(MLOPS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS AdPerformanceMaster (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ad_text TEXT NOT NULL,
        age TEXT DEFAULT '30-34',
        gender TEXT DEFAULT 'M',
        interest INTEGER DEFAULT 16,
        spent REAL DEFAULT 50.0,
        word_count INTEGER DEFAULT 12,
        urgency_score REAL DEFAULT 0.5,
        entity_power_score REAL DEFAULT 0.0,
        actual_ctr REAL NOT NULL,
        source TEXT NOT NULL,
        season_tag TEXT DEFAULT 'None',
        created_at DATETIME NOT NULL
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS SyncAndCronLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trigger_type TEXT,
        source TEXT,
        status TEXT,
        records_ingested INTEGER,
        message TEXT,
        timestamp DATETIME
    )''')
    conn.commit()
    conn.close()

def log_sync_event(trigger_type: str, source: str, status: str, count: int, message: str):
    conn = sqlite3.connect(MLOPS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO SyncAndCronLogs (trigger_type, source, status, records_ingested, message, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (trigger_type, source, status, count, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# =====================================================================
# 3. CRON-JOB ENGINE ("WAITING FOR API") & META ADS SYNC LOGIC
# =====================================================================

CRON_STATE = {
    "cron_enabled": True,
    "schedule": "Every 7 Days at 00:00 (Midnight Sunday)",
    "next_scheduled_run": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d 00:00:00"),
    "last_checked": None,
    "engine_status": "AWAITING_API_INTEGRATION"
}

def execute_meta_ads_sync(trigger_type: str = "MANUAL_REFRESH", simulate_if_no_fuel: bool = False):
    """
    Checks for META_ADS_ACCESS_TOKEN ('Fuel').
    If fuel is present -> Calls real Meta Graph API.
    If fuel is missing -> Safely stands by (or simulates pull if requested for demo).
    """
    meta_token = os.getenv("META_ADS_ACCESS_TOKEN", "").strip()
    meta_account_id = os.getenv("META_AD_ACCOUNT_ID", "").strip()
    CRON_STATE["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CASE 1: REAL FUEL IS PRESENT IN .ENV -> HIT META GRAPH API
    if meta_token and meta_account_id:
        CRON_STATE["engine_status"] = "ACTIVE_FUELED"
        try:
            url = f"https://graph.facebook.com/v19.0/act_{meta_account_id}/insights"
            params = {
                "access_token": meta_token,
                "date_preset": "last_7d",
                "fields": "ad_name,spend,impressions,clicks,ctr"
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw_data = resp.json().get("data", [])
            
            conn = sqlite3.connect(MLOPS_DB)
            ingested = 0
            for item in raw_data:
                ad_copy = item.get("ad_name", "Meta Ad Campaign")
                meta_info = extract_metadata_for_custom_ad(ad_copy)
                s_tag = detect_season_tag(ad_copy)
                conn.execute('''INSERT INTO AdPerformanceMaster 
                    (ad_text, age, gender, interest, spent, word_count, urgency_score, entity_power_score, actual_ctr, source, season_tag, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (ad_copy, '30-34', 'M', 16, float(item.get("spend", 50.0)),
                     meta_info["word_count"], meta_info["urgency_score"], meta_info["entity_power_score"],
                     float(item.get("ctr", 1.5)), "META_ADS_API", s_tag, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                ingested += 1
            conn.commit()
            conn.close()
            
            msg = "Pending Integration: META_ADS_ACCESS_TOKEN not found in .env. Please configure the token to activate live 7-day synchronization."
            log_sync_event(trigger_type, "META_ADS_API", "STANDBY_NO_TOKEN", 0, msg)
            return {"status": "SUCCESS", "fuel_present": True, "records_added": ingested, "message": msg}
        except Exception as e:
            msg = f"Meta API Fuel present, but Graph API returned error: {e}"
            log_sync_event(trigger_type, "META_ADS_API", "ERROR", 0, msg)
            return {"status": "ERROR", "fuel_present": True, "records_added": 0, "message": msg}

    # CASE 2: NO FUEL IN .ENV -> DEMO SIMULATION OR GRACEFUL STANDBY
    CRON_STATE["engine_status"] = "ARMED_AWAITING_FUEL"
    if simulate_if_no_fuel:
        # Insert 3 realistic mock Meta Ads records to demonstrate dual-source harmony in UI
        mock_meta_ads = [
            ("This Diwali smell like SRK! Grab your luxury perfume combo at Rs 999 today.", "30-34", "M", 20, 65.0, 4.85, "Diwali"),
            ("Upgrade to smart electric mobility before festive EV offers end tonight. Book now!", "35-39", "M", 29, 80.0, 3.40, "Diwali"),
            ("Exclusive skincare festive glow kit for her. Claim your 40% discount today!", "30-34", "F", 27, 45.0, 3.95, "Diwali")
        ]
        conn = sqlite3.connect(MLOPS_DB)
        for text, age, gen, intr, sp, ctr, stag in mock_meta_ads:
            meta_info = extract_metadata_for_custom_ad(text)
            conn.execute('''INSERT INTO AdPerformanceMaster 
                (ad_text, age, gender, interest, spent, word_count, urgency_score, entity_power_score, actual_ctr, source, season_tag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (text, age, gen, intr, sp, meta_info["word_count"], meta_info["urgency_score"],
                 meta_info["entity_power_score"], ctr, "META_ADS_API_SIMULATED", stag, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        msg = "No META_ADS_ACCESS_TOKEN in .env (Demo Mode): Simulated 7-day Meta Ads API pull of 3 campaign records into Master DB."
        log_sync_event(trigger_type, "META_ADS_API_SIMULATED", "SIMULATED_SUCCESS", 3, msg)
        return {"status": "SIMULATED_SUCCESS", "fuel_present": False, "records_added": 3, "message": msg, "cron_state": CRON_STATE}

    msg = "Engine Armed & Waiting for Fuel: META_ADS_ACCESS_TOKEN not found in .env. Add token to activate live 7-day pulls."
    log_sync_event(trigger_type, "META_ADS_API", "STANDBY_NO_FUEL", 0, msg)
    return {"status": "STANDBY_NO_FUEL", "fuel_present": False, "records_added": 0, "message": msg, "cron_state": CRON_STATE}

async def weekly_cron_worker():
    """Background daemon that wakes up and checks if 7-day Meta Ads sync should run."""
    # Run an initial check on startup so logs show the Cron Daemon is alive
    execute_meta_ads_sync(trigger_type="CRON_STARTUP_CHECK", simulate_if_no_fuel=False)
    while True:
        await asyncio.sleep(7 * 24 * 3600)  # 7 Days interval
        execute_meta_ads_sync(trigger_type="SCHEDULED_7DAY_CRON", simulate_if_no_fuel=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_mlops_db()
    cron_task = asyncio.create_task(weekly_cron_worker())
    yield
    cron_task.cancel()

# =====================================================================
# 4. FASTAPI APPLICATION & ENDPOINTS
# =====================================================================
app = FastAPI(
    title="Ad-Optima Autonomous AI Backend",
    description="End-to-End Hybrid Ad Generation, Evaluation (XGBoost + ANN + NER), Autopilot Demographics, and Dual-Source MLOps Engine.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schemas ---
class GenerateCampaignRequest(BaseModel):
    product_keyword: str = Field(..., example="SUV")
    target_age: Literal["30-34", "35-39", "40-44", "45-49"] = Field(default="30-34")
    target_gender: Literal["M", "F", "BOTH"] = Field(default="M")
    daily_budget: float = Field(default=50.0, ge=1.0)
    num_ads: int = Field(default=3, ge=1, le=15)
    force_refresh_news: bool = Field(default=False)

class CustomAdEvaluateRequest(BaseModel):
    custom_ad_text: str = Field(..., example="This Diwali smell like SRK grab your perfume combo at just Rs 999")
    target_age: Literal["30-34", "35-39", "40-44", "45-49"] = Field(default="30-34")
    target_gender: Literal["M", "F", "BOTH"] = Field(default="M")
    daily_budget: float = Field(default=50.0, ge=1.0)

# --- Endpoint 2: Chain Execution (News -> Generator -> Evaluator) ---
@app.post("/api/campaigns/generate-and-evaluate", tags=["1. Ad Generation & Evaluation"])
async def generate_and_evaluate_campaign(req: GenerateCampaignRequest):
    """
    Full Chain:
    1. Resolves Autopilot Interest from (Age, Gender).
    2. Calls news.py + generator.py to create high-converting ads.
    3. Calls evaluator.py (XGBoost + ANN + NER + LLM Reasoning) to rank and explain ads.
    """
    autopilot_aud = resolve_autopilot_audience(req.target_age, req.target_gender, req.daily_budget)
    aud_desc = f"Age {req.target_age}, Gender {req.target_gender}, Budget ${req.daily_budget}"
    
    ai_ads, model_used = generate_ad_variations(
        product_keyword=req.product_keyword,
        target_audience=aud_desc,
        num_ads=req.num_ads,
        force_refresh=req.force_refresh_news
    )
    
    if not ai_ads:
        raise HTTPException(status_code=500, detail="Ad generation failed. Check API keys.")
        
    ranked_ads = evaluate_ads(ai_ads, aud_desc, autopilot_aud)
    
    # Format clean UI-ready response (hide empty entity lists if none detected)
    for ad in ranked_ads:
        entities = ad.get("brand_or_celeb_detected", [])
        ad["ui_entity_badge"] = f"⭐ Star/Brand Trust Boost: {', '.join(entities)}" if entities else None

    return {
        "status": "success",
        "llm_router_used": model_used,
        "autopilot_config": autopilot_aud,
        "total_generated": len(ranked_ads),
        "top_3_winners": ranked_ads[:3],
        "all_ranked_ads": ranked_ads
    }

# --- Endpoint 3: Custom Ad Tester (User Type 2) ---
@app.post("/api/campaigns/evaluate-custom", tags=["1. Ad Generation & Evaluation"])
async def evaluate_custom_ad_endpoint(req: CustomAdEvaluateRequest):
    autopilot_aud = resolve_autopilot_audience(req.target_age, req.target_gender, req.daily_budget)
    aud_desc = f"Age {req.target_age}, Gender {req.target_gender}, Budget ${req.daily_budget}"
    
    ad_metadata = extract_metadata_for_custom_ad(req.custom_ad_text)
    ranked = evaluate_ads([ad_metadata], aud_desc, autopilot_aud)
    res = ranked[0]
    
    entities = res.get("brand_or_celeb_detected", [])
    res["ui_entity_badge"] = f"⭐ Star/Brand Trust Boost: {', '.join(entities)}" if entities else None
    
    return {
        "status": "success",
        "autopilot_config": autopilot_aud,
        "evaluation": res
    }

# --- Endpoint 4: Survey / Offline CSV Ingestion ---
@app.post("/api/mlops/upload-survey-csv", tags=["2. MLOps & Feedback Loop"])
async def upload_survey_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        required_cols = ["ad_text", "actual_ctr"]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column in CSV: '{col}'")
        
        conn = sqlite3.connect(MLOPS_DB)
        ingested = 0
        for _, row in df.iterrows():
            ad_text = str(row["ad_text"])
            meta = extract_metadata_for_custom_ad(ad_text)
            s_tag = detect_season_tag(ad_text)
            
            conn.execute('''INSERT INTO AdPerformanceMaster 
                (ad_text, age, gender, interest, spent, word_count, urgency_score, entity_power_score, actual_ctr, source, season_tag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    ad_text,
                    str(row.get("age", "30-34")),
                    str(row.get("gender", "M")),
                    int(row.get("interest", 16)),
                    float(row.get("spent", 50.0)),
                    meta["word_count"],
                    meta["urgency_score"],
                    meta["entity_power_score"],
                    float(row["actual_ctr"]),
                    "SURVEY_CSV_UPLOAD",
                    s_tag,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            ingested += 1
        conn.commit()
        conn.close()
        
        msg = f"Successfully ingested {ingested} records from survey CSV into Master DB!"
        log_sync_event("MANUAL_CSV_UPLOAD", "CSV_FILE", "SUCCESS", ingested, msg)
        return {"status": "SUCCESS", "records_ingested": ingested, "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {e}")

# --- Endpoint 5: Manual Meta-Sync / Fuel Check Trigger ---
@app.post("/api/mlops/meta-sync", tags=["2. MLOps & Feedback Loop"])
async def trigger_meta_sync(simulate_if_no_fuel: bool = Query(True)):
    result = execute_meta_ads_sync(trigger_type="MANUAL_API_TRIGGER", simulate_if_no_fuel=simulate_if_no_fuel)
    return result

# --- Endpoint 6: Seasonal-Protected Retraining with Time Decay ---
@app.post("/api/mlops/retrain", tags=["2. MLOps & Feedback Loop"])
async def retrain_models_endpoint():
    conn = sqlite3.connect(MLOPS_DB)
    df = pd.read_sql("SELECT * FROM AdPerformanceMaster", conn)
    conn.close()
    
    if len(df) < 5:
        return {"status": "SKIPPED", "message": "Not enough training records in Master DB yet. Upload more data or run sync."}
    
    now = datetime.now()
    weights = []
    current_month = now.month
    
    for _, row in df.iterrows():
        created = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
        delta_days = (now - created).days
        
        base_weight = np.exp(-0.015 * delta_days)
        
        stag = row.get("season_tag", "None")
        if stag != "None":
            is_active_season = False
            if stag == "Diwali" and current_month in [10, 11]: is_active_season = True
            elif stag == "Valentine" and current_month == 2: is_active_season = True
            elif stag == "Holi" and current_month == 3: is_active_season = True
            elif stag == "IPL" and current_month in [4, 5]: is_active_season = True
            elif stag == "Independence_Day" and current_month == 8: is_active_season = True
            
            if is_active_season:
                base_weight = 1.0
            else:
                base_weight = max(base_weight, 0.35)
                
        weights.append(max(base_weight, 0.05))
        
    weights = np.array(weights)
    
    xgb_trained_status = False
    try:
        X_xgb = pd.DataFrame({
            'age': df['age'].astype('category'),     
            'gender': df['gender'].astype('category'),
            'interest': df['interest'],
            'Spent': df['spent']
        })
        y_xgb = df['actual_ctr']
        
        # enable_categorical=True zaroori hai strings ko train karne ke liye
        xgb_model_new = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, enable_categorical=True)
        xgb_model_new.fit(X_xgb, y_xgb, sample_weight=weights)
        joblib.dump(xgb_model_new, XGB_PATH)
        xgb_trained_status = True
    except Exception as e:
        print(f"[-] XGBoost Retrain Error: {e}")
        
    ann_trained_status = False
    try:
        import tensorflow as tf
        headlines = df['ad_text'].tolist()
        embeddings = list(embedding_model.embed(headlines))
        X_ann = np.array(embeddings)
        y_ann = df['actual_ctr'].values
        
        ann_model_new = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_ann.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        ann_model_new.compile(optimizer='adam', loss='mse')
        ann_model_new.fit(X_ann, y_ann, sample_weight=weights, epochs=10, batch_size=16, verbose=0)
        ann_model_new.save(ANN_PATH)
        ann_trained_status = True
    except Exception as e:
        print(f"[-] ANN Retrain Error: {e}")
        
    msg = f"Retraining complete! Records processed: {len(df)}. XGBoost Retrained: {xgb_trained_status}, ANN Retrained: {ann_trained_status}."
    log_sync_event("MANUAL_RETRAIN", "ML_ENGINE", "SUCCESS", len(df), msg)
    return {
        "status": "SUCCESS",
        "total_records_trained": len(df),
        "xgb_retrained": xgb_trained_status,
        "ann_retrained": ann_trained_status,
        "message": msg
    }

# --- Endpoint 7: MLOps Logs & System Status ---
@app.get("/api/mlops/status", tags=["2. MLOps & Feedback Loop"])
async def get_mlops_status():
    conn = sqlite3.connect(MLOPS_DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM AdPerformanceMaster")
    total_records = c.fetchone()[0]
    
    c.execute("SELECT * FROM SyncAndCronLogs ORDER BY id DESC LIMIT 10")
    logs = c.fetchall()
    conn.close()
    
    return {
        "cron_state": CRON_STATE,
        "total_master_records": total_records,
        "recent_logs": [
            {
                "id": r[0], "trigger_type": r[1], "source": r[2],
                "status": r[3], "records_ingested": r[4], "message": r[5], "timestamp": r[6]
            } for r in logs
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)