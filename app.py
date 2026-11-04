import os
import sys
import time
import socket
import subprocess
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

BRAND_SVG_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#FF6B00"/>
  <path d="M32 14L48 48H39.5L36 40H28L24.5 48H16L32 14ZM33.5 33L32 28.5L30.5 33H33.5Z" fill="#FFFFFF"/>
</svg>
"""

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def ensure_backend_running():
    if not is_port_in_use(8000):
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            if is_port_in_use(8000):
                break
            time.sleep(0.5)

def inject_enterprise_css():
    st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; color: #0F172A; font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; max-width: 95% !important; }
        
        /* FIX: Header ko chhat se neeche laaya */
        .enterprise-header {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 6px solid #FF6B00;
            border-radius: 12px;
            padding: 24px 32px;
            margin-top: 40px !important; /* ROOF COLLISION FIXED */
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .enterprise-title { font-size: 2.4rem; font-weight: 800; color: #0F172A; margin: 0; }
        .enterprise-title span { color: #FF6B00; }
        .enterprise-subtitle { font-size: 1.15rem; color: #475569; margin-top: 8px; font-weight: 500; }
        
        div[data-baseweb="tab-list"] { gap: 32px; background-color: transparent; }
        button[data-baseweb="tab"] { font-size: 1.25rem !important; font-weight: 700 !important; padding: 12px 10px !important; color: #64748B !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #FF6B00 !important; border-bottom: 3px solid #FF6B00 !important; }
        
        /* FIX: Dropdown Options BIG and BOLD */
        label[data-testid="stWidgetLabel"] p { font-size: 1.1rem !important; font-weight: 700 !important; color: #1E293B !important; }
        div[data-baseweb="select"] { font-size: 1.15rem !important; font-weight: 700 !important; }
        div[data-baseweb="select"] span { font-size: 1.15rem !important; font-weight: 700 !important; color: #0F172A !important; }
        ul[data-baseweb="menu"] li { font-size: 1.15rem !important; font-weight: 600 !important; color: #0F172A !important; }
        input, textarea { font-size: 1.1rem !important; font-weight: 500 !important; }
        
        div.stButton > button { background-color: #FF6B00 !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; height: 3.5rem !important; font-size: 1.15rem !important; font-weight: 700 !important; }
        div.stButton > button:hover { background-color: #E05E00 !important; }
        
        .ad-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #FF6B00; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        .ad-rank-badge { display: inline-block; background: #FFF7ED; color: #EA580C; border: 1px solid #FED7AA; font-weight: 800; font-size: 0.95rem; padding: 6px 16px; border-radius: 6px; margin-bottom: 12px; }
        .ad-copy-text { font-size: 1.45rem; font-weight: 800; color: #0F172A; line-height: 1.5; margin: 12px 0; }
        
        .exec-summary-box { background: #F8FAFC; border-left: 4px solid #FF6B00; padding: 18px 24px; border-radius: 0 8px 8px 0; margin-top: 20px; font-size: 1.05rem; color: #334155; line-height: 1.6; border: 1px solid #E2E8F0; border-left: none; }
        
        /* FIX: Pure Blank Canvas Style */
        .blank-canvas { height: 480px; background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 900 !important; color: #EA580C !important; }
        div[data-testid="stMetricLabel"] p { font-size: 1rem !important; font-weight: 700 !important; color: #64748B !important; }
    </style>
    """, unsafe_allow_html=True)

def clean_badge_text(text: str) -> str:
    if not text: return ""
    return text.replace("", "").strip()

def trigger_auto_retrain():
    try:
        retrain_resp = requests.post(f"{API_BASE_URL}/api/mlops/retrain", timeout=60)
        if retrain_resp.status_code == 200: return retrain_resp.json()
        return {"status": "ERROR", "message": retrain_resp.text}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def render_ui():
    st.set_page_config(page_title="Ad-Optima | Commercial Ad Intelligence", page_icon=BRAND_SVG_ICON, layout="wide")
    ensure_backend_running()
    inject_enterprise_css()

    st.markdown("""
        <div class="enterprise-header">
            <h1 class="enterprise-title">Ad-<span>Optima</span></h1>
            <div class="enterprise-subtitle">
                Autonomous Creative Intelligence & Performance Prediction Suite
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1. Campaign Studio", "2. Custom Ad Evaluator", "3. Data Sync & Calibration"])

    # -----------------------------------------------------------------
    # TAB 1: CAMPAIGN STUDIO
    # -----------------------------------------------------------------
    with tab1:
        left_col, right_col = st.columns([1, 2.1], gap="large")
        with left_col:
            with st.container(border=True):
                st.subheader("Campaign Parameters")
                product = st.text_input("Product or Brand Name", placeholder="e.g., Electric SUV, Luxury Watch")
                
                c1, c2 = st.columns(2)
                age1 = c1.selectbox("Target Age Bracket", ["30-34", "35-39", "40-44", "45-49"], key="age1")
                gender1 = c2.selectbox("Target Segment", ["M", "F", "BOTH"], key="gen1")
                
                budget1 = st.number_input("Allocated Daily Spend ($)", min_value=10.0, value=50.0, step=10.0, key="bud1")
                num_ads = st.slider("Variations to Generate", min_value=1, max_value=5, value=3)
                force_ref = st.checkbox("Bypass Local Cache (Fetch Live Signals)", value=False)
                
                st.write("")
                gen_clicked = st.button("Generate & Rank Creatives", use_container_width=True)

        with right_col:
            if gen_clicked:
                if not product.strip():
                    st.warning("Please enter a valid Product or Brand Name.")
                else:
                    with st.spinner(f"Synthesizing market context for '{product}'..."):
                        try:
                            payload = {"product_keyword": product.strip(), "target_age": age1, "target_gender": gender1, "daily_budget": budget1, "num_ads": num_ads, "force_refresh_news": force_ref}
                            resp = requests.post(f"{API_BASE_URL}/api/campaigns/generate-and-evaluate", json=payload, timeout=90)
                            if resp.status_code == 200:
                                data = resp.json()
                                st.subheader(f"Ranked Variations ({data.get('total_generated', 0)} Evaluated)")
                                for idx, ad in enumerate(data.get("all_ranked_ads", []), 1):
                                    st.markdown(f"""
                                        <div class="ad-card">
                                            <div class="ad-rank-badge">PRIORITY RANK #{idx} &nbsp;|&nbsp; FORECASTED CTR: {ad['predicted_ctr']}%</div>
                                            <div class="ad-copy-text">"{ad['headline']}"</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Audience Fit", f"{ad['aud_base_ctr']}%")
                                    m2.metric("Copy Engagement", f"{ad['text_base_ctr']}%")
                                    m3.metric("Urgency Index", f"{ad['urgency_score']}")
                                    
                                    pills_html = ""
                                    if ad.get("trend_angle_used"): pills_html += f'<span class="tag-pill">Market Signal: {ad["trend_angle_used"]}</span>'
                                    if ad.get("ui_entity_badge"): pills_html += f'<span class="tag-pill">{clean_badge_text(ad["ui_entity_badge"])}</span>'
                                    if pills_html: st.markdown(pills_html, unsafe_allow_html=True)
                                        
                                    if ad.get("llm_reasoning"):
                                        st.markdown(f'<div class="exec-summary-box"><strong>Executive Optimization Notes:</strong><br>{ad["llm_reasoning"]}</div>', unsafe_allow_html=True)
                                    st.divider()
                            else:
                                st.error(f"Service Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
            else:
                st.markdown("<div class='blank-canvas'></div>", unsafe_allow_html=True)
    # -----------------------------------------------------------------
    # TAB 2: CUSTOM AD EVALUATOR
    # -----------------------------------------------------------------
    with tab2:
        left_col2, right_col2 = st.columns([1, 1.4], gap="large")
        with left_col2:
            with st.container(border=True):
                st.subheader("Custom Creative Audit")
                custom_text = st.text_area("Draft Ad Copy", height=130, placeholder="Enter draft headline...")
                
                c3, c4 = st.columns(2)
                age2 = c3.selectbox("Target Age Bracket", ["30-34", "35-39", "40-44", "45-49"], key="age2")
                gender2 = c4.selectbox("Target Segment", ["M", "F", "BOTH"], key="gen2")
                budget2 = st.number_input("Allocated Daily Spend ($)", min_value=10.0, value=50.0, step=10.0, key="bud2")
                
                st.write("")
                audit_clicked = st.button("Audit Creative Performance", use_container_width=True)

        with right_col2:
            if audit_clicked:
                if not custom_text.strip():
                    st.warning("Please provide draft ad copy.")
                else:
                    with st.spinner("Auditing compliance and click-through probability..."):
                        try:
                            payload = {"custom_ad_text": custom_text.strip(), "target_age": age2, "target_gender": gender2, "daily_budget": budget2}
                            resp = requests.post(f"{API_BASE_URL}/api/campaigns/evaluate-custom", json=payload, timeout=45)
                            if resp.status_code == 200:
                                res = resp.json()["evaluation"]
                                sentiment = res.get("sentiment_tag", "Neutral")
                                
                                st.subheader("Audit Report")
                                if sentiment == "Negative/Provocative":
                                    st.markdown(f'<div class="status-banner-alert">BRAND SAFETY ALERT: Hostile phrasing detected. Brand multipliers revoked (0.0).</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="status-banner-safe">BRAND SAFE: Tone aligns with commercial guidelines.</div>', unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="ad-card">
                                        <div class="ad-rank-badge">FORECASTED CTR: {res['predicted_ctr']}%</div>
                                        <div class="ad-copy-text">"{res['headline']}"</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                m1, m2, m3 = st.columns(3)
                                m1.metric("Audience Fit", f"{res['aud_base_ctr']}%")
                                m2.metric("Copy Engagement", f"{res['text_base_ctr']}%")
                                m3.metric("Urgency Index", f"{res['urgency_score']}")

                                if res.get("ui_entity_badge") and sentiment != "Negative/Provocative":
                                    st.markdown(f'<span class="tag-pill">{clean_badge_text(res["ui_entity_badge"])}</span>', unsafe_allow_html=True)

                                st.markdown(f'<div class="exec-summary-box"><strong>Executive Optimization Notes:</strong><br>{res.get("llm_reasoning", "")}</div>', unsafe_allow_html=True)
                            else:
                                st.error(f"Audit Service Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Audit request failed: {e}")
            else:
                st.markdown("<div class='blank-canvas'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 3: DATA SYNC & MODEL CALIBRATION
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("Data Ingestion & Pipeline Calibration")
        st.write("Ensure your optimization models remain accurate by feeding recent performance data. Once new data is ingested via API or CSV, the models automatically retrain to capture shifting market trends.")
        st.write("")

        col_meta, col_csv = st.columns(2, gap="large")

        # Pipeline 1: Meta Ads Connection
        with col_meta:
            with st.container(border=True):
                st.subheader("1. Meta Ads Graph API")
                st.write("Automatically pull campaign telemetry from a connected Meta Business account.")
                st.info("Status: API Keys Not Configured. Please integrate your META environment variables in the .env file to enable live synchronization.")
                sim_mode = st.checkbox("Simulate Meta API Payload (Demo Mode)", value=True)
                
                st.write("")
                if st.button("Sync Data & Calibrate", use_container_width=True):
                    with st.spinner("Step 1/2: Connecting to Meta API and ingesting data..."):
                        try:
                            sync_res = requests.post(f"{API_BASE_URL}/api/mlops/meta-sync", params={"simulate_if_no_fuel": sim_mode}, timeout=30)
                            if sync_res.status_code == 200:
                                st.success("Telemetry Ingestion Successful: Data synced to Master DB.")
                                with st.spinner("Step 2/2: Recalibrating models with fresh market trends..."):
                                    ret_json = trigger_auto_retrain()
                                    if ret_json.get("status") == "SUCCESS":
                                        st.success("Calibration Complete: Models successfully retrained on new data.")
                                    else:
                                        st.warning("Calibration skipped: Insufficient new data.")
                            else:
                                st.error("Sync Error: Connection failed.")
                        except Exception as e:
                            st.error(f"Execution failed: {e}")

        # Pipeline 2: CSV Upload
        with col_csv:
            with st.container(border=True):
                st.subheader("2. Offline Campaign CSV")
                st.write("Upload historical campaign exports. Ensure your CSV contains 'ad_text' and 'actual_ctr' columns.")
                st.write("")
                
                uploaded_csv = st.file_uploader("Select Campaign Performance CSV", type=["csv"], label_visibility="collapsed")
                
                if st.button("Ingest CSV & Calibrate", use_container_width=True):
                    if uploaded_csv is None:
                        st.warning("Please select a valid CSV file.")
                    else:
                        with st.spinner("Step 1/2: Extracting NLP metadata and appending to Warehouse..."):
                            try:
                                files = {"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")}
                                up_res = requests.post(f"{API_BASE_URL}/api/mlops/upload-survey-csv", files=files, timeout=60)
                                if up_res.status_code == 200:
                                    st.success("CSV Ingestion Successful: Records added to Master DB.")
                                    with st.spinner("Step 2/2: Recalibrating models with fresh market trends..."):
                                        ret_json = trigger_auto_retrain()
                                        if ret_json.get("status") == "SUCCESS":
                                            st.success("Calibration Complete: Models successfully retrained on new data.")
                                        else:
                                            st.warning("Calibration skipped: Insufficient new data.")
                                else:
                                    st.error("Upload Failed: Check CSV format.")
                            except Exception as e:
                                st.error(f"Ingestion error: {e}")

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if not in_streamlit:
        ensure_backend_running()
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__]))
    else:
        render_ui()