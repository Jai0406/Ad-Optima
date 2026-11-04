import os
import json
import joblib
import gdown
import spacy
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
import xgboost as xgb
import tensorflow as tf
from generator import call_llm_router, generate_ad_variations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer() 

# 2. Setup & Load .env Variables
load_dotenv()
XGB_DRIVE_ID = os.getenv("XGB_DRIVE_ID")
ANN_DRIVE_ID = os.getenv("ANN_DRIVE_ID")
XGB_PATH = "xgb_audience_model.pkl"
ANN_PATH = "ann_text_model_v3.keras"

# 3. Load Embeddings & NER (Locally)
print("[*] Loading Text Embedding & NER Models...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 4. Download & Load ML Models
def download_models():
    if not XGB_DRIVE_ID or not ANN_DRIVE_ID:
        print("[-] Missing Drive IDs in .env file!")
        return
        
    if not os.path.exists(XGB_PATH):
        print("[*] Downloading XGBoost Model from Drive...")
        gdown.download(f'https://drive.google.com/uc?id={XGB_DRIVE_ID}', XGB_PATH, quiet=False)
    if not os.path.exists(ANN_PATH):
        print("[*] Downloading ANN Model from Drive...")
        gdown.download(f'https://drive.google.com/uc?id={ANN_DRIVE_ID}', ANN_PATH, quiet=False)

def load_ml_models():
    download_models()
    xgb_model = None
    ann_model = None
    try:
        xgb_model = joblib.load(XGB_PATH)
        ann_model = tf.keras.models.load_model(ANN_PATH) 
        print("[+] ML Models (XGBoost + Keras ANN) loaded successfully!")
    except Exception as e:
        print(f"[-] Could not load ML models (Using robust fallback for Demo): {e}")
    return xgb_model, ann_model

xgb_model, ann_model = load_ml_models()

# 5. Custom Ad Feature Extraction
def extract_metadata_for_custom_ad(ad_text):
    word_count = len(ad_text.split())
    
    # 1. VADER Sentiment Analysis (Detecting the Tone automatically)
    sentiment_score = analyzer.polarity_scores(ad_text)['compound']
    if sentiment_score >= 0.05:
        sentiment_tag = "Positive"
    elif sentiment_score <= -0.05:
        sentiment_tag = "Negative/Provocative"
    else:
        sentiment_tag = "Neutral"

    # 2. NER Extract (No Hardcoded Garbage Lists!)
    doc = nlp(ad_text) # Removed aggressive .title() to avoid fake capitalizing slangs
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "PRODUCT"]]
    
    if not entities:
        # Fallback just in case user wrote valid brand in small letters
        doc_title = nlp(ad_text.title())
        entities = [ent.text for ent in doc_title.ents if ent.label_ in ["PERSON", "ORG", "PRODUCT"]]
    
    # 3. THE SMART LINK (VADER overrides NER)
    # Agar Ad Negative/Insulting hai, toh hum kisi bhi entity ko "Brand Trust Boost" (0.5) NAHI denge.
    # Kyunki brand ko gaali dene se trust nahi badhta!
    if sentiment_tag == "Negative/Provocative":
        entity_power = 0.0
    else:
        entity_power = 0.5 if entities else 0.0 
        
    urgency_words = ["now", "today", "hurry", "limited", "fast", "offer", "sale", "ends"]
    urgency_score = 0.8 if any(w in ad_text.lower() for w in urgency_words) else 0.2
    
    return {
        "headline": ad_text,
        "word_count": word_count,
        "urgency_score": urgency_score,
        "humor_level": 0.1,
        "uses_live_trend": False,
        "trend_angle_used": "Custom User Copy",
        "brand_or_celeb_detected": entities,
        "entity_power_score": entity_power,
        "sentiment_tag": sentiment_tag,
        "sentiment_score": sentiment_score
    }

# 6. The Hybrid Predictor (The Math Formula)
def predict_final_ctr(ad_data, audience_data):
    try:
        if xgb_model:
            features = pd.DataFrame([{
                'age': audience_data['age_encoded'], 
                'gender': audience_data['gender_encoded'], 
                'interest': audience_data['interest'], 
                'Spent': audience_data['spent']
            }])
            ctr_aud = xgb_model.predict(features)[0]
        else:
            ctr_aud = 1.2 + (audience_data['spent'] * 0.005)
    except:
        ctr_aud = 1.5

    try:
        if ann_model:
            text_vector = list(embedding_model.embed([ad_data['headline']]))[0]
            ctr_text = ann_model.predict(np.array([text_vector]), verbose=0)[0][0]
        else:
            ctr_text = 2.0 + (ad_data['word_count'] * 0.05)
    except:
        ctr_text = 2.5

    base_ctr = (0.40 * ctr_aud) + (0.60 * ctr_text)
    final_ctr = base_ctr * (1 + (0.10 * ad_data['urgency_score']) + (0.15 * ad_data['entity_power_score']))
    
    return round(final_ctr, 3), round(ctr_aud, 3), round(ctr_text, 3)

# 7. LLM Explainability (DRY - Single Call to generator.py Router)
def get_llm_reasoning(ad_data, audience):
    json_system_prompt = """You are an Executive Chief Marketing Officer (CMO) and Ad Performance Expert.
Analyze the ad metrics below and return a STRICT JSON object with a single key 'reasoning'.
Write 2 clean, high-impact sentences in professional business language (NO technical acronyms like ANN, XGBoost, or embeddings):
1. Explain the commercial driver: Why is this ad expected to succeed or fail based on its psychological hook, audience alignment, emotional tone, and urgency?
2. Give one sharp, strategic recommendation to optimize the copy for maximum ROI."""
    
    user_prompt = f"""
    Ad Copy: "{ad_data['headline']}"
    Target Audience: {audience}
    Word Count: {ad_data.get('word_count', 0)} words
    Emotional Tone: {ad_data.get('sentiment_tag', 'Neutral')} (Score: {ad_data.get('sentiment_score', 0.0)})
    Detected Star/Brand Entities: {ad_data.get('brand_or_celeb_detected', [])} 
    Urgency Score: {ad_data.get('urgency_score', 0.0)}
    --- MODEL BREAKDOWN ---
    FINAL HYBRID PREDICTED CTR: {ad_data.get('predicted_ctr')}%
    """
    
    try:
        raw_json, model_used = call_llm_router(json_system_prompt, user_prompt, max_tokens=220)
        parsed = json.loads(raw_json)
        clean_badge = "Ad-Optima Intelligence Engine"
        return f"[{clean_badge}] {parsed.get('reasoning', 'Strong commercial potential detected.')}"
    except Exception as e:
        return f"Strong semantic hook combined with audience match. (LLM Note: {e})"

# evaluator.py mein Line 135 par evaluate_ads() ko isse replace karein:
def evaluate_ads(ads_list, audience_raw, audience_data=None):
    if audience_data is None:
        audience_data = {
            'age_encoded': '30-34',
            'gender_encoded': 'M',
            'interest': 20,
            'spent': 50.0
        }
    
    evaluated_results = []
    for ad in ads_list:
        # Handle 'BOTH' gender by averaging Male and Female XGBoost predictions
        if audience_data.get('gender_encoded') == 'BOTH':
            aud_m = {**audience_data, 'gender_encoded': 'M'}
            aud_f = {**audience_data, 'gender_encoded': 'F'}
            ctr_m, aud_ctr_m, text_ctr = predict_final_ctr(ad, aud_m)
            ctr_f, aud_ctr_f, _ = predict_final_ctr(ad, aud_f)
            final_ctr = round((ctr_m + ctr_f) / 2.0, 3)
            aud_ctr = round((aud_ctr_m + aud_ctr_f) / 2.0, 3)
        else:
            final_ctr, aud_ctr, text_ctr = predict_final_ctr(ad, audience_data)
            
        ad['predicted_ctr'] = final_ctr
        ad['aud_base_ctr'] = aud_ctr
        ad['text_base_ctr'] = text_ctr
        ad['autopilot_interest_used'] = audience_data['interest']
        evaluated_results.append(ad)
        
    evaluated_results.sort(key=lambda x: x['predicted_ctr'], reverse=True)
    
    # Top 3 ads ko LLM reasoning dena (for rich UI cards)
    for idx, ad_item in enumerate(evaluated_results[:3]):
        ad_item['llm_reasoning'] = get_llm_reasoning(ad_item, audience_raw)
    
    return evaluated_results

if __name__ == "__main__":
    print("\n=======================================================")
    print("AD-OPTIMA: MODULE 3 (THE HYBRID EVALUATOR)")
    print("=======================================================\n")
    
    mode = input("Select Mode [1 for AI Ads from Generator, 2 for Custom Ad]: ")
    aud_input = "Age 30-34, Male, Interest 16, $50 Budget"
    
    if mode == "1":
        prod = input("Enter Product to generate & evaluate: ")
        print("[*] Calling Generator...")
        ai_ads, _ = generate_ad_variations(prod, aud_input, num_ads=3)
        
        print("\n[*] Evaluating Ads...")
        results = evaluate_ads(ai_ads, aud_input)
        
    elif mode == "2":
        custom_text = input("Paste your Custom Ad Text: ")
        print("[*] Extracting Metadata via NLP...")
        ad_metadata = extract_metadata_for_custom_ad(custom_text)
        
        print("\n[*] Evaluating Custom Ad...")
        results = evaluate_ads([ad_metadata], aud_input)
        
    print("\n EVALUATION RESULTS")
    print("-" * 50)
    for i, res in enumerate(results):
        print(f"Rank #{i+1} | CTR: {res['predicted_ctr']}% | Entities: {res.get('brand_or_celeb_detected', [])}")
        print(f"Ad: {res['headline']}")
        if i == 0 and 'llm_reasoning' in res:
            print(f"AI Reason: {res['llm_reasoning']}")
        print("-" * 50)