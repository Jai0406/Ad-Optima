import os
import json
import requests
import spacy
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from news import init_db, master_context_engine, get_top_marketing_trends
from openai import OpenAI

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")
BASETEN_ENDPOINT_URL = os.getenv("BASETEN_ENDPOINT_URL", "")

# --- 1. Load spaCy for Brand Trust & Star Power NER ---
print("[*] Loading spaCy NER Model (en_core_web_sm)...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- 2. Strict Pydantic Schema ---
class LLMGeneratedAd(BaseModel):
    headline: str = Field(description="Punchy, conversion-focused ad copy with a Hook, Benefit, and Call-To-Action (CTA)")
    urgency_score: float = Field(ge=0.0, le=1.0, description="Urgency/FOMO level from 0.0 to 1.0")
    humor_level: float = Field(ge=0.0, le=1.0, description="Humor/Wit level from 0.0 to 1.0")
    uses_live_trend: bool = Field(description="True if it uses a market trend/fact from the context")
    trend_angle_used: str = Field(description="Which market fact/trend was used to create the hook")

class AdBatchSchema(BaseModel):
    ads: list[LLMGeneratedAd]

# --- 3. Smart NER: Brand Trust & Celebrity Power Calculator ---
def enrich_ads_locally(raw_ads_list, raw_trends_data, product_keyword):
    """
    Calculates word_count and checks for Brand Trust (e.g., VW, Tata) or 
    Celebrity Power (e.g., SRK, Virat Kohli) without counting generic words like 'India' or 'EV'.
    """
    all_trends_text = " ".join([item['clean_text'].lower() for item in raw_trends_data])
    
    # Generic words jinhe Brand/Celebrity nahi maanna hai
    ignore_terms = {product_keyword.lower(), "india", "indian", "ev", "evs", "tv", "tvs", "suv", "suvs", "rs", "inr"}
    
    enriched_ads = []
    for idx, ad in enumerate(raw_ads_list, 1):
        ad_text = ad.headline.strip()
        word_count = len(ad_text.split())
        
        doc = nlp(ad_text)
        # Sirf PERSON (Celebs like SRK/Kohli) aur ORG/PRODUCT (Brands like VW/Apple) nikalo
        valid_entities = []
        for ent in doc.ents:
            clean_ent = ent.text.strip()
            if ent.label_ in ["PERSON", "ORG", "PRODUCT"] and clean_ent.lower() not in ignore_terms:
                valid_entities.append(clean_ent)
        
        # Remove duplicates
        valid_entities = list(set(valid_entities))
        
        # Calculate Entity Power Score (Star Power / Brand Trust)
        entity_power_score = 0.0
        if valid_entities:
            # Agar wo Celeb/Brand aaj ke live trends mein bhi viral hai -> Max Boost (1.0)
            is_trending_entity = any(ent.lower() in all_trends_text for ent in valid_entities)
            if is_trending_entity:
                entity_power_score = 1.0  # Viral Star / Trending Brand
            else:
                entity_power_score = 0.5  # Established Brand Trust / Celeb Mention (Not in today's news, but adds trust)
                
        enriched_ads.append({
            "ad_id": idx,
            "headline": ad_text,
            "word_count": word_count,
            "urgency_score": round(ad.urgency_score, 2),
            "humor_level": round(ad.humor_level, 2),
            "uses_live_trend": ad.uses_live_trend,
            "trend_angle_used": ad.trend_angle_used,
            "brand_or_celeb_detected": valid_entities,
            "entity_power_score": entity_power_score
        })
        
    return enriched_ads

# --- 4. Dual-Cloud LLM Router ---
def call_llm_router(system_prompt, user_prompt, max_tokens=1200):
    if BASETEN_API_KEY:
        print("[*] Routing to Primary LLM: Meta Muse Spark (via Baseten)...")
        try:
            client = OpenAI(
                base_url="https://inference.baseten.co/v1",
                api_key=BASETEN_API_KEY,
            )
            response = client.chat.completions.create(
                model="meta-models/muse-spark-open-1026",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.75,
                response_format={"type": "json_object"}
            )
            
            msg_obj = response.choices[0].message
            content = getattr(msg_obj, "content", None)
            
            if not content and hasattr(msg_obj, "reasoning_content"):
                content = msg_obj.reasoning_content
                
            if content:
                return content, "Meta-Muse-Spark-1026 (Baseten)"
            else:
                print(f"  [!] Baseten returned empty content object. Full response: {response}")
                raise ValueError("Empty content from Baseten")
                
        except Exception as e:
            print(f"  [!] Baseten Primary Route failed ({e}). Switching to Groq Fallback...")

    print("[*] Routing to Secondary LLM: qwen/qwen3.8-27b (via Groq)...")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env!")
        
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.75,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content, "qwen/qwen3.8-27b (Groq)"

# --- 5. Main Generation Pipeline ---
def generate_ad_variations(product_keyword, target_audience, num_ads=5, force_refresh=False):
    init_db()
    raw_trends = master_context_engine(product_keyword, force_refresh=force_refresh)
    
    if raw_trends:
        top_news, top_keywords = get_top_marketing_trends(raw_trends, product_keyword, top_n=10)
        news_lines = "\n".join([f"- {item['clean_text']}" for item in top_news])
    else:
        news_lines = "High consumer demand and festive season offers."

    # TRUE AD COPYWRITER SYSTEM PROMPT (No News Reporter Tone!)
    system_prompt = f"""You are a Performance Marketing Copywriter for '{product_keyword}'.
Target Audience: {target_audience}

MARKET CONTEXT (Recent News):
{news_lines}

CRITICAL COPYWRITING RULES:
1. DO NOT write news headlines or articles! Write persuasive, high-converting Meta/Instagram AD COPY that sells '{product_keyword}' directly to the buyer (use "You", "Your").
2. DO NOT mention competitor brand names from the news (like Tata, MG, LG, Samsung) unless '{product_keyword}' specifically includes that brand name! Instead, extract the *underlying market trend* (e.g., "EV demand is 3x higher than supply", "75% surge in buyers", "Festive discounts") and use it to create FOMO for '{product_keyword}'.
3. If the user's input includes a specific Brand (e.g., "VW SUV") or Celebrity endorsement, highlight that brand trust/star power.
4. Every ad MUST have: [A Catchy Hook/Question] + [Value/Benefit] + [Strong Call-To-Action like 'Book Now', 'Claim Offer', 'Switch Today']. Length: 12 to 22 words.
5. STRICTLY FORBIDDEN: Do NOT mention or include the numeric advertising budget or dollar values (like "$50 budget") inside the ad headline or copy under any circumstances!
6. Output STRICT JSON matching this structure:
{{
  "ads": [
    {{
      "headline": "Still burning cash on petrol? Join the 75% surge in smart drivers switching to electric. Book your free test ride today!",
      "urgency_score": 0.85,
      "humor_level": 0.1,
      "uses_live_trend": true,
      "trend_angle_used": "75% surge in EV adoption"
    }}
  ]
}}"""

    user_prompt = f"Write {num_ads} killer conversion ads for '{product_keyword}' in JSON format."

    try:
        raw_json_str, model_used = call_llm_router(system_prompt, user_prompt, max_tokens=1200)
        # Clean markdown backticks if LLM adds them
        clean_json_str = raw_json_str.strip()
        if clean_json_str.startswith("```"):
            clean_json_str = clean_json_str.split("```")[1]
            if clean_json_str.startswith("json"):
                clean_json_str = clean_json_str[4:]
        clean_json_str = clean_json_str.strip()
        
        parsed_data = json.loads(clean_json_str)
        validated_batch = AdBatchSchema(**parsed_data)
        
        final_ads = enrich_ads_locally(validated_batch.ads, raw_trends, product_keyword)
        return final_ads, model_used
        
    except ValidationError as ve:
        print(f"[-] Pydantic JSON Validation Error: {ve}")
        return [], "Validation Failed"
    except Exception as e:
        print(f"[-] LLM Generation Error: {e}")
        return [], "Error"

if __name__ == "__main__":
    print("\n=======================================================")
    print(" AD-OPTIMA: MODULE 2 (TRUE AD COPYWRITER + BRAND NER)")
    print("=======================================================")
    
    prod = input("\nEnter Product/Brand (e.g., EV, Volkswagen SUV, SRK Denver Deo): ")
    aud = input("Enter Target Audience (e.g., Age 30-34, Male, Auto Buyer): ")
    
    ads_output, active_model = generate_ad_variations(prod, aud, num_ads=5, force_refresh=False)
    
    if ads_output:
        print("\n=======================================================")
        print(f" GENERATED {len(ads_output)} COMMERCIAL ADS VIA [{active_model}]")
        print("=======================================================\n")
        print(json.dumps(ads_output, indent=2))