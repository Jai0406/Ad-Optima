import urllib.parse
import feedparser
import os
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq
from tavily import TavilyClient
from fastembed import TextEmbedding

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

analyzer = SentimentIntensityAnalyzer()

print("[*] Loading Semantic Filter Model (FastEmbed)...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

DB_NAME = "trends_cache.db"
CACHE_TTL_HOURS = 1

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trends 
                 (keyword TEXT PRIMARY KEY, data TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def get_cached_trends(keyword):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT data, timestamp FROM trends WHERE keyword=?", (keyword.strip().lower(),))
        row = c.fetchone()
        conn.close()
        
        if row and row[0]:
            cached_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                cached_list = json.loads(row[0])
                if len(cached_list) > 0:
                    print(f"[*] Cache Hit! Loading data from SQLite for '{keyword}'...")
                    return cached_list
    except Exception as e:
        print(f"  [!] Cache read error (fetching fresh): {e}")
    return None

def save_to_cache(keyword, data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO trends (keyword, data, timestamp) 
                 VALUES (?, ?, ?) 
                 ON CONFLICT(keyword) DO UPDATE SET data=excluded.data, timestamp=excluded.timestamp''', 
              (keyword.lower(), json.dumps(data), timestamp))
    conn.commit()
    conn.close()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        tag = "Positive"
    elif score <= -0.05:
        tag = "Negative"
    else:
        tag = "Neutral"
    return score, tag

def fetch_google_news(product_keyword, days=15):
    valid_news = []
    # Generic Commercial Queries: Works for any domain (Tech, Finance, FMCG, Auto)
    search_queries = [
        f'"{product_keyword}" India when:{days}d',
        f'{product_keyword} (market OR consumers OR demand OR launch OR price) India when:{days}d'
    ]
    
    for query in search_queries:
        safe_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Basic cleaning: remove source name at the end (e.g., " - The Times of India")
                clean_title = entry.title.rsplit(" - ", 1)[0]
                score, tag = analyze_sentiment(clean_title)
                
                # Brand Safety: Drop highly negative/tragic news, aur agar "dies/dead/accident" word hai toh seedha block
                danger_words = ["dies", "dead", "killed", "accident", "crash", "murder", "suicide", "seizure"]
                if score > -0.45 and not any(w in clean_title.lower() for w in danger_words):
                    valid_news.append({
                        "source": "Google News",
                        "text": entry.title,
                        "clean_text": clean_title,
                        "sentiment_score": round(score, 2),
                        "sentiment_tag": tag
                    })
        except Exception as e:
            print(f"  [!] Google News failed for query '{query}': {e}")
            
    return valid_news

def fetch_pytrends_queries(product_keyword):
    valid_trends = []
    try:
        pytrend = TrendReq(hl='en-IN', tz=330, timeout=(5, 10))
        pytrend.build_payload(kw_list=[product_keyword], geo='IN', timeframe='today 1-m')
        
        related_queries = pytrend.related_queries()
        if related_queries and product_keyword in related_queries:
            top_queries = related_queries[product_keyword].get('top')
            if top_queries is not None:
                for index, row in top_queries.iterrows():
                    valid_trends.append({
                        "source": "Google Trends",
                        "text": row['query'],
                        "clean_text": row['query'],
                        "sentiment_score": 0.0,
                        "sentiment_tag": "Neutral"
                    })
        return valid_trends
    except Exception as e:
        print(f"  [!] PyTrends skipped/rate-limited: {e}")
        return []

def fetch_tavily_fallback(product_keyword):
    if not TAVILY_API_KEY:
        return []
        
    valid_trends = []
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=f"latest consumer news market trends {product_keyword} India",
            search_depth="basic",
            max_results=5
        )
        for res in response.get('results', []):
            score, tag = analyze_sentiment(res['title'])
            if score > -0.45:
                valid_trends.append({
                    "source": "Tavily (Fallback)",
                    "text": res['title'],
                    "clean_text": res['title'],
                    "sentiment_score": round(score, 2),
                    "sentiment_tag": tag
                })
        return valid_trends
    except Exception as e:
        print(f"  [!] Tavily failed: {e}")
        return []

def master_context_engine(product_keyword, force_refresh=True):
    if not force_refresh:
        cached_data = get_cached_trends(product_keyword)
        if cached_data:
            return cached_data
        
    print(f"[*] Fetching fresh data from APIs (Bypassing cache)...")
    master_context = []
    
    news = fetch_google_news(product_keyword, days=15)
    master_context.extend(news)
    
    trends = fetch_pytrends_queries(product_keyword)
    master_context.extend(trends)
    
    if len(news) < 5:
        tavily_data = fetch_tavily_fallback(product_keyword)
        master_context.extend(tavily_data)
        
    if master_context:
        # Deduplicate based on clean_text
        unique_context = {item['clean_text']: item for item in master_context}.values()
        master_context = list(unique_context)
        save_to_cache(product_keyword, master_context)
        
    return master_context

def get_top_marketing_trends(fetched_trends, product_keyword, top_n=15):
    # Separate News from PyTrends (Taaki PyTrends news ki seat na khayein)
    news_items = [item for item in fetched_trends if item['source'] != "Google Trends"]
    pytrends_items = [item for item in fetched_trends if item['source'] == "Google Trends"]
    
    if not news_items:
        return [], pytrends_items[:5]

    print(f"[*] Semantically filtering {len(news_items)} news articles using Dynamic Anchors for '{product_keyword}'...")
    
    # UNIVERSAL DYNAMIC ANCHORS (Domain-Agnostic, Product-Injected)
    dynamic_anchors = [
        f"Consumers buying, upgrading, or showing high demand for {product_keyword} in India",
        f"New product launch, latest features, or technology innovation in {product_keyword}",
        f"Price drop, festive discount, special offer, or affordability of {product_keyword}",
        f"Market growth, record sales, or industry trends for {product_keyword}",
        f"Important consumer alert, policy change, or financial benefit related to {product_keyword}"
    ]
    
    # Embed anchors (Shape: 5 x 384) and news (Shape: N x 384)
    anchor_vectors = np.array(list(embedding_model.embed(dynamic_anchors)))
    news_texts = [item['clean_text'] for item in news_items]
    news_vectors = np.array(list(embedding_model.embed(news_texts)))
    
    scored_news = []
    for i, item in enumerate(news_items):
        similarities = np.dot(anchor_vectors, news_vectors[i])
        max_sim_score = float(np.max(similarities))
        
        scored_news.append((max_sim_score, item))
    
    scored_news.sort(key=lambda x: x[0], reverse=True)
    top_news = [{**item, "relevance_score": round(score, 3)} for score, item in scored_news[:top_n]]
    
    return top_news, pytrends_items[:5]

def display_segregated_output(trends_data):
    pos_items = [item for item in trends_data if item['sentiment_tag'] == 'Positive']
    neg_items = [item for item in trends_data if item['sentiment_tag'] == 'Negative']
    neu_items = [item for item in trends_data if item['sentiment_tag'] == 'Neutral']
    
    print("\n=======================================================")
    print(f"TRENDS SUMMARY (TOTAL: {len(trends_data)} UNIQUE ITEMS)")
    print(f"Positive: {len(pos_items)}  |  Negative: {len(neg_items)}  |  Neutral: {len(neu_items)}")
    print("=======================================================\n")
    
    sections = [
        (" POSITIVE TRENDS", pos_items),
        (" NEGATIVE TRENDS", neg_items),
        (" NEUTRAL TRENDS", neu_items)
    ]
    
    for title, items in sections:
        if items:
            print(f"{title}")
            print("-" * 40)
            for idx, item in enumerate(items, 1):
                print(f"{idx:02d}. [{item['source']}] (VADER: {item['sentiment_score']})")
                print(f"    {item['text']}\n")

def display_context_digest(top_news, top_keywords):
    print("\n=======================================================")
    print(f" TOP {len(top_news)} COMMERCIAL NEWS HOOKS (FOR LLM)")
    print("=======================================================\n")
    for idx, item in enumerate(top_news, 1):
         print(f"{idx:02d}. [Match: {item['relevance_score']}] {item['text']}")
         
    if top_keywords:
        print("\n=======================================================")
        print(f" TOP SEARCH KEYWORDS (PYTRENDS)")
        print("=======================================================\n")
        keywords_str = ", ".join([k['text'] for k in top_keywords])
        print(f"   {keywords_str}")
        
    print("\n[Note: LLM will be instructed to pick the best 3-4 hooks from above and ignore any noise]")

if __name__ == "__main__":
    init_db()
    
    test_product = input("\nEnter the product or keyword (e.g., Smart TV, Health Insurance, SUV): ")
    print(f"\n INITIALIZING CONTEXT ENGINE FOR: '{test_product.upper()}'")
    
    final_trends = master_context_engine(test_product, force_refresh=True)
    display_segregated_output(final_trends)
    
    if final_trends:
        top_news_hooks, top_search_kw = get_top_marketing_trends(final_trends, test_product, top_n=15)
        display_context_digest(top_news_hooks, top_search_kw)