"""
CineChat — AI & ML Modulu
Data Science Final Layihəsi

Bu modulda:
1. LLM əsaslı chat (OpenRouter API)
2. TF-IDF əsaslı film recommender (scikit-learn)
3. Sentiment analizi (VADER)
4. Əsas məlumat işləmə (pandas, numpy)
"""

import httpx
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Konfiqurasiya ──
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_URL        = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "openrouter/free"

# ── Film verilənlər bazası (TF-IDF recommender üçün) ──
MOVIES_DB = pd.DataFrame([
    {"title": "The Shawshank Redemption", "tags": "drama hope prison friendship redemption 1994"},
    {"title": "The Godfather",            "tags": "crime family mafia power drama 1972"},
    {"title": "The Dark Knight",          "tags": "superhero batman action crime joker 2008"},
    {"title": "Schindler's List",         "tags": "war history holocaust drama 1993"},
    {"title": "Forrest Gump",             "tags": "drama comedy history love life 1994"},
    {"title": "Inception",                "tags": "scifi thriller dream action mind 2010"},
    {"title": "Pulp Fiction",             "tags": "crime drama thriller nonlinear 1994"},
    {"title": "Fight Club",               "tags": "drama thriller psychology identity 1999"},
    {"title": "Goodfellas",               "tags": "crime biography mafia drama 1990"},
    {"title": "The Matrix",               "tags": "scifi action philosophy reality simulation 1999"},
    {"title": "Interstellar",             "tags": "scifi space drama time love 2014"},
    {"title": "The Silence of the Lambs", "tags": "thriller crime psychology horror 1991"},
    {"title": "Parasite",                 "tags": "drama thriller class society korean 2019"},
    {"title": "Spirited Away",            "tags": "animation fantasy adventure japanese 2001"},
    {"title": "The Lion King",            "tags": "animation drama family adventure 1994"},
    {"title": "Titanic",                  "tags": "romance drama history disaster 1997"},
    {"title": "Avengers Endgame",         "tags": "superhero action adventure marvel 2019"},
    {"title": "Joker",                    "tags": "drama crime psychology comics 2019"},
    {"title": "1917",                     "tags": "war drama history action 2019"},
    {"title": "La La Land",               "tags": "romance musical drama dreams 2016"},
    {"title": "Whiplash",                 "tags": "drama music ambition perfection 2014"},
    {"title": "Get Out",                  "tags": "horror thriller race psychology 2017"},
    {"title": "Mad Max Fury Road",        "tags": "action scifi post-apocalyptic 2015"},
    {"title": "Her",                      "tags": "scifi romance drama technology 2013"},
    {"title": "The Grand Budapest Hotel", "tags": "comedy drama adventure quirky 2014"},
    {"title": "Hereditary",               "tags": "horror drama family grief 2018"},
    {"title": "Moonlight",                "tags": "drama identity race sexuality 2016"},
    {"title": "A Beautiful Mind",         "tags": "biography drama mathematics genius 2001"},
    {"title": "Good Will Hunting",        "tags": "drama mathematics friendship therapy 1997"},
    {"title": "The Truman Show",          "tags": "drama comedy scifi reality media 1998"},
    {"title": "Eternal Sunshine",         "tags": "romance scifi drama memory 2004"},
    {"title": "No Country for Old Men",   "tags": "thriller crime drama violence 2007"},
    {"title": "There Will Be Blood",      "tags": "drama history oil ambition 2007"},
    {"title": "12 Angry Men",             "tags": "drama legal justice jury 1957"},
    {"title": "Blade Runner 2049",        "tags": "scifi drama dystopia 2017"},
    {"title": "Dune",                     "tags": "scifi adventure epic desert 2021"},
    {"title": "The Notebook",             "tags": "romance drama love memory 2004"},
    {"title": "Oppenheimer",              "tags": "biography drama history science war 2023"},
    {"title": "Barbie",                   "tags": "comedy fantasy adventure identity 2023"},
    {"title": "Poor Things",              "tags": "drama fantasy scifi feminism 2023"},
])

# ── TF-IDF modeli ──
_tfidf  = TfidfVectorizer(stop_words="english")
_matrix = _tfidf.fit_transform(MOVIES_DB["tags"])

# ── Sentiment analizi ──
_sentiment = SentimentIntensityAnalyzer()


# ═══════════════════════════════════════
#  TF-IDF ƏSASLI RECOMMENDER
# ═══════════════════════════════════════
def recommend_by_tfidf(movie_title: str, n: int = 5) -> list:
    """
    TF-IDF + Cosine Similarity əsaslı film recommender.
    Data Science kursunun əsas texnikası.
    """
    title_lower = movie_title.lower()
    idx = None

    # Filmi bazada axtar
    for i, row in MOVIES_DB.iterrows():
        if row["title"].lower() == title_lower:
            idx = i
            break

    if idx is not None:
        # Cosine similarity hesabla
        sim_scores = cosine_similarity(_matrix[idx], _matrix).flatten()
        sim_scores[idx] = 0  # özünü çıxart
        top_indices = np.argsort(sim_scores)[::-1][:n]
        return MOVIES_DB.iloc[top_indices]["title"].tolist()

    # Film bazada yoxdursa — LLM ilə tövsiyyə edəcəyik
    return []


# ═══════════════════════════════════════
#  SENTIMENT ANALİZİ
# ═══════════════════════════════════════
def analyze_sentiment(text: str) -> dict:
    """
    İstifadəçi mesajının sentiment analizini aparır.
    VADER (Valence Aware Dictionary and sEntiment Reasoner) istifadə edir.
    """
    scores = _sentiment.polarity_scores(text)
    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "scores": scores}


# ═══════════════════════════════════════
#  LLM ƏSASLI TÖVSIYYƏ (fallback)
# ═══════════════════════════════════════
async def recommend_by_llm(movie: str, lang: str) -> list:
    """
    TF-IDF-də film tapılmadıqda LLM-dən tövsiyyə alır.
    """
    if lang == "az":
        prompt = f'"{movie}" filminə oxşar 5 film adı ver. Yalnız JSON array: ["Film1","Film2","Film3","Film4","Film5"]'
    else:
        prompt = f'Give 5 movies similar to "{movie}". Return ONLY a JSON array: ["Film1","Film2","Film3","Film4","Film5"]'

    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "max_tokens": 120,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        data = res.json()
        text = data["choices"][0]["message"]["content"]
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        # Vergüllə ayrılmış siyahı kimi parse et
        return [s.strip().strip('"') for s in text.split(",") if s.strip()][:5]


# ═══════════════════════════════════════
#  ANA RECOMMENDER
# ═══════════════════════════════════════
async def get_recommendations(movie: str, lang: str) -> list:
    """
    1. TF-IDF ilə cəhd et
    2. Tapılmadıqda LLM-ə düş
    """
    films = recommend_by_tfidf(movie, n=5)
    if not films:
        films = await recommend_by_llm(movie, lang)
    return films


# ═══════════════════════════════════════
#  LLM CHAT
# ═══════════════════════════════════════
SYSTEM_PROMPTS = {
    "az_before": """Sən CineChat adlı film müzakirə köməkçisisən.
Qaydalar:
- Spoiler vermə — filmin sonunu, twist-ləri, ölümləri açıqlama
- Yalnız janr, aktyorlar, ümumi ab-hava barədə danış
- Azərbaycan dilində düzgün, səlis yazırsan
- Qısa və faydalı cavablar ver
- Cavabda heç vaxt ** işarəsi işlətmə""",

    "az_after": """Sən CineChat adlı film müzakirə köməkçisisən.
Qaydalar:
- Spoilerləri sərbəst işlədə bilərsən
- Personaj psixologiyası, simvolizm, rejissorun mesajı barədə dərin analiz et
- Azərbaycan dilində düzgün, səlis yazırsan
- Cavabda heç vaxt ** işarəsi işlətmə""",

    "en_before": """You are CineChat, a friendly movie assistant.
Rules:
- No spoilers at all — do not reveal endings, twists, deaths
- Only talk about genre, cast, general mood
- Use simple English: short sentences, common words (B1 level)
- Never use ** in your response""",

    "en_after": """You are CineChat, a friendly movie assistant.
Rules:
- Spoilers are fine — the user watched the film
- Give deep analysis: themes, characters, symbols, director choices
- Use simple English: short sentences, common words (B1 level)
- Never use ** in your response""",
}

async def get_chat_response(message: str, history: list,
                             mode: str, lang: str, movie: str) -> str:
    key    = f"{lang}_{mode}"
    system = SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["en_before"])

    # Sentiment analizi aparırıq (Data Science elementi)
    sentiment = analyze_sentiment(message)
    if sentiment["label"] == "negative" and lang == "az":
        system += "\nİstifadəçi mənfi əhval-ruhiyyədə görünür, daha həssas cavab ver."
    elif sentiment["label"] == "negative":
        system += "\nThe user seems upset. Be extra supportive and kind."

    messages = [{"role": "system", "content": system}]
    messages += history[-12:]  # Son 12 mesaj (context window idarəsi)
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "max_tokens": 1000, "messages": messages},
        )
        data = res.json()
        return data["choices"][0]["message"]["content"]
