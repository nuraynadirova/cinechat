"""
CineChat — Python Backend (FastAPI)
Data Science Final Layihəsi
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

from ai import get_chat_response, get_recommendations
from movies import search_movie

app = FastAPI(title="CineChat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request modelləri ──
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    mode: str        # "before" | "after"
    lang: str        # "az" | "en"
    movie: str

class RecommendRequest(BaseModel):
    movie: str
    lang: str

class SearchRequest(BaseModel):
    title: str


# ── Endpointlər ──

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = await get_chat_response(
        message=req.message,
        history=req.history,
        mode=req.mode,
        lang=req.lang,
        movie=req.movie,
    )
    # ** bold ** → <b>bold</b>  (frontend-də render olunur)
    reply = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', reply)
    return {"reply": reply}


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    films = await get_recommendations(req.movie, req.lang)
    return {"films": films}


@app.post("/search")
async def search(req: SearchRequest):
    result = await search_movie(req.title)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
