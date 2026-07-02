"""
CineChat — Python Backend (FastAPI)
Data Science Final Layihəsi
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import re
import logging

from ai import get_chat_response, get_recommendations
from movies import search_movie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cinechat")

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
    try:
        reply = await get_chat_response(
            message=req.message,
            history=req.history,
            mode=req.mode,
            lang=req.lang,
            movie=req.movie,
        )
    except Exception as e:
        logger.exception("chat endpoint failed")
        return JSONResponse(status_code=500, content={"detail": f"Chat error: {str(e)}"})
    # ** bold ** → <b>bold</b>  (frontend-də render olunur)
    reply = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', reply)
    return {"reply": reply}


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        films = await get_recommendations(req.movie, req.lang)
    except Exception as e:
        logger.exception("recommend endpoint failed")
        return JSONResponse(status_code=500, content={"detail": f"Recommend error: {str(e)}"})
    return {"films": films}


@app.post("/search")
async def search(req: SearchRequest):
    try:
        result = await search_movie(req.title)
    except Exception as e:
        logger.exception("search endpoint failed")
        return JSONResponse(status_code=500, content={"detail": f"Search error: {str(e)}"})
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
