"""
CineChat — OMDB Film Axtarış Modulu
"""

import httpx
import os

OMDB_KEY = os.getenv("OMDB_API_KEY", "")

async def search_movie(title: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        res  = await client.get(
            "https://www.omdbapi.com/",
            params={"t": title, "apikey": OMDB_KEY}
        )
        data = res.json()

    if data.get("Response") == "True":
        return {
            "found":    True,
            "title":    data.get("Title", ""),
            "year":     data.get("Year", ""),
            "poster":   data.get("Poster", ""),
            "rating":   data.get("imdbRating", "N/A"),
            "genre":    data.get("Genre", "").split(",")[0].strip(),
            "runtime":  data.get("Runtime", ""),
            "director": data.get("Director", ""),
            "plot":     data.get("Plot", ""),
        }
    return {"found": False}
