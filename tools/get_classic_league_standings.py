from fastapi import APIRouter, HTTPException, Query
import httpx

router = APIRouter()

FPL_BASE_URL = "https://fantasy.premierleague.com/api"

@router.get("/tools/get_classic_league_standings/{lid}")
async def get_classic_league_standings(lid: int, page: int = Query(1, ge=1)):
    url = f"{FPL_BASE_URL}/leagues-classic/{lid}/standings/?page_standings={page}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch league standings")

    data = resp.json()
    return {
        "league_name": data.get("league", {}).get("name"),
        "league_id": lid,
        "page": page,
        "standings": data.get("standings", {}).get("results", []),
        "has_next": data.get("standings", {}).get("has_next", False),
        "total_entries": data.get("standings", {}).get("entry_count", 0),
    }
