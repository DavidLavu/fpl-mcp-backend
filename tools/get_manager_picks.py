from fastapi import APIRouter, Path, HTTPException
import httpx
from utils.bootstrap import get_cached_bootstrap, build_element_map

router = APIRouter()

@router.get("/tools/get_manager_picks/{tid}/{gw}")
async def get_manager_picks(
    tid: int = Path(..., description="Manager (team) ID"),
    gw: int  = Path(..., description="Gameweek number"),
):
    # 1) Fetch the picks from FPL
    url = f"https://fantasy.premierleague.com/api/entry/{tid}/event/{gw}/picks/"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Team picks not found")
    picks_data = resp.json()

    # 2) Build & cache our element → name map
    element_map = await build_element_map()        # { "players": {id: name}, ... }
    player_map   = element_map["players"]          # { player_id: "Web Name" }

    # 3) Assemble the response
    picks = []
    for p in picks_data.get("picks", []):
        pid = p["element"]
        picks.append({
            "player_id":      pid,
            "player_name":    player_map.get(pid, "Unknown"),
            "multiplier":     p["multiplier"],
            "position":       p["position"],
            "is_captain":     p["is_captain"],
            "is_vice_captain":p["is_vice_captain"],
        })

    return {
        "success": True,
        "team_id":  tid,
        "gameweek": gw,
        "picks":    picks,
    }
