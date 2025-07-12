from fastapi import APIRouter, Path
import httpx
from utils.mapping import build_player_map
from utils.bootstrap import get_cached_bootstrap  # ✅ Use cache

router = APIRouter()

@router.get("/tools/get_event_live/{gw}")
async def get_event_live(gw: int = Path(..., description="Gameweek number")):
    live_url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"

    # ✅ Use cached bootstrap data
    bootstrap_data = await get_cached_bootstrap()

    async with httpx.AsyncClient() as client:
        live_res = await client.get(live_url)
        live_data = live_res.json()

    player_map = build_player_map(bootstrap_data)
    elements = live_data.get("elements", [])

    results = [
        {
            "player_id": e["id"],
            "player_name": player_map.get(e["id"], "Unknown"),
            "points": e["stats"]["total_points"],
            "minutes": e["stats"]["minutes"],
            "goals": e["stats"]["goals_scored"],
            "assists": e["stats"]["assists"],
            "bps": e["stats"]["bps"]
        }
        for e in elements if e["stats"]["minutes"] > 0
    ]

    return {"success": True, "gw": gw, "players": results}
