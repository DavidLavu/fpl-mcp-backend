from fastapi import APIRouter, Path
import httpx
from utils.mapping import build_player_map
from utils.bootstrap import get_cached_bootstrap  # ✅ Use cached bootstrap

router = APIRouter()

@router.get("/tools/get_dream_team/{gw}")
async def get_dream_team(gw: int = Path(..., description="Gameweek number")):
    dream_url = f"https://fantasy.premierleague.com/api/dream-team/{gw}/"

    async with httpx.AsyncClient() as client:
        dream_res = await client.get(dream_url)
        dream_data = dream_res.json()

    # ✅ Use cached bootstrap
    bootstrap_data = await get_cached_bootstrap()
    player_map = build_player_map(bootstrap_data)

    top_player_id = dream_data.get("top_element")
    top_player_points = dream_data.get("top_element_points")

    top_player = {
        "player_id": top_player_id if top_player_id else None,
        "player_name": player_map.get(top_player_id, "Unavailable") if top_player_id else "Unavailable",
        "points": top_player_points if top_player_points is not None else 0
    }

    team = dream_data.get("team", [])

    top_11 = [
        {
            "player_id": p["element"],
            "player_name": player_map.get(p["element"], "Unknown"),
            "position": p["position"]
        }
        for p in team
    ]

    return {
        "success": True,
        "gw": gw,
        "top_player": top_player,
        "dream_team": top_11
    }
