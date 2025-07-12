from fastapi import APIRouter
from utils.bootstrap import get_cached_bootstrap

router = APIRouter()

@router.get("/tools/get_crowd_trends_by_gw/{gw}")
async def get_crowd_trends_by_gw(gw: int):
    bootstrap = await get_cached_bootstrap()
    players = bootstrap["elements"]

    # Map element_type to position names
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Group players by position
    players_by_position = {pos: [] for pos in position_map.values()}
    for p in players:
        pos = position_map.get(p["element_type"])
        if pos:
            players_by_position[pos].append(p)

    # For each position, get top 5 selected
    top_selected_by_position = {}
    for pos, plist in players_by_position.items():
        top5 = sorted(plist, key=lambda x: float(x.get("selected_by_percent", 0)), reverse=True)[:5]
        top_selected_by_position[pos] = [
            {"player": p["web_name"], "selected_by_percent": p["selected_by_percent"]}
            for p in top5
        ]

    return {
        "gameweek": gw,
        "most_selected_by_position": top_selected_by_position
    }
