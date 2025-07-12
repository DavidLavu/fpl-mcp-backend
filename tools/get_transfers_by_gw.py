from fastapi import APIRouter
import httpx
from utils.bootstrap import get_cached_bootstrap
from typing import List

router = APIRouter()

@router.get("/tools/get_transfers_by_gw/{tid}/{gw}")
async def get_transfers_by_gw(tid: int, gw: int):
    # Fetch bootstrap mappings
    bootstrap = await get_cached_bootstrap()
    player_map = {el["id"]: el["web_name"] for el in bootstrap["elements"]}

    # Fetch all transfers
    url = f"https://fantasy.premierleague.com/api/entry/{tid}/transfers/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        transfers = response.json()

    # Filter and enrich
    gw_transfers = [
        {
            "element_in": player_map.get(t["element_in"], f"ID:{t['element_in']}"),
            "element_out": player_map.get(t["element_out"], f"ID:{t['element_out']}"),
            "cost": t.get("cost", 0),  # ✅ Safe fallback
        }
        for t in transfers
        if t["event"] == gw
    ]

    return {
        "team_id": tid,
        "gameweek": gw,
        "transfers": gw_transfers,
        "transfer_count": len(gw_transfers),
    }
