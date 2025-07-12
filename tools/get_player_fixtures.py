from fastapi import APIRouter
import httpx
from utils.bootstrap import get_cached_bootstrap

router = APIRouter()

@router.get("/tools/get_player_fixtures/{eid}")
async def get_player_fixtures(eid: int):
    # Load team and player maps
    bootstrap = await get_cached_bootstrap()
    team_map = {team["id"]: team["name"] for team in bootstrap["teams"]}
    player_map = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    player_name = player_map.get(eid, f"ID:{eid}")

    # Fetch player fixtures from element summary
    url = f"https://fantasy.premierleague.com/api/element-summary/{eid}/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    fixtures = data.get("fixtures", [])
    upcoming = [
        {
            "event": f"GW{f['event']}",
            "opponent_team": team_map.get(f["opponent_team"], f"ID:{f['opponent_team']}"),
            "is_home": f["is_home"],
            "difficulty": f["difficulty"]
        }
        for f in fixtures
    ]

    return {
        "player_id": eid,
        "player_name": player_name,
        "upcoming_fixtures": upcoming,
        "fixture_count": len(upcoming)
    }
