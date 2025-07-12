from fastapi import APIRouter
import httpx
from utils.bootstrap import get_cached_bootstrap

router = APIRouter()

@router.get("/tools/get_player_profile/{eid}")
async def get_player_profile(eid: int):
    # Load bootstrap data
    bootstrap = await get_cached_bootstrap()
    team_map = {team["id"]: team["name"] for team in bootstrap["teams"]}
    player_map = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    player_name = player_map.get(eid, f"ID:{eid}")

    # Fetch player summary (history + fixtures)
    url = f"https://fantasy.premierleague.com/api/element-summary/{eid}/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    # Process match history
    match_history = [
        {
            "round": h["round"],
            "gameweek_name": f"GW{h['round']}",
            "opponent_team": team_map.get(h["opponent_team"], f"ID:{h['opponent_team']}"),
            "was_home": h["was_home"],
            "minutes": h["minutes"],
            "goals_scored": h["goals_scored"],
            "assists": h["assists"],
            "clean_sheets": h["clean_sheets"],
            "goals_conceded": h["goals_conceded"],
            "own_goals": h["own_goals"],
            "penalties_saved": h["penalties_saved"],
            "penalties_missed": h["penalties_missed"],
            "yellow_cards": h["yellow_cards"],
            "red_cards": h["red_cards"],
            "saves": h["saves"],
            "bonus": h["bonus"],
            "bps": h["bps"],
            "total_points": h["total_points"],
            "value": h["value"] / 10,
        }
        for h in data.get("history", [])
    ]

    # Process upcoming fixtures
    upcoming_fixtures = [
        {
            "event": f"GW{f['event']}",
            "opponent_team": team_map.get(f["opponent_team"], f"ID:{f['opponent_team']}"),
            "is_home": f["is_home"],
            "difficulty": f["difficulty"]
        }
        for f in data.get("fixtures", [])
    ]

    return {
        "player_id": eid,
        "player_name": player_name,
        "match_history": match_history,
        "upcoming_fixtures": upcoming_fixtures
    }
