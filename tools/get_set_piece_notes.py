from fastapi import APIRouter
import httpx
from utils.bootstrap import get_cached_bootstrap

router = APIRouter()

@router.get("/tools/get_set_piece_notes")
async def get_set_piece_notes():
    # Get team name map
    bootstrap = await get_cached_bootstrap()
    team_map = {team["id"]: team["name"] for team in bootstrap["teams"]}

    # Fetch set-piece notes
    url = "https://fantasy.premierleague.com/api/team/set-piece-notes/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    # Process each team's notes
    enriched_notes = []
    for team in data.get("teams", []):
        team_id = team.get("id")
        notes_list = team.get("notes", [])

        if not notes_list:
            continue  # skip teams with no notes

        enriched_notes.append({
            "team": team_map.get(team_id, f"ID:{team_id}"),
            "notes": [n["info_message"] for n in notes_list],
            "sources": [n["source_link"] for n in notes_list if n.get("source_link")]
        })

    return {
        "last_updated": data.get("last_updated"),
        "set_piece_notes": enriched_notes
    }
