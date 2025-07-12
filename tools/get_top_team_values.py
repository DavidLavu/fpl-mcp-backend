from fastapi import APIRouter
import httpx

router = APIRouter()

@router.get("/tools/get_top_team_values")
async def get_top_team_values():
    url = "https://fantasy.premierleague.com/api/stats/most-valuable-teams/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    teams = [
        {
            "manager_id": team["entry"],
            "player_name": team["player_name"],
            "team_name": team["name"],
            "value_with_bank": team["value_with_bank"] / 10,  # convert to £M
            "total_transfers": team["total_transfers"],
        }
        for team in data
    ]

    return {
        "top_valuable_teams": teams,
        "count": len(teams)
    }
