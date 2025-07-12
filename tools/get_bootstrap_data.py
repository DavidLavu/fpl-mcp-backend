from fastapi import APIRouter
import httpx

router = APIRouter()

@router.get("/tools/get_bootstrap_data")
async def get_bootstrap_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    # Extract mapping dictionaries
    teams = {t["id"]: t["name"] for t in data["teams"]}
    positions = {p["id"]: p["singular_name"] for p in data["element_types"]}
    
    # Filter top 5 players by points_per_game
    top_players = sorted(data["elements"], key=lambda p: float(p["points_per_game"]), reverse=True)[:5]

    result = [
        {
            "name": f"{p['first_name']} {p['second_name']}",
            "team": teams[p["team"]],
            "position": positions[p["element_type"]],
            "points_per_game": float(p["points_per_game"]),
        }
        for p in top_players
    ]

    return {
        "success": True,
        "players": result
    }
