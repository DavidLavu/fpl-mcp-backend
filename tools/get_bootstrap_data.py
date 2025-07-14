from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import httpx

router = APIRouter()

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

@router.get("/tools/get_bootstrap_data")
async def get_bootstrap_data() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(BOOTSTRAP_URL)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch bootstrap data: {e}")

    teams = {team["id"]: team["name"] for team in data.get("teams", [])}
    positions = {pos["id"]: pos["singular_name"] for pos in data.get("element_types", [])}
    events = data.get("events", [])
    elements = data.get("elements", [])
    element_stats = data.get("element_stats", [])
    game_settings = data.get("game_settings", {})
    phases = data.get("phases", [])
    total_players = data.get("total_players", 0)

    top_players = sorted(
        elements,
        key=lambda p: float(p.get("points_per_game", 0.0)),
        reverse=True
    )[:5]

    top_players_summary = [
        {
            "name": f"{p['first_name']} {p['second_name']}",
            "team": teams.get(p["team"], "Unknown"),
            "position": positions.get(p["element_type"], "Unknown"),
            "points_per_game": round(float(p.get("points_per_game", 0.0)), 2),
        }
        for p in top_players
    ]

    return {
        "success": True,
        "summary": {
            "top_players_by_ppg": top_players_summary,
            "total_players": total_players,
            "game_settings": {
                "currency_multiplier": game_settings.get("currency_multiplier"),
                "transfers_limit": game_settings.get("transfers_limit"),
                "ui_selection_limit": game_settings.get("ui_selection_limit"),
                "chips": game_settings.get("chips", [])
            },
            "current_event": next((e for e in events if e.get("is_current")), None),
            "next_event": next((e for e in events if e.get("is_next")), None),
        },
        "raw": {
            "teams": data.get("teams", []),
            "elements": elements,
            "element_types": data.get("element_types", []),
            "events": events,
            "element_stats": element_stats,
            "game_settings": game_settings,
            "phases": phases
        }
    }
