from fastapi import APIRouter, Query
import httpx
from utils.mapping import build_team_map, map_team
from utils.bootstrap import get_cached_bootstrap  # ✅ cache

router = APIRouter()

@router.get("/tools/get_fixtures_by_gw")
async def get_fixtures_by_gw(gw: int = Query(..., description="Gameweek number")):
    fixtures_url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"

    async with httpx.AsyncClient() as client:
        fixtures_res = await client.get(fixtures_url)
        fixtures = fixtures_res.json()

    bootstrap = await get_cached_bootstrap()
    team_map = build_team_map(bootstrap)

    results = sorted([
        {
            "event": gw,
            "kickoff_time": f.get("kickoff_time"),
            "team_h": map_team(f.get("team_h"), team_map),
            "team_a": map_team(f.get("team_a"), team_map),
            "team_h_difficulty": f.get("team_h_difficulty"),
            "team_a_difficulty": f.get("team_a_difficulty"),
            "finished": f.get("finished")
        }
        for f in fixtures
    ], key=lambda x: x["kickoff_time"] or "")

    return {"success": True, "gw": gw, "fixtures": results}
