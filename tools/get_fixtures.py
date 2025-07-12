from fastapi import APIRouter
import httpx
from utils.mapping import build_team_map, map_team
from utils.bootstrap import get_cached_bootstrap  # ✅ cache

router = APIRouter()

@router.get("/tools/get_fixtures")
async def get_fixtures():
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"

    async with httpx.AsyncClient() as client:
        fixtures_res = await client.get(fixtures_url)
        fixtures = fixtures_res.json()

    bootstrap = await get_cached_bootstrap()
    team_map = build_team_map(bootstrap)

    results = [
        {
            "event": f.get("event"),
            "kickoff_time": f.get("kickoff_time"),
            "team_h": map_team(f.get("team_h"), team_map),
            "team_a": map_team(f.get("team_a"), team_map),
            "finished": f.get("finished")
        }
        for f in fixtures
    ]

    return {"success": True, "fixtures": results}
