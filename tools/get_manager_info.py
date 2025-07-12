from fastapi import APIRouter, Path
import httpx

router = APIRouter()

@router.get("/tools/get_manager_info/{tid}")
async def get_manager_info(tid: int = Path(..., description="FPL team ID")):
    url = f"https://fantasy.premierleague.com/api/entry/{tid}/"

    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        if res.status_code != 200:
            return {"success": False, "error": "Manager not found"}

        data = res.json()

    return {
        "success": True,
        "team_id": tid,
        "team_name": data.get("name"),
        "manager_name": f"{data.get('player_first_name')} {data.get('player_last_name')}",
        "region": data.get("player_region_name"),
        "started_event": data.get("started_event"),
        "favourite_team_id": data.get("favorite_team"),
        "summary_overall_points": data.get("summary_overall_points"),
        "summary_overall_rank": data.get("summary_overall_rank"),
        "summary_event_points": data.get("summary_event_points"),
        "summary_event_rank": data.get("summary_event_rank"),
    }
