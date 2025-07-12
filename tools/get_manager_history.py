from fastapi import APIRouter, Path
import httpx

router = APIRouter()

@router.get("/tools/get_manager_history/{tid}")
async def get_manager_history(tid: int = Path(..., description="FPL team ID")):
    url = f"https://fantasy.premierleague.com/api/entry/{tid}/history/"

    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        if res.status_code != 200:
            return {"success": False, "error": "Manager history not found"}

        data = res.json()

    return {
        "success": True,
        "team_id": tid,
        "chips": data.get("chips", []),
        "past_seasons": data.get("past", []),
        "current_season": data.get("current", [])
    }
