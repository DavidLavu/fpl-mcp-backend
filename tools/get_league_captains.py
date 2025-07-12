from fastapi import APIRouter, HTTPException
import httpx

from .get_rival_comparison import fetch_all_league_entries, fetch_manager_picks

router = APIRouter()
FPL_BASE = "https://fantasy.premierleague.com/api"

POSITION_MAP = {
    1: "Goalkeeper",
    2: "Defender",
    3: "Midfielder",
    4: "Forward",
}

@router.get("/tools/get_league_captains/{lid}/{gw}")
async def get_league_captains(lid: int, gw: int):
    # 1) Fetch league metadata (to get name)
    async with httpx.AsyncClient() as client:
        meta_resp = await client.get(f"{FPL_BASE}/leagues-classic/{lid}/standings/")
        if meta_resp.status_code != 200:
            raise HTTPException(
                status_code=meta_resp.status_code,
                detail="Failed to fetch league metadata"
            )
        league_data = meta_resp.json().get("league", {})
        league_name = league_data.get("name", f"League {lid}")

        # 2) Fetch bootstrap to build player → (name, position, club) map
        bs_resp = await client.get(f"{FPL_BASE}/bootstrap-static/")
        if bs_resp.status_code != 200:
            raise HTTPException(
                status_code=bs_resp.status_code,
                detail="Failed to fetch bootstrap data"
            )
        bs = bs_resp.json()
        team_map = {t["id"]: t["name"] for t in bs["teams"]}
        player_info = {
            e["id"]: {
                "name": e["web_name"],
                "position": POSITION_MAP.get(e["element_type"], "Unknown"),
                "club": team_map.get(e["team"], "Unknown"),
            }
            for e in bs["elements"]
        }

    # 3) Get all entries (up to first 20 for perf)
    try:
        entries = await fetch_all_league_entries(lid)
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Failed to fetch league entries: {e.detail}"
        )
    entries = entries[:20]
    total_managers = len(entries)

    # 4) Tally captain counts
    cap_counts: dict[int, int] = {}
    for entry in entries:
        picks_data = await fetch_manager_picks(entry["entry"], gw)
        for p in picks_data.get("picks", []):
            if p.get("is_captain"):
                pid = p["player_id"]
                cap_counts[pid] = cap_counts.get(pid, 0) + 1

    # 5) Build sorted list and enrich with name/position/club/percentage
    most_captained = []
    for pid, cnt in sorted(cap_counts.items(), key=lambda kv: -kv[1]):
        info = player_info.get(pid, {})
        percent = round(cnt / total_managers * 100, 1) if total_managers else 0.0
        most_captained.append({
            "id": pid,
            "name": info.get("name", "Unknown"),
            "count": cnt,
            "percent": percent,                   # New field
            "position": info.get("position", "Unknown"),
            "club": info.get("club", "Unknown"),
        })

    return {
        "league_id": lid,
        "league_name": league_name,
        "gameweek": gw,
        "most_captained": most_captained
    }
