from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()

FPL_BASE = "https://fantasy.premierleague.com/api"

async def fetch_all_league_entries(lid: int):
    entries = []
    page = 1
    async with httpx.AsyncClient() as client:
        while True:
            url = f"{FPL_BASE}/leagues-classic/{lid}/standings/?page_standings={page}"
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Failed to fetch league standings")
            data = resp.json()
            results = data.get("standings", {}).get("results", [])
            if not results:
                break
            entries.extend(results)
            if not data["standings"].get("has_next", False):
                break
            page += 1
    return entries

async def fetch_manager_picks(tid: int, gw: int):
    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8000/tools/get_manager_picks/{tid}/{gw}"
        resp = await client.get(url)
        return resp.json() if resp.status_code == 200 else {}

async def fetch_event_live_data(gw: int):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{FPL_BASE}/event/{gw}/live/")
        return resp.json()

def get_player_points_map(event_data):
    return {
        player["id"]: player["stats"].get("total_points", 0)
        for player in event_data.get("elements", [])
    }

@router.get("/tools/get_rival_comparison/{lid}/{gw}")
async def get_rival_comparison(lid: int, gw: int):
    entries = await fetch_all_league_entries(lid)
    entries = entries[:20]  # Limit for performance

    event_data = await fetch_event_live_data(gw)
    points_map = get_player_points_map(event_data)

    enriched = []
    cap_counts = {}

    # Build raw enriched list
    for entry in entries:
        tid = entry["entry"]
        picks_data = await fetch_manager_picks(tid, gw)
        picks = picks_data.get("picks", [])
        active_chip = picks_data.get("active_chip")

        captain = next((p for p in picks if p.get("is_captain")), None)
        vice = next((p for p in picks if p.get("is_vice_captain")), None)

        cap_id = captain["player_id"] if captain else None
        vice_id = vice["player_id"] if vice else None
        cap_name = captain["player_name"] if captain else None
        vice_name = vice["player_name"] if vice else None

        cap_counts[cap_id] = cap_counts.get(cap_id, 0) + 1 if cap_id else 0

        enriched.append({
            "manager_id": tid,
            "manager_name": entry["entry_name"],
            "player_name": entry["player_name"],
            "total_points": entry["total"],
            "active_chip": active_chip,
            "captain_id": cap_id,
            "captain_name": cap_name,
            "vice_id": vice_id,
            "vice_name": vice_name,
            "picks": picks
        })

    # Identify top manager baseline
    top = enriched[0]
    top_cap_id = top["captain_id"]
    top_cap_points = points_map.get(top_cap_id, 0)

    # Enrich each rival with captain swing and sorted differentials
    for m in enriched:
        m_cap_points = points_map.get(m["captain_id"], 0)
        m["captain"] = {
            "id": m["captain_id"],
            "name": m["captain_name"],
            "points": m_cap_points
        }
        m["captain_differs"] = m["captain_id"] != top_cap_id
        m["captain_point_swing_vs_top"] = (m_cap_points - top_cap_points) * 2

        top_ids = {p["player_id"] for p in top["picks"]}
        m_ids = {p["player_id"] for p in m["picks"]}
        diffs = m_ids - top_ids

        # Build and sort differentials by points desc
        diffs_list = [
            {
                "id": d,
                "name": next((p["player_name"] for p in m["picks"] if p["player_id"] == d), "Unknown"),
                "points": points_map.get(d, 0)
            } for d in diffs
        ]
        m["differentials"] = sorted(diffs_list, key=lambda x: -x["points"])

    return {
        "league_id": lid,
        "gameweek": gw,
        "most_captained": [
            {"id": cid, "name": next((m["captain_name"] for m in enriched if m["captain_id"] == cid), "Unknown"), "count": count}
            for cid, count in sorted(cap_counts.items(), key=lambda x: -x[1])
        ],
        "top_manager": {
            "name": top["manager_name"],
            "player_name": top["player_name"],
            "captain": {
                "id": top_cap_id,
                "name": top["captain_name"],
                "points": top_cap_points
            },
            "active_chip": top["active_chip"]
        },
        "rival_analysis": enriched
    }
