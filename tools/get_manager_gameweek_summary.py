from fastapi import APIRouter, HTTPException
import httpx
import asyncio
import time
import logging
from utils.bootstrap import get_cached_bootstrap

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"  # Adjust if needed

async def safe_api_call(client: httpx.AsyncClient, url: str, fallback_data=None, timeout: float = 10.0, params=None):
    """Safely make API call with fallback and timeout"""
    try:
        response = await client.get(url, timeout=timeout, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"API call failed: {url}, status: {response.status_code}")
            return fallback_data
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.error(f"API call error for {url}: {str(e)}")
        return fallback_data

def calculate_enhanced_captain_effectiveness(picks_enriched):
    """Enhanced captain effectiveness calculation"""
    captain_points = 0
    captain_actual_points = 0
    
    for p in picks_enriched:
        if p["is_captain"]:
            captain_points = p["points"]
            captain_actual_points = p["points"] * p["multiplier"]
            break
    
    max_points = max((p["points"] for p in picks_enriched if p["is_starting"]), default=0)
    
    return {
        "captain_points": captain_actual_points,
        "captain_base_points": captain_points,
        "max_possible_points": max_points,
        "difference": max_points - captain_points,
        "efficiency_percentage": round((captain_points / max_points * 100), 1) if max_points > 0 else 0,
        "verdict": "Optimal" if captain_points == max_points else "Suboptimal"
    }

@router.get("/tools/get_manager_gameweek_summary/{tid}/{gw}")
async def get_manager_gameweek_summary(tid: int, gw: int):
    """Enhanced gameweek summary with parallel processing and fixture difficulty"""
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Execute all API calls in parallel - MAJOR PERFORMANCE BOOST (now includes fixtures)
        picks_task = safe_api_call(client, f"{BASE_URL}/tools/get_manager_picks/{tid}/{gw}", {})
        history_task = safe_api_call(client, f"{BASE_URL}/tools/get_manager_history/{tid}", {})
        transfers_task = safe_api_call(client, f"{BASE_URL}/tools/get_transfers_by_gw/{tid}/{gw}", {})
        live_task = safe_api_call(client, f"{BASE_URL}/tools/get_event_live/{gw}", {})
        crowd_task = safe_api_call(client, f"{BASE_URL}/tools/get_crowd_trends_by_gw/{gw}", {})
        fixtures_task = safe_api_call(client, f"{BASE_URL}/tools/get_fixtures_by_gw", {}, params={"gw": gw})
        
        # Wait for all tasks to complete
        picks_data, history_data, transfers_data, live_data, crowd_data, fixtures_data = await asyncio.gather(
            picks_task, history_task, transfers_task, live_task, crowd_task, fixtures_task,
            return_exceptions=True
        )

    # Validate critical data with graceful degradation
    if not picks_data or "picks" not in picks_data:
        raise HTTPException(status_code=404, detail="Manager picks not found")

    # Get bootstrap data for player-team mapping
    try:
        bootstrap_data = await get_cached_bootstrap()
        player_team_map = {el["id"]: el["team"] for el in bootstrap_data["elements"]}
        player_position_map = {el["id"]: el["element_type"] for el in bootstrap_data["elements"]}
    except Exception as e:
        logger.error(f"Failed to get bootstrap data: {e}")
        player_team_map = {}
        player_position_map = {}

    # Build fixture difficulty mapping
    difficulty_map = {}
    fixtures_available = bool(fixtures_data and fixtures_data.get("fixtures"))
    
    if fixtures_available:
        fixtures = fixtures_data.get("fixtures", [])
        for f in fixtures:
            if f.get("finished") or f.get("started"):  # Include ongoing and finished fixtures
                # Handle the team structure from your fixtures endpoint
                team_h = f.get("team_h", {})
                team_a = f.get("team_a", {})
                
                team_h_id = team_h.get("id") if isinstance(team_h, dict) else team_h
                team_a_id = team_a.get("id") if isinstance(team_a, dict) else team_a
                
                if team_h_id:
                    difficulty_map[team_h_id] = f.get("team_h_difficulty", 3)
                if team_a_id:
                    difficulty_map[team_a_id] = f.get("team_a_difficulty", 3)
        
        logger.info(f"Built difficulty map for GW{gw}: {len(difficulty_map)} teams mapped")
    else:
        logger.warning(f"No fixtures data available for GW{gw}, fixture difficulty will be null")

    # Parse chips used this GW
    chips_used = []
    if history_data and "chips" in history_data:
        chips_used = [chip["name"] for chip in history_data["chips"] if chip.get("event") == gw]

    # Prepare live points dict with error handling
    live_points = {}
    if live_data and "players" in live_data:
        live_points = {e.get("player_id"): e.get("points", 0) for e in live_data["players"] if e.get("player_id")}

    # Enhanced picks enrichment with fixture difficulty
    picks_enriched = []
    for p in picks_data.get("picks", []):
        pid = p.get("player_id")
        if pid is None:
            continue
        
        points = live_points.get(pid, 0)
        position = p.get("position", 0)
        
        # Get fixture difficulty for this player
        player_team = player_team_map.get(pid)
        fixture_difficulty = difficulty_map.get(player_team) if player_team else None
        
        # Map position type
        position_type_map = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
        player_position = player_position_map.get(pid, 0)
        position_type = position_type_map.get(player_position, "Unknown")
        
        picks_enriched.append({
            "id": pid,
            "name": p.get("player_name", f"ID:{pid}"),
            "multiplier": p.get("multiplier", 1),
            "is_captain": p.get("is_captain", False),
            "is_vice_captain": p.get("is_vice_captain", False),
            "position": position,
            "points": points,
            "is_starting": position <= 11,
            "is_bench": position > 11,
            "fixture_difficulty": fixture_difficulty,
            "position_type": position_type,
            "team_id": player_team
        })

    # Enhanced captain effectiveness
    captain_effectiveness = calculate_enhanced_captain_effectiveness(picks_enriched)

    # Transfer cost summary with better error handling
    transfers_this_gw = []
    total_transfer_cost = 0
    
    if transfers_data and "transfers" in transfers_data:
        transfers_this_gw = transfers_data["transfers"]
        total_transfer_cost = sum(t.get("cost", 0) for t in transfers_this_gw)

    # Compose transfers with names using cached bootstrap
    player_map = {}
    if bootstrap_data:
        player_map = {el["id"]: el["web_name"] for el in bootstrap_data["elements"]}
    
    transfers_enriched = []
    for t in transfers_this_gw:
        transfers_enriched.append({
            "element_in": player_map.get(t.get("element_in"), f"ID:{t.get('element_in')}"),
            "element_out": player_map.get(t.get("element_out"), f"ID:{t.get('element_out')}"),
            "cost": t.get("cost", 0),
        })

    # Calculate total points
    total_points = sum(p["points"] * p["multiplier"] for p in picks_enriched if p["is_starting"])
    
    # Calculate bench points
    bench_points = sum(p["points"] for p in picks_enriched if p["is_bench"])
    
    # Performance metrics
    processing_time = (time.time() - start_time) * 1000

    return {
        "manager_id": tid,
        "gameweek": gw,
        "chips_used": chips_used,
        "picks": picks_enriched,
        "transfers": transfers_enriched,
        "total_transfer_cost": total_transfer_cost,
        "captain_effectiveness": captain_effectiveness,
        "crowd_trends": crowd_data or {},
        "total_points": total_points,
        "bench_points": bench_points,
        "processing_time_ms": round(processing_time, 2),
        "data_quality": {
            "picks_available": bool(picks_data),
            "live_data_available": bool(live_data),
            "history_available": bool(history_data),
            "transfers_available": bool(transfers_data),
            "crowd_data_available": bool(crowd_data),
            "fixtures_available": fixtures_available,
            "bootstrap_available": bool(bootstrap_data)
        }
    }