from datetime import datetime
from collections import Counter, defaultdict
from fastapi import APIRouter, HTTPException
import httpx
import asyncio
import time
import logging
import os

from utils.bootstrap import get_cached_bootstrap
from utils.mapping import build_team_map, build_position_map

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("INTERNAL_API_URL", "http://localhost:8000")

async def safe_api_call(client: httpx.AsyncClient, url: str, fallback_data=None, timeout: float = 15.0, params=None):
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

def analyze_upcoming_fixtures(player_team_map, upcoming_fixtures, next_gws=5):
    """Analyze fixture difficulty for upcoming 5 gameweeks"""
    fixture_analysis = defaultdict(lambda: {"fixtures": [], "avg_difficulty": 0, "easy_fixtures": 0, "hard_fixtures": 0})
    
    for fixture in upcoming_fixtures:
        gw = fixture.get("event")
        if not gw:
            continue
            
        team_h = fixture.get("team_h", {})
        team_a = fixture.get("team_a", {})
        
        team_h_id = team_h.get("id") if isinstance(team_h, dict) else team_h
        team_a_id = team_a.get("id") if isinstance(team_a, dict) else team_a
        
        if team_h_id:
            difficulty = fixture.get("team_h_difficulty", 3)
            fixture_analysis[team_h_id]["fixtures"].append({
                "gameweek": gw,
                "opponent": team_a.get("name", f"Team {team_a_id}") if isinstance(team_a, dict) else f"Team {team_a_id}",
                "home": True,
                "difficulty": difficulty
            })
            
        if team_a_id:
            difficulty = fixture.get("team_a_difficulty", 3)
            fixture_analysis[team_a_id]["fixtures"].append({
                "gameweek": gw,
                "opponent": team_h.get("name", f"Team {team_h_id}") if isinstance(team_h, dict) else f"Team {team_h_id}",
                "home": False,
                "difficulty": difficulty
            })
    
    # Calculate summary stats for each team
    for team_id, data in fixture_analysis.items():
        fixtures = data["fixtures"]
        if fixtures:
            difficulties = [f["difficulty"] for f in fixtures]
            data["avg_difficulty"] = round(sum(difficulties) / len(difficulties), 1)
            data["easy_fixtures"] = sum(1 for d in difficulties if d <= 2)
            data["hard_fixtures"] = sum(1 for d in difficulties if d >= 4)
            data["fixture_run_quality"] = (
                "Excellent" if data["avg_difficulty"] <= 2.5 else 
                "Good" if data["avg_difficulty"] <= 3.0 else 
                "Average" if data["avg_difficulty"] <= 3.5 else 
                "Difficult"
            )
    
    return fixture_analysis

def recommend_transfers(current_picks, fixture_analysis, bootstrap_data, free_transfers=1):
    """Enhanced transfer recommendations based on 5-gameweek fixture runs"""
    # Get player data
    element_prices = {el["id"]: el["now_cost"] / 10 for el in bootstrap_data["elements"]}
    element_teams = {el["id"]: el["team"] for el in bootstrap_data["elements"]}
    element_positions = {el["id"]: el["element_type"] for el in bootstrap_data["elements"]}
    element_names = {el["id"]: el["web_name"] for el in bootstrap_data["elements"]}
    
    # Analyze current players
    players_out = []
    players_to_watch = []
    
    for pick in current_picks:
        if pick["is_starting"]:
            player_id = pick["id"]
            team_id = element_teams.get(player_id)
            
            if team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                avg_difficulty = team_fixtures["avg_difficulty"]
                hard_fixtures = team_fixtures["hard_fixtures"]
                fixture_run_quality = team_fixtures["fixture_run_quality"]
                
                player_analysis = {
                    "player": pick,
                    "avg_difficulty": avg_difficulty,
                    "hard_fixtures_count": hard_fixtures,
                    "fixture_run_quality": fixture_run_quality,
                    "upcoming_fixtures": [f["difficulty"] for f in team_fixtures["fixtures"][:5]]
                }
                
                # Flag for transfer out if poor fixtures
                if avg_difficulty >= 3.8 or hard_fixtures >= 3:
                    players_out.append(player_analysis)
                elif avg_difficulty >= 3.3 or hard_fixtures >= 2:
                    players_to_watch.append(player_analysis)
    
    # Find transfer targets with good fixture runs
    transfer_targets = []
    for element in bootstrap_data["elements"]:
        team_id = element["team"]
        if team_id in fixture_analysis:
            team_fixtures = fixture_analysis[team_id]
            if team_fixtures["avg_difficulty"] <= 2.8 and team_fixtures["easy_fixtures"] >= 3:
                transfer_targets.append({
                    "name": element["web_name"],
                    "position": element["element_type"],
                    "team": team_id,
                    "price": element["now_cost"] / 10,
                    "avg_difficulty": team_fixtures["avg_difficulty"],
                    "easy_fixtures": team_fixtures["easy_fixtures"],
                    "fixture_run_quality": team_fixtures["fixture_run_quality"]
                })
    
    # Sort transfer targets by fixture quality and price
    transfer_targets.sort(key=lambda x: (x["avg_difficulty"], -x["easy_fixtures"]))
    
    # Sort players out by worst fixtures
    players_out.sort(key=lambda x: (-x["avg_difficulty"], -x["hard_fixtures_count"]))
    
    return {
        "transfer_out_candidates": players_out[:5],  # Top 5 to consider selling
        "players_to_monitor": players_to_watch[:3],  # Keep an eye on these
        "transfer_targets": transfer_targets[:10],  # Top 10 targets with good fixtures
        "free_transfers": free_transfers,
        "recommendation": get_transfer_recommendation(players_out, transfer_targets, free_transfers)
    }

def get_transfer_recommendation(players_out, transfer_targets, free_transfers):
    """Generate transfer recommendation text"""
    if not players_out:
        return "Hold transfers - no urgent moves needed based on fixtures"
    
    if len(players_out) >= 2 and free_transfers >= 1:
        return f"Priority: Transfer out {players_out[0]['player']['name']} - poor fixture run ({players_out[0]['fixture_run_quality']})"
    elif len(players_out) >= 1:
        return f"Consider transferring {players_out[0]['player']['name']} - difficult upcoming fixtures"
    
    return "Monitor fixture developments"

def recommend_captain(current_picks, fixture_analysis, crowd_trends):
    """Enhanced captain recommendation based on fixture runs"""
    captain_candidates = []
    
    # Get template players for safer options
    template_players = set()
    most_selected = crowd_trends.get("most_selected_by_position", {})
    for position, players in most_selected.items():
        for player_data in players[:2]:
            if float(player_data["selected_by_percent"]) > 20:
                template_players.add(player_data["player"])
    
    for pick in current_picks:
        if pick["is_starting"]:
            # Enhanced scoring based on fixture run
            base_score = pick.get("points", 0)
            fixture_bonus = 0
            consistency_bonus = 0
            next_fixture_difficulty = 3
            
            # Get team fixture analysis
            team_id = pick.get("team_id")
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                next_fixture_difficulty = team_fixtures["fixtures"][0]["difficulty"] if team_fixtures["fixtures"] else 3
                
                # Immediate fixture bonus (next GW)
                fixture_bonus = (6 - next_fixture_difficulty) * 3  # Max 15 bonus for FDR 1
                
                # Fixture run bonus (next 3-4 GWs)
                if team_fixtures["easy_fixtures"] >= 2:
                    consistency_bonus = 5
                elif team_fixtures["avg_difficulty"] <= 2.5:
                    consistency_bonus = 3
            
            # Template bonus for safer picks
            template_bonus = 5 if pick["name"] in template_players else 0
            
            captain_score = base_score + fixture_bonus + consistency_bonus + template_bonus
            
            captain_candidates.append({
                "name": pick["name"],
                "position_type": pick.get("position_type", "Unknown"),
                "captain_score": captain_score,
                "next_fixture_difficulty": next_fixture_difficulty,
                "fixture_run_quality": fixture_analysis[team_id]["fixture_run_quality"] if team_id and team_id in fixture_analysis else "Unknown",
                "is_template": pick["name"] in template_players,
                "reasoning": f"Score: {captain_score} (Form: {base_score}, Next fixture: +{fixture_bonus}, Run: +{consistency_bonus}, Template: +{template_bonus})"
            })
    
    # Sort by captain score
    captain_candidates.sort(key=lambda x: x["captain_score"], reverse=True)
    
    return {
        "top_captain_pick": captain_candidates[0] if captain_candidates else None,
        "alternatives": captain_candidates[1:4],  # Next 3 options
        "captain_rotation_plan": get_captain_rotation_plan(captain_candidates, fixture_analysis),
        "recommendation": f"Captain {captain_candidates[0]['name']} - {captain_candidates[0]['reasoning']}" if captain_candidates else "No clear captain recommendation"
    }

def get_captain_rotation_plan(captain_candidates, fixture_analysis):
    """Suggest captain rotation over next 3 gameweeks"""
    rotation_plan = []
    
    # Get top 3 captain options
    top_captains = captain_candidates[:3]
    
    for gw_offset in range(3):  # Next 3 gameweeks
        best_captain_for_gw = None
        best_score = 0
        
        for candidate in top_captains:
            # Need to get team_id somehow - let's assume it's available
            team_id = candidate.get("team_id")
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]["fixtures"]
                if gw_offset < len(team_fixtures):
                    fixture_difficulty = team_fixtures[gw_offset]["difficulty"]
                    gw_score = candidate["captain_score"] + (6 - fixture_difficulty) * 2
                    
                    if gw_score > best_score:
                        best_score = gw_score
                        best_captain_for_gw = {
                            "name": candidate["name"],
                            "difficulty": fixture_difficulty,
                            "opponent": team_fixtures[gw_offset]["opponent"]
                        }
        
        if best_captain_for_gw:
            rotation_plan.append(best_captain_for_gw)
    
    return rotation_plan

def analyze_chip_timing(current_picks, fixture_analysis, bench_points=0):
    """Enhanced chip timing analysis based on 5-gameweek fixture runs"""
    chip_recommendations = {}
    
    # Bench Boost analysis
    if bench_points >= 15:
        chip_recommendations["bench_boost"] = {
            "recommended": True,
            "reason": f"Strong bench with {bench_points} points - good BB opportunity",
            "timing": "This gameweek"
        }
    elif bench_points >= 10:
        chip_recommendations["bench_boost"] = {
            "recommended": False,
            "reason": f"Decent bench ({bench_points} pts) but wait for stronger opportunity",
            "timing": "Wait for double gameweek or stronger bench"
        }
    else:
        chip_recommendations["bench_boost"] = {
            "recommended": False,
            "reason": f"Weak bench ({bench_points} pts) - improve before using BB",
            "timing": "Build stronger bench first"
        }
    
    # Enhanced Triple Captain analysis
    starting_players = [p for p in current_picks if p.get("is_starting", False)]
    if starting_players:
        # Find best TC candidate based on fixture run
        best_tc_candidate = None
        best_tc_score = 0
        
        for player in starting_players:
            team_id = player.get("team_id")
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                
                # Score based on recent form + fixture run quality
                form_score = player.get("points", 0)
                fixture_score = 0
                
                # Next 3 fixtures for TC timing
                next_3_fixtures = team_fixtures["fixtures"][:3]
                avg_difficulty = 3
                if next_3_fixtures:
                    avg_difficulty = sum(f["difficulty"] for f in next_3_fixtures) / len(next_3_fixtures)
                    fixture_score = (4 - avg_difficulty) * 5  # Bonus for easy fixtures
                
                tc_score = form_score + fixture_score
                if tc_score > best_tc_score:
                    best_tc_score = tc_score
                    best_tc_candidate = {
                        "name": player["name"],
                        "tc_score": tc_score,
                        "avg_difficulty": round(avg_difficulty, 1),
                        "fixture_run": team_fixtures["fixture_run_quality"]
                    }
        
        if best_tc_candidate and best_tc_candidate["avg_difficulty"] <= 2.5:
            chip_recommendations["triple_captain"] = {
                "recommended": True,
                "reason": f"{best_tc_candidate['name']} has excellent fixture run ({best_tc_candidate['fixture_run']})",
                "timing": "Consider this gameweek or next",
                "best_candidate": best_tc_candidate
            }
        else:
            chip_recommendations["triple_captain"] = {
                "recommended": False,
                "reason": "Wait for better fixture combination for premium players",
                "timing": "Hold for easier fixture run",
                "best_candidate": best_tc_candidate
            }
    
    # Free Hit analysis
    # Count players with difficult fixtures
    players_with_bad_fixtures = sum(1 for p in starting_players 
                                   if p.get("team_id") and p["team_id"] in fixture_analysis 
                                   and fixture_analysis[p["team_id"]]["avg_difficulty"] >= 4)
    
    if players_with_bad_fixtures >= 6:
        chip_recommendations["free_hit"] = {
            "recommended": True,
            "reason": f"{players_with_bad_fixtures} players have very difficult fixtures",
            "timing": "Consider for this difficult gameweek"
        }
    else:
        chip_recommendations["free_hit"] = {
            "recommended": False,
            "reason": "Most players have manageable fixtures",
            "timing": "Save for blank/double gameweeks"
        }
    
    return chip_recommendations

def optimize_formation(current_picks, all_upcoming_fixtures):
    """Suggest optimal formation based on upcoming fixtures"""
    starting_players = [p for p in current_picks if p.get("is_starting", False)]
    
    # Count players by position with good fixtures (FDR <= 3)
    good_fixtures_by_position = Counter()
    total_by_position = Counter()
    
    for player in starting_players:
        pos_type = player.get("position_type", "Unknown")
        difficulty = player.get("fixture_difficulty", 3)
        
        total_by_position[pos_type] += 1
        if difficulty <= 3:
            good_fixtures_by_position[pos_type] += 1
    
    # Remove goalkeeper from formation string
    def_count = total_by_position.get("Defender", 0)
    mid_count = total_by_position.get("Midfielder", 0) 
    fwd_count = total_by_position.get("Forward", 0)
    
    current_formation = f"{def_count}-{mid_count}-{fwd_count}"
    
    formation_analysis = {
        "current_formation": current_formation,
        "formation_strength": {
            "defenders_good_fixtures": good_fixtures_by_position.get("Defender", 0),
            "midfielders_good_fixtures": good_fixtures_by_position.get("Midfielder", 0),
            "forwards_good_fixtures": good_fixtures_by_position.get("Forward", 0)
        },
        "recommendation": "Current formation looks good" if sum(good_fixtures_by_position.values()) >= 8 else "Consider formation change based on fixtures"
    }
    
    return formation_analysis

def get_fixture_run_summary(fixture_analysis, current_picks):
    """Generate summary of fixture runs for current squad"""
    summary = {
        "excellent_fixtures": [],
        "difficult_fixtures": [],
        "overall_assessment": ""
    }
    
    starting_players = [p for p in current_picks if p.get("is_starting", False)]
    
    for player in starting_players:
        team_id = player.get("team_id")
        if team_id and team_id in fixture_analysis:
            team_fixtures = fixture_analysis[team_id]
            
            if team_fixtures["fixture_run_quality"] in ["Excellent", "Good"]:
                summary["excellent_fixtures"].append({
                    "player": player["name"],
                    "quality": team_fixtures["fixture_run_quality"],
                    "avg_difficulty": team_fixtures["avg_difficulty"]
                })
            elif team_fixtures["fixture_run_quality"] == "Difficult":
                summary["difficult_fixtures"].append({
                    "player": player["name"],
                    "quality": team_fixtures["fixture_run_quality"],
                    "avg_difficulty": team_fixtures["avg_difficulty"]
                })
    
    # Overall assessment
    good_fixtures = len(summary["excellent_fixtures"])
    bad_fixtures = len(summary["difficult_fixtures"])
    
    if good_fixtures >= 8:
        summary["overall_assessment"] = "Excellent fixture period - great time for points"
    elif good_fixtures >= 6:
        summary["overall_assessment"] = "Good fixture period - expect decent returns"
    elif bad_fixtures >= 6:
        summary["overall_assessment"] = "Difficult fixture period - consider transfers/chips"
    else:
        summary["overall_assessment"] = "Mixed fixture period - selective captaincy needed"
    
    return summary

@router.get("/tools/get_upcoming_gameweek_planner/{tid}/{next_gw}")
async def get_upcoming_gameweek_planner(tid: int, next_gw: int, gameweeks_ahead: int = 3):
    """Enhanced gameweek planner with configurable gameweeks ahead (1-5 gameweeks)"""
    start_time = time.time()
    
    # Validate gameweeks_ahead parameter
    gameweeks_ahead = max(1, min(5, gameweeks_ahead))  # Limit between 1-5 gameweeks
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get current team data
        current_summary_task = safe_api_call(client, f"{BASE_URL}/tools/get_manager_gameweek_summary/{tid}/{next_gw-1}", {})
        manager_info_task = safe_api_call(client, f"{BASE_URL}/tools/get_manager_info/{tid}", {})
        
        # Fetch fixtures for specified number of gameweeks ahead
        fixture_tasks = []
        for gw_offset in range(gameweeks_ahead):
            target_gw = next_gw + gw_offset
            fixture_tasks.append(
                safe_api_call(client, f"{BASE_URL}/tools/get_fixtures_by_gw", {}, params={"gw": target_gw})
            )
        
        # Execute all tasks
        all_tasks = [current_summary_task, manager_info_task] + fixture_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        current_summary = results[0]
        manager_info = results[1]
        fixture_results = results[2:]

    # Validate data
    if not current_summary or "picks" not in current_summary:
        raise HTTPException(status_code=404, detail="Could not retrieve current team data")
    
    # Get bootstrap data
    try:
        bootstrap = await get_cached_bootstrap()
        player_team_map = {el["id"]: el["team"] for el in bootstrap["elements"]}
    except Exception as e:
        logger.error(f"Failed to get bootstrap data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve player data")

    current_picks = current_summary.get("picks", [])
    
    # Combine all upcoming fixtures from specified gameweeks
    all_upcoming_fixtures = []
    analyzed_gameweeks = []
    
    for i, fixture_result in enumerate(fixture_results):
        target_gw = next_gw + i
        if fixture_result and "fixtures" in fixture_result:
            all_upcoming_fixtures.extend(fixture_result["fixtures"])
            analyzed_gameweeks.append(target_gw)
        else:
            logger.warning(f"No fixtures data for GW{target_gw}")
    
    if not all_upcoming_fixtures:
        raise HTTPException(status_code=404, detail=f"No fixture data available for gameweeks {next_gw}-{next_gw + gameweeks_ahead - 1}")
    
    # Enhanced analysis with configurable gameweek view
    fixture_analysis = analyze_upcoming_fixtures(player_team_map, all_upcoming_fixtures, next_gws=gameweeks_ahead)
    
    # Get manager data for transfer info
    free_transfers = 1  # Default
    if manager_info and "current_event_transfers" in manager_info:
        transfers_made = manager_info.get("current_event_transfers", 0)
        free_transfers = max(1 - transfers_made, 0)
    
    # Enhanced recommendations with configurable gameweek view
    transfer_recommendations = recommend_transfers(current_picks, fixture_analysis, bootstrap, free_transfers)
    
    captain_recommendation = recommend_captain(current_picks, fixture_analysis, current_summary.get("crowd_trends", {}))
    
    bench_points = current_summary.get("bench_points", 0)
    chip_analysis = analyze_chip_timing(current_picks, fixture_analysis, bench_points)
    
    formation_optimization = optimize_formation(current_picks, all_upcoming_fixtures)
    
    # Enhanced planning summary with fixture run insights
    planning_summary = {
        "gameweek": next_gw,
        "gameweeks_analyzed": analyzed_gameweeks,
        "gameweeks_ahead": gameweeks_ahead,
        "free_transfers": free_transfers,
        "priority_actions": [],
        "key_insights": [],
        "fixture_run_summary": get_fixture_run_summary(fixture_analysis, current_picks)
    }
    
    # Add priority actions based on enhanced analysis
    if transfer_recommendations["transfer_out_candidates"]:
        top_transfer_out = transfer_recommendations["transfer_out_candidates"][0]
        planning_summary["priority_actions"].append(
            f"Transfer out {top_transfer_out['player']['name']} - {top_transfer_out['fixture_run_quality']} fixture run"
        )
    
    if chip_analysis.get("triple_captain", {}).get("recommended"):
        tc_candidate = chip_analysis["triple_captain"].get("best_candidate", {})
        planning_summary["priority_actions"].append(
            f"Triple Captain opportunity: {tc_candidate.get('name', 'premium player')} has excellent fixtures"
        )
    
    if chip_analysis.get("bench_boost", {}).get("recommended"):
        planning_summary["priority_actions"].append("Strong bench boost opportunity this gameweek")
    
    if captain_recommendation["top_captain_pick"]:
        planning_summary["priority_actions"].append(
            f"Captain {captain_recommendation['top_captain_pick']['name']} - {captain_recommendation['top_captain_pick']['fixture_run_quality']} fixtures"
        )
    
    # Add key insights
    if transfer_recommendations["transfer_targets"]:
        top_target = transfer_recommendations["transfer_targets"][0]
        planning_summary["key_insights"].append(
            f"Best transfer target: {top_target['name']} ({top_target['fixture_run_quality']} fixture run)"
        )
    
    if captain_recommendation.get("captain_rotation_plan"):
        planning_summary["key_insights"].append("Captain rotation plan available for optimal fixtures")
    
    # Add strategic insight based on gameweeks ahead
    if gameweeks_ahead == 1:
        planning_summary["key_insights"].append("Short-term analysis - focus on immediate decisions")
    elif gameweeks_ahead <= 3:
        planning_summary["key_insights"].append("Medium-term analysis - good for transfer planning")
    else:
        planning_summary["key_insights"].append("Long-term analysis - ideal for chip timing and strategy")
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000

    return {
        "meta": {
            "analysis_type": "configurable_upcoming_gameweek_planner",
            "manager_id": tid,
            "target_gameweek": next_gw,
            "gameweeks_analyzed": analyzed_gameweeks,
            "gameweeks_ahead": gameweeks_ahead,
            "generated_at": datetime.utcnow().isoformat(),
            "data_sources": ["manager_gameweek_summary", "fixtures", "bootstrap", "manager_info"],
            "processing_time_ms": round(processing_time, 2)
        },
        "planning_summary": planning_summary,
        "transfer_recommendations": transfer_recommendations,
        "captain_recommendation": captain_recommendation,
        "chip_analysis": chip_analysis,
        "formation_optimization": formation_optimization,
        "upcoming_fixtures_overview": {
            "total_fixtures_analyzed": len(all_upcoming_fixtures),
            "gameweeks_covered": analyzed_gameweeks,
            "fixtures_available": bool(all_upcoming_fixtures)
        }
    }