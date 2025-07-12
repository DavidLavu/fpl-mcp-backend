from datetime import datetime
from collections import Counter
from fastapi import APIRouter, HTTPException
import httpx
import time
import logging

from utils.bootstrap import get_cached_bootstrap
from utils.mapping import build_position_map

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

async def safe_api_call(client: httpx.AsyncClient, url: str, fallback_data=None, timeout: float = 15.0):
    """Safely make API call with fallback and timeout"""
    try:
        response = await client.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"API call failed: {url}, status: {response.status_code}")
            return fallback_data
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.error(f"API call error for {url}: {str(e)}")
        return fallback_data

def analyze_formation(picks):
    """Analyze team formation based on positions"""
    starting_players = [p for p in picks if p["is_starting"]]
    formation_count = Counter()
    
    # Map position types correctly
    position_mapping = {
        "Goalkeeper": "GKP",
        "Defender": "DEF", 
        "Midfielder": "MID",
        "Forward": "FWD"
    }
    
    for p in starting_players:
        pos_type = p.get("position_type", "Unknown")
        mapped_pos = position_mapping.get(pos_type, pos_type)
        if mapped_pos != "GKP":  # Exclude goalkeeper from formation
            formation_count[mapped_pos] += 1
    
    formation_string = f"{formation_count.get('DEF', 0)}-{formation_count.get('MID', 0)}-{formation_count.get('FWD', 0)}"
    
    return {
        "formation": formation_string,
        "defender_count": formation_count.get('DEF', 0),
        "midfielder_count": formation_count.get('MID', 0),
        "forward_count": formation_count.get('FWD', 0),
        "is_balanced": all(count >= 2 for pos, count in formation_count.items()),
        "formation_type": get_formation_analysis(formation_string)
    }

def get_formation_analysis(formation_string):
    """Provide tactical analysis of formation"""
    formation_insights = {
        "3-4-3": "Attacking formation with wing-backs, high risk/reward",
        "3-5-2": "Midfield-heavy, good for controlling games",
        "4-3-3": "Balanced attacking formation, popular choice",
        "4-4-2": "Classic balanced formation, solid defensive structure",
        "4-5-1": "Defensive formation, prioritizes clean sheets",
        "5-3-2": "Very defensive, focuses on defensive returns",
        "5-4-1": "Ultra-defensive, minimal attacking threat"
    }
    return formation_insights.get(formation_string, "Custom formation")

def calculate_value_efficiency(picks, bootstrap_data):
    """Calculate points per million for each player"""
    element_prices = {el["id"]: el["now_cost"] / 10 for el in bootstrap_data["elements"]}  # Convert to millions
    
    efficiency_data = []
    total_starting_value = 0
    total_starting_points = 0
    
    for p in picks:
        if p["is_starting"]:
            price = element_prices.get(p["id"], 0)
            points = p["points"] * p["multiplier"]
            ppm = points / price if price > 0 else 0
            
            efficiency_data.append({
                "name": p["name"],
                "price": price,
                "points": points,
                "points_per_million": round(ppm, 2),
                "position_type": p.get("position_type", "Unknown")
            })
            
            total_starting_value += price
            total_starting_points += points
    
    # Sort by efficiency
    efficiency_data.sort(key=lambda x: x["points_per_million"], reverse=True)
    
    return {
        "players": efficiency_data,
        "team_average_ppm": round(total_starting_points / total_starting_value, 2) if total_starting_value > 0 else 0,
        "best_value": efficiency_data[0] if efficiency_data else None,
        "worst_value": efficiency_data[-1] if efficiency_data else None,
        "total_starting_value": round(total_starting_value, 1)
    }

def analyze_template_vs_differential(picks, crowd_trends):
    """Analyze how template vs differential the team is"""
    template_players = set()
    
    # Extract popular players from crowd trends (top 2 in each position)
    most_selected = crowd_trends.get("most_selected_by_position", {})
    for position, players in most_selected.items():
        for player_data in players[:2]:  # Top 2 most selected
            if float(player_data["selected_by_percent"]) > 20:  # 20%+ ownership
                template_players.add(player_data["player"])
    
    starting_picks = [p for p in picks if p["is_starting"]]
    template_picks = []
    differential_picks = []
    
    for p in starting_picks:
        if p["name"] in template_players:
            template_picks.append({
                "name": p["name"],
                "points": p["points"] * p["multiplier"],
                "position_type": p.get("position_type", "Unknown")
            })
        else:
            differential_picks.append({
                "name": p["name"],
                "points": p["points"] * p["multiplier"],
                "position_type": p.get("position_type", "Unknown")
            })
    
    template_ratio = round(len(template_picks) / len(starting_picks) * 100, 1) if starting_picks else 0
    template_points = sum(p["points"] for p in template_picks)
    differential_points = sum(p["points"] for p in differential_picks)
    
    return {
        "template_picks": template_picks,
        "differential_picks": differential_picks,
        "template_ratio": template_ratio,
        "template_points": template_points,
        "differential_points": differential_points,
        "strategy_type": (
            "Template Heavy" if template_ratio >= 70 else
            "Differential Heavy" if template_ratio <= 30 else
            "Balanced Approach"
        ),
        "verdict": f"{len(template_picks)} template, {len(differential_picks)} differentials"
    }

def analyze_captain_decision(picks):
    """Enhanced captain analysis with alternatives"""
    captain = next((p for p in picks if p["is_captain"]), None)
    starting_players = [p for p in picks if p["is_starting"]]
    sorted_by_points = sorted(starting_players, key=lambda x: x["points"], reverse=True)
    
    # Get top 5 alternatives
    alternatives = []
    for i, player in enumerate(sorted_by_points[:5]):
        alternatives.append({
            "rank": i + 1,
            "name": player["name"],
            "points": player["points"],
            "potential_captain_points": player["points"] * 2,
            "position_type": player.get("position_type", "Unknown"),
            "is_current_captain": player.get("is_captain", False)
        })
    
    captain_points = captain["points"] * captain["multiplier"] if captain else 0
    best_alternative = alternatives[0] if alternatives else None
    
    return {
        "captain_name": captain["name"] if captain else None,
        "captain_points": captain_points,
        "captain_base_points": captain["points"] if captain else 0,
        "best_alternatives": [alt for alt in alternatives if not alt["is_current_captain"]][:3],
        "decision_quality": {
            "optimal": best_alternative and captain and captain["points"] == best_alternative["points"],
            "points_lost": (best_alternative["points"] - captain["points"]) if (captain and best_alternative) else 0,
            "verdict": "Optimal choice" if (best_alternative and captain and captain["points"] == best_alternative["points"]) 
                      else f"Could have gained {(best_alternative['points'] - captain['points']) if (captain and best_alternative) else 0} more points"
        }
    }

def analyze_fixture_performance(picks):
    """Analyze how players performed relative to fixture difficulty"""    
    performance_flags = []
    
    for p in picks:
        if p["is_starting"] and p.get("fixture_difficulty") is not None:
            difficulty = p["fixture_difficulty"]
            points = p["points"]
            
            # Flag underperformers in easy fixtures
            if difficulty <= 2 and points <= 2:
                performance_flags.append({
                    "name": p["name"],
                    "points": points,
                    "difficulty": difficulty,
                    "type": "underperform",
                    "insight": f"Only {points} pts vs easy opponent (FDR: {difficulty})"
                })
            # Flag overperformers in hard fixtures
            elif difficulty >= 4 and points >= 6:
                performance_flags.append({
                    "name": p["name"],
                    "points": points,
                    "difficulty": difficulty,
                    "type": "overperform",
                    "insight": f"Excellent {points} pts vs tough opponent (FDR: {difficulty})"
                })
    
    return {
        "total_flagged": len(performance_flags),
        "overperformers": [f for f in performance_flags if f["type"] == "overperform"],
        "underperformers": [f for f in performance_flags if f["type"] == "underperform"],
        "summary": f"{len([f for f in performance_flags if f['type'] == 'overperform'])} overperformed, {len([f for f in performance_flags if f['type'] == 'underperform'])} underperformed"
    }

def analyze_bench_strategy(picks):
    """Analyze bench composition and bench boost potential"""
    bench_players = [p for p in picks if p["is_bench"]]
    bench_total = sum(p["points"] for p in bench_players)
    
    # Sort bench by points
    bench_sorted = sorted(bench_players, key=lambda x: x["points"], reverse=True)
    
    # Analyze bench composition
    bench_positions = Counter(p.get("position_type", "Unknown") for p in bench_players)
    
    return {
        "total_bench_points": bench_total,
        "bench_players": len(bench_players),
        "top_bench_performer": {
            "name": bench_sorted[0]["name"],
            "points": bench_sorted[0]["points"],
            "position": bench_sorted[0].get("position_type", "Unknown")
        } if bench_sorted else None,
        "bench_composition": dict(bench_positions),
        "bench_boost_value": bench_total,
        "bench_boost_worthy": bench_total >= 15,  # Threshold for good bench boost
        "wasted_points": bench_total  # Points not contributing to score
    }

def generate_enhanced_llm_prompt(analysis_data):
    """Generate comprehensive prompt for GPT analysis"""
    meta = analysis_data["meta"]
    
    return f"""
Analyze this FPL manager's comprehensive GW{meta['gameweek']} performance:

TEAM OVERVIEW:
- Total Points: {analysis_data.get('total_points', 0)}
- Formation: {analysis_data.get('formation_analysis', {}).get('formation', 'Unknown')} ({analysis_data.get('formation_analysis', {}).get('formation_type', '')})
- Team Value: £{analysis_data.get('value_efficiency', {}).get('total_starting_value', 0)}m
- Average PPM: {analysis_data.get('value_efficiency', {}).get('team_average_ppm', 0)}

CAPTAIN ANALYSIS:
- Captain: {analysis_data.get('captain_advice', {}).get('captain_name', 'Unknown')} ({analysis_data.get('captain_advice', {}).get('captain_points', 0)} pts)
- Decision Quality: {analysis_data.get('captain_advice', {}).get('decision_quality', {}).get('verdict', 'Unknown')}

STRATEGY BREAKDOWN:
- Template Ratio: {analysis_data.get('template_analysis', {}).get('template_ratio', 0)}%
- Strategy: {analysis_data.get('template_analysis', {}).get('strategy_type', 'Unknown')}
- Template Points: {analysis_data.get('template_analysis', {}).get('template_points', 0)}
- Differential Points: {analysis_data.get('template_analysis', {}).get('differential_points', 0)}

BENCH ANALYSIS:
- Bench Points: {analysis_data.get('bench_summary', {}).get('total_bench_points', 0)}
- Bench Boost Value: {analysis_data.get('bench_summary', {}).get('bench_boost_value', 0)} pts
- Top Bench: {analysis_data.get('bench_summary', {}).get('top_bench_performer', {}).get('name', 'None')}

VALUE EFFICIENCY:
- Best Value: {analysis_data.get('value_efficiency', {}).get('best_value', {}).get('name', 'Unknown')} ({analysis_data.get('value_efficiency', {}).get('best_value', {}).get('points_per_million', 0)} pts/£m)
- Worst Value: {analysis_data.get('value_efficiency', {}).get('worst_value', {}).get('name', 'Unknown')} ({analysis_data.get('value_efficiency', {}).get('worst_value', {}).get('points_per_million', 0)} pts/£m)

TRANSFERS:
- Transfers Made: {analysis_data.get('transfer_justification', {}).get('total_transfers', 0)}
- Transfer Cost: £{analysis_data.get('transfer_justification', {}).get('total_cost', 0)}m
- Net Points Impact: {analysis_data.get('transfer_justification', {}).get('net_points_gained', 0)}

FIXTURE PERFORMANCE:
{analysis_data.get('expected_points_delta', {}).get('summary', 'No significant fixture-based performance noted')}

ANALYSIS FOCUS:
1. Overall gameweek performance assessment
2. Captain choice effectiveness and alternatives
3. Formation and strategy evaluation  
4. Value efficiency and transfer impact
5. Template vs differential balance
6. Bench composition and chip opportunities
7. Key recommendations for next gameweek

Provide a detailed, actionable analysis with specific strategic insights and recommendations.
"""

@router.get("/tools/get_manager_gameweek_analysis/{tid}/{gw}")
async def get_manager_gameweek_analysis(tid: int, gw: int):
    """Enhanced comprehensive gameweek analysis using enriched summary data"""
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get enriched summary data (includes fixture difficulty and position types)
        summary = await safe_api_call(client, f"{BASE_URL}/tools/get_manager_gameweek_summary/{tid}/{gw}", {})

    # Validate required data
    if not summary or "picks" not in summary:
        raise HTTPException(status_code=500, detail="Failed to retrieve gameweek summary")
    
    # Extract data quality information
    fixtures_available = summary.get("data_quality", {}).get("fixtures_available", False)

    # Get cached bootstrap data for value efficiency calculations
    try:
        bootstrap = await get_cached_bootstrap()
    except Exception as e:
        logger.error(f"Failed to get bootstrap data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve player data")

    # Use enriched picks from summary (already includes fixture_difficulty and position_type)
    enriched_picks = summary.get("picks", [])

    # ✅ Enhanced Captain Analysis
    captain_advice = analyze_captain_decision(enriched_picks)

    # ✅ Transfer Analysis
    transfers = summary.get("transfers", [])
    transfer_justification = {
        "total_transfers": len(transfers),
        "total_cost": summary.get("total_transfer_cost", 0),
        "transfers": transfers,
        "net_points_gained": 0,  # Could be enhanced with transfer impact calculation
        "verdict": "No transfers made" if not transfers else f"{len(transfers)} transfers analyzed"
    }

    # ✅ Template vs Differential Analysis
    crowd_trends = summary.get("crowd_trends", {})
    template_analysis = analyze_template_vs_differential(enriched_picks, crowd_trends)

    # ✅ Fixture Performance Analysis
    expected_points_delta = analyze_fixture_performance(enriched_picks)

    # ✅ Enhanced Bench Analysis
    bench_summary = analyze_bench_strategy(enriched_picks)

    # ✅ Total Points Calculation
    total_points = summary.get("total_points", 0)

    # ✅ Enhanced Position Summary
    pos_points = Counter()
    pos_players = Counter()
    
    for p in enriched_picks:
        if p["is_starting"]:
            pos_type = p["position_type"]
            pos_points[pos_type] += p["points"] * p["multiplier"]
            pos_players[pos_type] += 1
    
    position_summary = {
        "breakdown": dict(pos_points),
        "player_count": dict(pos_players),
        "average_by_position": {pos: round(points / pos_players[pos], 1) 
                               for pos, points in pos_points.items() if pos_players[pos] > 0},
        "top_contributor": max(pos_points, key=pos_points.get, default=None),
        "weakest_position": min(pos_points, key=pos_points.get, default=None) if pos_points else None
    }

    # ✅ Formation Analysis
    formation_analysis = analyze_formation(enriched_picks)

    # ✅ Value Efficiency Analysis
    value_efficiency = calculate_value_efficiency(enriched_picks, bootstrap)

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000

    # Prepare comprehensive response
    analysis_result = {
        "meta": {
            "analysis_type": "enhanced_comprehensive_gameweek_analysis",
            "manager_id": tid,
            "gameweek": gw,
            "generated_at": datetime.utcnow().isoformat(),
            "data_sources": ["bootstrap", "manager_gameweek_summary"],
            "processing_time_ms": round(processing_time, 2),
            "data_quality": {
                "summary_available": bool(summary),
                "fixtures_available": fixtures_available,
                "bootstrap_available": bool(bootstrap),
                "crowd_trends_available": bool(summary.get("crowd_trends"))
            }
        },
        "total_points": total_points,
        "picks": enriched_picks,
        "captain_advice": captain_advice,
        "transfer_justification": transfer_justification,
        "template_analysis": template_analysis,
        "expected_points_delta": expected_points_delta,
        "bench_summary": bench_summary,
        "position_summary": position_summary,
        "formation_analysis": formation_analysis,
        "value_efficiency": value_efficiency
    }
    
    # Generate enhanced LLM prompt
    analysis_result["llm_prompt_suggestion"] = generate_enhanced_llm_prompt(analysis_result)
    
    return analysis_result