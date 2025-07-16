

from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
import os

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import httpx

# Domain Models
# =============

class FixtureRunQuality(str, Enum):
    """Fixture run quality assessment"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    DIFFICULT = "Difficult"

class ChipTiming(str, Enum):
    """Chip timing recommendations"""
    USE_NOW = "This gameweek"
    USE_SOON = "Consider this gameweek or next"
    WAIT_FOR_OPPORTUNITY = "Wait for better opportunity"
    WAIT_FOR_DOUBLE_GW = "Wait for double gameweek"
    BUILD_FIRST = "Build stronger bench first"
    HOLD_FOR_BLANKS = "Save for blank/double gameweeks"

class TransferPriority(str, Enum):
    """Transfer priority levels"""
    URGENT = "Urgent"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MONITOR = "Monitor"

@dataclass
class FixtureInfo:
    """Individual fixture information"""
    gameweek: int
    opponent: str
    home: bool
    difficulty: int
    
    @property
    def difficulty_text(self) -> str:
        """Human-readable difficulty"""
        ratings = {1: "Very Easy", 2: "Easy", 3: "Average", 4: "Hard", 5: "Very Hard"}
        return ratings.get(self.difficulty, "Unknown")
    
    @property
    def venue_text(self) -> str:
        """Home/Away indicator"""
        return "H" if self.home else "A"

@dataclass
class TeamFixtureAnalysis:
    """Complete fixture analysis for a team"""
    team_id: int
    fixtures: List[FixtureInfo]
    avg_difficulty: float
    easy_fixtures: int
    hard_fixtures: int
    fixture_run_quality: FixtureRunQuality
    
    @property
    def next_fixture(self) -> Optional[FixtureInfo]:
        """Get next fixture"""
        return self.fixtures[0] if self.fixtures else None
    
    @property
    def difficulty_sequence(self) -> List[int]:
        """Get sequence of difficulties"""
        return [f.difficulty for f in self.fixtures]

@dataclass
class PlayerTransferAnalysis:
    """Player transfer out analysis"""
    player_id: int
    player_name: str
    position_type: str
    team_id: int
    avg_difficulty: float
    hard_fixtures_count: int
    fixture_run_quality: FixtureRunQuality
    upcoming_fixtures: List[int]
    priority: TransferPriority
    
    @property
    def transfer_urgency_score(self) -> float:
        """Calculate urgency score for transfers"""
        base_score = self.avg_difficulty * 2
        hard_fixture_penalty = self.hard_fixtures_count * 1.5
        return base_score + hard_fixture_penalty

@dataclass
class TransferTarget:
    """Transfer target with fixture analysis"""
    player_id: int
    name: str
    position: int
    team_id: int
    price: float
    avg_difficulty: float
    easy_fixtures: int
    fixture_run_quality: FixtureRunQuality
    value_score: float
    
    @property
    def target_priority_score(self) -> float:
        """Calculate target desirability score"""
        fixture_score = (5 - self.avg_difficulty) * 2
        easy_fixture_bonus = self.easy_fixtures * 0.5
        return fixture_score + easy_fixture_bonus

@dataclass
class CaptainCandidate:
    """Captain recommendation candidate"""
    player_id: int
    name: str
    position_type: str
    team_id: Optional[int]
    captain_score: float
    next_fixture_difficulty: int
    fixture_run_quality: str
    is_template: bool
    reasoning: str
    form_score: float
    fixture_bonus: float
    consistency_bonus: float
    template_bonus: float

@dataclass
class CaptainRotationPlan:
    """Captain rotation plan for multiple gameweeks"""
    gameweek: int
    recommended_captain: str
    opponent: str
    difficulty: int
    reasoning: str

@dataclass
class ChipRecommendation:
    """Chip usage recommendation"""
    chip_name: str
    recommended: bool
    timing: ChipTiming
    reason: str
    best_candidate: Optional[Dict[str, Any]] = None
    value_assessment: Optional[str] = None

@dataclass
class PlannerSummary:
    """Planning summary and insights"""
    gameweek: int
    gameweeks_analyzed: List[int]
    gameweeks_ahead: int
    free_transfers: int
    priority_actions: List[str]
    key_insights: List[str]
    fixture_run_summary: Dict[str, Any]
    strategic_recommendation: str

# Infrastructure Services
# =======================

class ConfigurationService:
    """Railway-optimized configuration service"""
    
    def __init__(self):
        self.base_url = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
        self.api_timeout = float(os.getenv("PLANNER_API_TIMEOUT", "45.0"))  # Longer for complex planning
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
        self.connection_pool_size = int(os.getenv("CONNECTION_POOL_SIZE", "12"))
        
        # Railway-specific settings
        self.port = int(os.getenv("PORT", "8000"))
        self.environment = os.getenv("RAILWAY_ENVIRONMENT", "development")
        self.is_production = self.environment == "production"
        
        # Planner-specific settings
        self.max_gameweeks_ahead = int(os.getenv("MAX_GAMEWEEKS_AHEAD", "5"))
        self.default_gameweeks_ahead = int(os.getenv("DEFAULT_GAMEWEEKS_AHEAD", "3"))

class HttpClientService:
    """Railway-optimized HTTP client for planner service"""
    
    def __init__(self, config: ConfigurationService):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with Railway optimizations"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.config.api_timeout,
                    write=10.0,
                    pool=8.0
                ),
                limits=httpx.Limits(
                    max_connections=self.config.connection_pool_size,
                    max_keepalive_connections=6
                ),
                headers={
                    "User-Agent": "FPL-Planner-Service/1.0",
                    "Accept": "application/json",
                    "Connection": "keep-alive"
                },
                follow_redirects=True
            )
        return self._client
    
    async def safe_api_call(
        self,
        url: str,
        fallback_data: Any = None,
        timeout: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        context: str = "api_call"
    ) -> Any:
        """Enhanced API call with Railway-specific optimizations"""
        timeout = timeout or self.config.api_timeout
        retries = self.config.max_retries
        
        async with self._semaphore:
            for attempt in range(retries + 1):
                try:
                    client = await self.get_client()
                    
                    response = await client.get(url, timeout=timeout, params=params)
                    
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except Exception as json_error:
                            logger.error(f"JSON parsing error ({context}): {json_error}")
                            return fallback_data
                    
                    elif response.status_code == 404:
                        logger.warning(f"Resource not found ({context}): {url}")
                        return fallback_data
                    
                    elif response.status_code == 429:  # Rate limited
                        if attempt < retries:
                            wait_time = min(2 ** attempt * 3, 45)  # Longer waits for planning
                            logger.warning(f"Rate limited ({context}), waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded ({context}): {url}")
                            return fallback_data
                    
                    elif response.status_code >= 500:
                        if attempt < retries:
                            wait_time = min(2 ** attempt * 2, 20)
                            logger.warning(f"Server error ({context}), retry {attempt + 1} in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Server error after retries ({context}): {url}")
                            return fallback_data
                    
                    else:
                        logger.warning(f"Unexpected status ({context}): {response.status_code}")
                        return fallback_data
                
                except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    if attempt < retries:
                        wait_time = min(2 ** attempt * 2, 15)
                        logger.warning(f"Timeout ({context}) attempt {attempt + 1}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Final timeout error ({context}): {str(e)}")
                        return fallback_data
                
                except (httpx.ConnectError, httpx.NetworkError) as e:
                    if attempt < retries:
                        wait_time = min(2 ** attempt * 2.5, 18)
                        logger.warning(f"Network error ({context}) attempt {attempt + 1}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Final network error ({context}): {str(e)}")
                        return fallback_data
                
                except Exception as e:
                    logger.error(f"Unexpected error ({context}): {str(e)}")
                    return fallback_data
        
        return fallback_data
    
    async def batch_api_calls(
        self,
        calls: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]]
    ) -> List[Any]:
        """Execute batch API calls with proper error handling"""
        tasks = [
            self.safe_api_call(url, fallback, params=params, context=context)
            for url, context, fallback, params in calls
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch task {i} ({calls[i][1]}) failed: {str(result)}")
                    processed_results.append(calls[i][2])  # fallback data
                else:
                    processed_results.append(result)
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Batch API calls failed: {str(e)}")
            return [call[2] for call in calls]
    
    async def cleanup(self):
        """Cleanup HTTP client resources"""
        if self._client:
            await self._client.aclose()
            self._client = None

# Domain Services
# ===============

class FixtureAnalysisService:
    """Service for analyzing fixture difficulty and runs"""
    
    def analyze_upcoming_fixtures(
        self,
        player_team_map: Dict[int, int],
        upcoming_fixtures: List[Dict[str, Any]],
        gameweeks_ahead: int = 5
    ) -> Dict[int, TeamFixtureAnalysis]:
        """Analyze fixture difficulty for upcoming gameweeks"""
        fixture_data = defaultdict(list)
        
        # Process fixtures
        for fixture in upcoming_fixtures:
            gw = fixture.get("event")
            if not gw:
                continue
            
            team_h = fixture.get("team_h", {})
            team_a = fixture.get("team_a", {})
            
            # Extract team IDs
            team_h_id = team_h.get("id") if isinstance(team_h, dict) else team_h
            team_a_id = team_a.get("id") if isinstance(team_a, dict) else team_a
            
            # Process home team
            if team_h_id:
                difficulty = fixture.get("team_h_difficulty", 3)
                opponent_name = team_a.get("name", f"Team {team_a_id}") if isinstance(team_a, dict) else f"Team {team_a_id}"
                
                fixture_data[team_h_id].append(FixtureInfo(
                    gameweek=gw,
                    opponent=opponent_name,
                    home=True,
                    difficulty=difficulty
                ))
            
            # Process away team
            if team_a_id:
                difficulty = fixture.get("team_a_difficulty", 3)
                opponent_name = team_h.get("name", f"Team {team_h_id}") if isinstance(team_h, dict) else f"Team {team_h_id}"
                
                fixture_data[team_a_id].append(FixtureInfo(
                    gameweek=gw,
                    opponent=opponent_name,
                    home=False,
                    difficulty=difficulty
                ))
        
        # Analyze each team's fixture run
        team_analyses = {}
        for team_id, fixtures in fixture_data.items():
            # Sort by gameweek
            fixtures.sort(key=lambda x: x.gameweek)
            
            # Calculate stats
            difficulties = [f.difficulty for f in fixtures]
            if difficulties:
                avg_difficulty = sum(difficulties) / len(difficulties)
                easy_fixtures = sum(1 for d in difficulties if d <= 2)
                hard_fixtures = sum(1 for d in difficulties if d >= 4)
                
                # Determine quality
                if avg_difficulty <= 2.5:
                    quality = FixtureRunQuality.EXCELLENT
                elif avg_difficulty <= 3.0:
                    quality = FixtureRunQuality.GOOD
                elif avg_difficulty <= 3.5:
                    quality = FixtureRunQuality.AVERAGE
                else:
                    quality = FixtureRunQuality.DIFFICULT
                
                team_analyses[team_id] = TeamFixtureAnalysis(
                    team_id=team_id,
                    fixtures=fixtures,
                    avg_difficulty=round(avg_difficulty, 1),
                    easy_fixtures=easy_fixtures,
                    hard_fixtures=hard_fixtures,
                    fixture_run_quality=quality
                )
        
        return team_analyses

class TransferAnalysisService:
    """Service for analyzing transfer opportunities"""
    
    def recommend_transfers(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis],
        bootstrap_data: Dict[str, Any],
        free_transfers: int = 1
    ) -> Dict[str, Any]:
        """Enhanced transfer recommendations based on fixture runs"""
        
        element_prices = {el["id"]: el["now_cost"] / 10 for el in bootstrap_data["elements"]}
        element_teams = {el["id"]: el["team"] for el in bootstrap_data["elements"]}
        element_names = {el["id"]: el["web_name"] for el in bootstrap_data["elements"]}
        
        # Analyze current players for transfer out
        players_out = []
        players_to_watch = []
        
        for pick in current_picks:
            if pick.get("is_starting", False):
                player_id = pick["id"]
                team_id = element_teams.get(player_id)
                
                if team_id in fixture_analysis:
                    team_fixtures = fixture_analysis[team_id]
                    
                    # Determine transfer priority
                    if team_fixtures.avg_difficulty >= 3.8 or team_fixtures.hard_fixtures >= 3:
                        priority = TransferPriority.URGENT if team_fixtures.avg_difficulty >= 4.0 else TransferPriority.HIGH
                    elif team_fixtures.avg_difficulty >= 3.3 or team_fixtures.hard_fixtures >= 2:
                        priority = TransferPriority.MEDIUM
                    else:
                        priority = TransferPriority.LOW
                    
                    player_analysis = PlayerTransferAnalysis(
                        player_id=player_id,
                        player_name=pick["name"],
                        position_type=pick.get("position_type", "Unknown"),
                        team_id=team_id,
                        avg_difficulty=team_fixtures.avg_difficulty,
                        hard_fixtures_count=team_fixtures.hard_fixtures,
                        fixture_run_quality=team_fixtures.fixture_run_quality,
                        upcoming_fixtures=team_fixtures.difficulty_sequence,
                        priority=priority
                    )
                    
                    if priority in [TransferPriority.URGENT, TransferPriority.HIGH]:
                        players_out.append(player_analysis)
                    elif priority == TransferPriority.MEDIUM:
                        players_to_watch.append(player_analysis)
        
        # Find transfer targets with good fixture runs
        transfer_targets = []
        for element in bootstrap_data["elements"]:
            team_id = element["team"]
            if team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                if (team_fixtures.avg_difficulty <= 2.8 and 
                    team_fixtures.easy_fixtures >= 3 and 
                    team_fixtures.fixture_run_quality in [FixtureRunQuality.EXCELLENT, FixtureRunQuality.GOOD]):
                    
                    target = TransferTarget(
                        player_id=element["id"],
                        name=element["web_name"],
                        position=element["element_type"],
                        team_id=team_id,
                        price=element["now_cost"] / 10,
                        avg_difficulty=team_fixtures.avg_difficulty,
                        easy_fixtures=team_fixtures.easy_fixtures,
                        fixture_run_quality=team_fixtures.fixture_run_quality,
                        value_score=team_fixtures.easy_fixtures / (element["now_cost"] / 10)
                    )
                    transfer_targets.append(target)
        
        # Sort by priority and value
        players_out.sort(key=lambda x: x.transfer_urgency_score, reverse=True)
        transfer_targets.sort(key=lambda x: x.target_priority_score, reverse=True)
        
        return {
            "transfer_out_candidates": players_out[:5],
            "players_to_monitor": players_to_watch[:3],
            "transfer_targets": transfer_targets[:10],
            "free_transfers": free_transfers,
            "recommendation": self._get_transfer_recommendation(players_out, transfer_targets, free_transfers)
        }
    
    def _get_transfer_recommendation(
        self,
        players_out: List[PlayerTransferAnalysis],
        transfer_targets: List[TransferTarget],
        free_transfers: int
    ) -> str:
        """Generate transfer recommendation text"""
        if not players_out:
            return "Hold transfers - no urgent moves needed based on fixtures"
        
        urgent_transfers = [p for p in players_out if p.priority == TransferPriority.URGENT]
        
        if urgent_transfers and free_transfers >= 1:
            return f"Priority: Transfer out {urgent_transfers[0].player_name} - {urgent_transfers[0].fixture_run_quality.value} fixture run"
        elif players_out:
            return f"Consider transferring {players_out[0].player_name} - difficult upcoming fixtures"
        
        return "Monitor fixture developments"

class CaptainAnalysisService:
    """Service for captain recommendation and rotation planning"""
    
    def recommend_captain(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis],
        crowd_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced captain recommendation based on fixture runs"""
        
        # Get template players for safer options
        template_players = set()
        most_selected = crowd_trends.get("most_selected_by_position", {})
        for position, players in most_selected.items():
            for player_data in players[:2]:
                if float(player_data.get("selected_by_percent", 0)) > 20:
                    template_players.add(player_data["player"])
        
        captain_candidates = []
        
        for pick in current_picks:
            if pick.get("is_starting", False):
                # Base scoring
                base_score = pick.get("points", 0)
                fixture_bonus = 0
                consistency_bonus = 0
                next_fixture_difficulty = 3
                
                # Fixture analysis
                team_id = pick.get("team_id")
                if team_id and team_id in fixture_analysis:
                    team_fixtures = fixture_analysis[team_id]
                    next_fixture = team_fixtures.next_fixture
                    
                    if next_fixture:
                        next_fixture_difficulty = next_fixture.difficulty
                        fixture_bonus = (6 - next_fixture_difficulty) * 3
                    
                    # Fixture run bonus
                    if team_fixtures.easy_fixtures >= 2:
                        consistency_bonus = 5
                    elif team_fixtures.avg_difficulty <= 2.5:
                        consistency_bonus = 3
                
                # Template bonus
                template_bonus = 5 if pick["name"] in template_players else 0
                
                captain_score = base_score + fixture_bonus + consistency_bonus + template_bonus
                
                candidate = CaptainCandidate(
                    player_id=pick["id"],
                    name=pick["name"],
                    position_type=pick.get("position_type", "Unknown"),
                    team_id=team_id,
                    captain_score=captain_score,
                    next_fixture_difficulty=next_fixture_difficulty,
                    fixture_run_quality=fixture_analysis[team_id].fixture_run_quality.value if team_id in fixture_analysis else "Unknown",
                    is_template=pick["name"] in template_players,
                    reasoning=f"Score: {captain_score} (Form: {base_score}, Next fixture: +{fixture_bonus}, Run: +{consistency_bonus}, Template: +{template_bonus})",
                    form_score=base_score,
                    fixture_bonus=fixture_bonus,
                    consistency_bonus=consistency_bonus,
                    template_bonus=template_bonus
                )
                
                captain_candidates.append(candidate)
        
        # Sort by captain score
        captain_candidates.sort(key=lambda x: x.captain_score, reverse=True)
        
        return {
            "top_captain_pick": captain_candidates[0] if captain_candidates else None,
            "alternatives": captain_candidates[1:4],
            "captain_rotation_plan": self._get_captain_rotation_plan(captain_candidates, fixture_analysis),
            "recommendation": f"Captain {captain_candidates[0].name} - {captain_candidates[0].reasoning}" if captain_candidates else "No clear captain recommendation"
        }
    
    def _get_captain_rotation_plan(
        self,
        captain_candidates: List[CaptainCandidate],
        fixture_analysis: Dict[int, TeamFixtureAnalysis]
    ) -> List[CaptainRotationPlan]:
        """Suggest captain rotation over next 3 gameweeks"""
        rotation_plan = []
        top_captains = captain_candidates[:3]
        
        for gw_offset in range(3):
            best_captain_for_gw = None
            best_score = 0
            
            for candidate in top_captains:
                if candidate.team_id and candidate.team_id in fixture_analysis:
                    team_fixtures = fixture_analysis[candidate.team_id].fixtures
                    if gw_offset < len(team_fixtures):
                        fixture = team_fixtures[gw_offset]
                        gw_score = candidate.captain_score + (6 - fixture.difficulty) * 2
                        
                        if gw_score > best_score:
                            best_score = gw_score
                            best_captain_for_gw = CaptainRotationPlan(
                                gameweek=fixture.gameweek,
                                recommended_captain=candidate.name,
                                opponent=fixture.opponent,
                                difficulty=fixture.difficulty,
                                reasoning=f"Excellent fixture vs {fixture.opponent} ({fixture.venue_text})"
                            )
            
            if best_captain_for_gw:
                rotation_plan.append(best_captain_for_gw)
        
        return rotation_plan

class ChipAnalysisService:
    """Service for analyzing chip timing opportunities"""
    
    def analyze_chip_timing(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis],
        bench_points: int = 0
    ) -> Dict[str, ChipRecommendation]:
        """Enhanced chip timing analysis"""
        
        chip_recommendations = {}
        
        # Bench Boost analysis
        chip_recommendations["bench_boost"] = self._analyze_bench_boost(bench_points)
        
        # Triple Captain analysis
        chip_recommendations["triple_captain"] = self._analyze_triple_captain(
            current_picks, fixture_analysis
        )
        
        # Free Hit analysis
        chip_recommendations["free_hit"] = self._analyze_free_hit(
            current_picks, fixture_analysis
        )
        
        return chip_recommendations
    
    def _analyze_bench_boost(self, bench_points: int) -> ChipRecommendation:
        """Analyze bench boost timing"""
        if bench_points >= 15:
            return ChipRecommendation(
                chip_name="bench_boost",
                recommended=True,
                timing=ChipTiming.USE_NOW,
                reason=f"Strong bench with {bench_points} points - excellent BB opportunity",
                value_assessment="High value"
            )
        elif bench_points >= 10:
            return ChipRecommendation(
                chip_name="bench_boost",
                recommended=False,
                timing=ChipTiming.WAIT_FOR_OPPORTUNITY,
                reason=f"Decent bench ({bench_points} pts) but wait for stronger opportunity",
                value_assessment="Medium value"
            )
        else:
            return ChipRecommendation(
                chip_name="bench_boost",
                recommended=False,
                timing=ChipTiming.BUILD_FIRST,
                reason=f"Weak bench ({bench_points} pts) - improve before using BB",
                value_assessment="Low value"
            )
    
    def _analyze_triple_captain(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis]
    ) -> ChipRecommendation:
        """Analyze triple captain timing"""
        starting_players = [p for p in current_picks if p.get("is_starting", False)]
        
        if not starting_players:
            return ChipRecommendation(
                chip_name="triple_captain",
                recommended=False,
                timing=ChipTiming.WAIT_FOR_OPPORTUNITY,
                reason="No starting players data available"
            )
        
        best_tc_candidate = None
        best_tc_score = 0
        
        for player in starting_players:
            team_id = player.get("team_id")
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                
                form_score = player.get("points", 0)
                fixture_score = (4 - team_fixtures.avg_difficulty) * 5
                tc_score = form_score + fixture_score
                
                if tc_score > best_tc_score:
                    best_tc_score = tc_score
                    best_tc_candidate = {
                        "name": player["name"],
                        "tc_score": tc_score,
                        "avg_difficulty": team_fixtures.avg_difficulty,
                        "fixture_run": team_fixtures.fixture_run_quality.value
                    }
        
        if best_tc_candidate and best_tc_candidate["avg_difficulty"] <= 2.5:
            return ChipRecommendation(
                chip_name="triple_captain",
                recommended=True,
                timing=ChipTiming.USE_SOON,
                reason=f"{best_tc_candidate['name']} has excellent fixture run ({best_tc_candidate['fixture_run']})",
                best_candidate=best_tc_candidate
            )
        else:
            return ChipRecommendation(
                chip_name="triple_captain",
                recommended=False,
                timing=ChipTiming.WAIT_FOR_OPPORTUNITY,
                reason="Wait for better fixture combination for premium players",
                best_candidate=best_tc_candidate
            )
    
    def _analyze_free_hit(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis]
    ) -> ChipRecommendation:
        """Analyze free hit timing"""
        starting_players = [p for p in current_picks if p.get("is_starting", False)]
        
        players_with_bad_fixtures = 0
        for player in starting_players:
            team_id = player.get("team_id")
            if team_id and team_id in fixture_analysis:
                if fixture_analysis[team_id].avg_difficulty >= 4:
                    players_with_bad_fixtures += 1
        
        if players_with_bad_fixtures >= 6:
            return ChipRecommendation(
                chip_name="free_hit",
                recommended=True,
                timing=ChipTiming.USE_NOW,
                reason=f"{players_with_bad_fixtures} players have very difficult fixtures"
            )
        else:
            return ChipRecommendation(
                chip_name="free_hit",
                recommended=False,
                timing=ChipTiming.HOLD_FOR_BLANKS,
                reason="Most players have manageable fixtures"
            )

class FormationAnalysisService:
    """Service for formation optimization based on fixtures"""
    
    def optimize_formation(
        self,
        current_picks: List[Dict[str, Any]],
        fixture_analysis: Dict[int, TeamFixtureAnalysis]
    ) -> Dict[str, Any]:
        """Suggest optimal formation based on upcoming fixtures"""
        starting_players = [p for p in current_picks if p.get("is_starting", False)]
        
        # Count players by position with good fixtures
        good_fixtures_by_position = Counter()
        total_by_position = Counter()
        
        for player in starting_players:
            pos_type = player.get("position_type", "Unknown")
            team_id = player.get("team_id")
            
            total_by_position[pos_type] += 1
            
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                if team_fixtures.next_fixture and team_fixtures.next_fixture.difficulty <= 3:
                    good_fixtures_by_position[pos_type] += 1
        
        # Calculate formation string (excluding goalkeeper)
        def_count = total_by_position.get("Defender", 0)
        mid_count = total_by_position.get("Midfielder", 0)
        fwd_count = total_by_position.get("Forward", 0)
        
        current_formation = f"{def_count}-{mid_count}-{fwd_count}"
        
        return {
            "current_formation": current_formation,
            "formation_strength": {
                "defenders_good_fixtures": good_fixtures_by_position.get("Defender", 0),
                "midfielders_good_fixtures": good_fixtures_by_position.get("Midfielder", 0),
                "forwards_good_fixtures": good_fixtures_by_position.get("Forward", 0),
                "total_good_fixtures": sum(good_fixtures_by_position.values())
            },
            "recommendation": "Current formation looks good" if sum(good_fixtures_by_position.values()) >= 8 else "Consider formation change based on fixtures",
            "fixture_strength_percentage": round((sum(good_fixtures_by_position.values()) / len(starting_players)) * 100, 1) if starting_players else 0
        }

class PlannerSummaryService:
    """Service for generating comprehensive planning summaries"""
    
    def get_fixture_run_summary(
        self,
        fixture_analysis: Dict[int, TeamFixtureAnalysis],
        current_picks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of fixture runs for current squad"""
        summary = {
            "excellent_fixtures": [],
            "good_fixtures": [],
            "difficult_fixtures": [],
            "overall_assessment": "",
            "fixture_quality_distribution": Counter()
        }
        
        starting_players = [p for p in current_picks if p.get("is_starting", False)]
        
        for player in starting_players:
            team_id = player.get("team_id")
            if team_id and team_id in fixture_analysis:
                team_fixtures = fixture_analysis[team_id]
                quality = team_fixtures.fixture_run_quality
                
                summary["fixture_quality_distribution"][quality.value] += 1
                
                player_summary = {
                    "player": player["name"],
                    "quality": quality.value,
                    "avg_difficulty": team_fixtures.avg_difficulty,
                    "easy_fixtures": team_fixtures.easy_fixtures,
                    "hard_fixtures": team_fixtures.hard_fixtures
                }
                
                if quality in [FixtureRunQuality.EXCELLENT]:
                    summary["excellent_fixtures"].append(player_summary)
                elif quality == FixtureRunQuality.GOOD:
                    summary["good_fixtures"].append(player_summary)
                elif quality == FixtureRunQuality.DIFFICULT:
                    summary["difficult_fixtures"].append(player_summary)
        
        # Overall assessment
        excellent_count = len(summary["excellent_fixtures"])
        good_count = len(summary["good_fixtures"])
        difficult_count = len(summary["difficult_fixtures"])
        total_players = len(starting_players)
        
        if excellent_count >= 6:
            summary["overall_assessment"] = "Excellent fixture period - great time for points and chip usage"
        elif excellent_count + good_count >= 8:
            summary["overall_assessment"] = "Good fixture period - expect decent returns"
        elif difficult_count >= 6:
            summary["overall_assessment"] = "Difficult fixture period - consider transfers and defensive strategies"
        else:
            summary["overall_assessment"] = "Mixed fixture period - selective captaincy and tactical planning needed"
        
        return summary
    
    def generate_planning_summary(
        self,
        gameweek: int,
        analyzed_gameweeks: List[int],
        gameweeks_ahead: int,
        free_transfers: int,
        transfer_recommendations: Dict[str, Any],
        captain_recommendation: Dict[str, Any],
        chip_analysis: Dict[str, ChipRecommendation],
        fixture_run_summary: Dict[str, Any]
    ) -> PlannerSummary:
        """Generate comprehensive planning summary"""
        
        priority_actions = []
        key_insights = []
        
        # Add transfer priorities
        if transfer_recommendations.get("transfer_out_candidates"):
            top_transfer = transfer_recommendations["transfer_out_candidates"][0]
            priority_actions.append(
                f"Transfer priority: {top_transfer.player_name} - {top_transfer.fixture_run_quality.value} fixture run"
            )
        
        # Add chip recommendations
        for chip_name, chip_rec in chip_analysis.items():
            if chip_rec.recommended:
                priority_actions.append(f"{chip_name.replace('_', ' ').title()}: {chip_rec.reason}")
        
        # Add captain recommendation
        if captain_recommendation.get("top_captain_pick"):
            captain = captain_recommendation["top_captain_pick"]
            priority_actions.append(f"Captain {captain.name} - {captain.fixture_run_quality} fixtures")
        
        # Generate key insights
        if transfer_recommendations.get("transfer_targets"):
            top_target = transfer_recommendations["transfer_targets"][0]
            key_insights.append(f"Best transfer target: {top_target.name} ({top_target.fixture_run_quality.value} fixtures)")
        
        if captain_recommendation.get("captain_rotation_plan"):
            key_insights.append("Captain rotation opportunities available for optimal fixtures")
        
        # Strategic recommendation based on analysis depth
        if gameweeks_ahead == 1:
            strategic_rec = "Short-term focus: Optimize immediate decisions for next gameweek"
        elif gameweeks_ahead <= 3:
            strategic_rec = "Medium-term planning: Balance immediate needs with upcoming fixture swings"
        else:
            strategic_rec = "Long-term strategy: Position for sustained success and optimal chip timing"
        
        key_insights.append(strategic_rec)
        
        return PlannerSummary(
            gameweek=gameweek,
            gameweeks_analyzed=analyzed_gameweeks,
            gameweeks_ahead=gameweeks_ahead,
            free_transfers=free_transfers,
            priority_actions=priority_actions,
            key_insights=key_insights,
            fixture_run_summary=fixture_run_summary,
            strategic_recommendation=strategic_rec
        )

# Error Handling
# ==============

class PlannerServiceException(Exception):
    """Custom exception for planner service errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

async def planner_exception_handler(request: Request, exc: PlannerServiceException):
    """Global exception handler for planner service"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": str(request.url),
            "service": "gameweek-planner"
        }
    )

# Router Setup
# ============

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize services
config = ConfigurationService()
http_client = HttpClientService(config)
fixture_service = FixtureAnalysisService()
transfer_service = TransferAnalysisService()
captain_service = CaptainAnalysisService()
chip_service = ChipAnalysisService()
formation_service = FormationAnalysisService()
summary_service = PlannerSummaryService()

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for Railway monitoring"""
    return {
        "status": "healthy",
        "service": "gameweek-planner",
        "environment": config.environment,
        "max_gameweeks_ahead": config.max_gameweeks_ahead,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/tools/get_upcoming_gameweek_planner/{tid}/{next_gw}")
async def get_upcoming_gameweek_planner(
    tid: int,
    next_gw: int,
    gameweeks_ahead: int = Query(default=3, ge=1, le=5, description="Number of gameweeks to analyze ahead (1-5)")
) -> Dict[str, Any]:
    """Enhanced gameweek planner with Railway optimizations and configurable lookahead"""
    start_time = time.time()
    
    try:
        # Input validation
        if not (1 <= next_gw <= 38):
            raise PlannerServiceException("Invalid gameweek number", 400)
        if tid <= 0:
            raise PlannerServiceException("Invalid team ID", 400)
        
        gameweeks_ahead = max(1, min(config.max_gameweeks_ahead, gameweeks_ahead))
        
        # Prepare API calls for batch execution
        api_calls = [
            (f"{config.base_url}/tools/get_manager_gameweek_summary/{tid}/{next_gw-1}", "current_summary", {}, None),
            (f"{config.base_url}/tools/get_manager_info/{tid}", "manager_info", {}, None)
        ]
        
        # Add fixture calls for each gameweek
        for gw_offset in range(gameweeks_ahead):
            target_gw = next_gw + gw_offset
            if target_gw <= 38:  # Don't go beyond season end
                api_calls.append((
                    f"{config.base_url}/tools/get_fixtures_by_gw",
                    f"fixtures_gw_{target_gw}",
                    {},
                    {"gw": target_gw}
                ))
        
        # Execute all API calls in batch
        logger.info(f"Starting planner analysis for manager {tid}, GW{next_gw}, {gameweeks_ahead} weeks ahead")
        results = await http_client.batch_api_calls(api_calls)
        
        # Extract results
        current_summary = results[0]
        manager_info = results[1]
        fixture_results = results[2:]
        
        # Validate critical data
        if not current_summary or "picks" not in current_summary:
            raise PlannerServiceException("Could not retrieve current team data", 404)
        
        # Get bootstrap data
        try:
            from utils.bootstrap import get_cached_bootstrap
            bootstrap = await get_cached_bootstrap()
            player_team_map = {el["id"]: el["team"] for el in bootstrap["elements"]}
        except Exception as e:
            logger.error(f"Failed to get bootstrap data: {e}")
            raise PlannerServiceException("Failed to retrieve player data", 500)
        
        # Process fixtures
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
            raise PlannerServiceException(
                f"No fixture data available for gameweeks {next_gw}-{next_gw + gameweeks_ahead - 1}",
                404
            )
        
        # Perform comprehensive analysis
        current_picks = current_summary.get("picks", [])
        
        # Fixture analysis
        fixture_analysis = fixture_service.analyze_upcoming_fixtures(
            player_team_map, all_upcoming_fixtures, gameweeks_ahead
        )
        
        # Get manager transfer info
        free_transfers = 1
        if manager_info and "current_event_transfers" in manager_info:
            transfers_made = manager_info.get("current_event_transfers", 0)
            free_transfers = max(1 - transfers_made, 0)
        
        # Transfer analysis
        transfer_recommendations = transfer_service.recommend_transfers(
            current_picks, fixture_analysis, bootstrap, free_transfers
        )
        
        # Captain analysis
        captain_recommendation = captain_service.recommend_captain(
            current_picks, fixture_analysis, current_summary.get("crowd_trends", {})
        )
        
        # Chip analysis
        bench_points = current_summary.get("bench_points", 0)
        chip_analysis = chip_service.analyze_chip_timing(current_picks, fixture_analysis, bench_points)
        
        # Formation analysis
        formation_optimization = formation_service.optimize_formation(current_picks, fixture_analysis)
        
        # Generate comprehensive summary
        fixture_run_summary = summary_service.get_fixture_run_summary(fixture_analysis, current_picks)
        
        planning_summary = summary_service.generate_planning_summary(
            next_gw, analyzed_gameweeks, gameweeks_ahead, free_transfers,
            transfer_recommendations, captain_recommendation, chip_analysis, fixture_run_summary
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        return {
            "meta": {
                "analysis_type": "railway_optimized_gameweek_planner",
                "manager_id": tid,
                "target_gameweek": next_gw,
                "gameweeks_analyzed": analyzed_gameweeks,
                "gameweeks_ahead": gameweeks_ahead,
                "generated_at": datetime.utcnow().isoformat(),
                "data_sources": ["manager_gameweek_summary", "fixtures", "bootstrap", "manager_info"],
                "processing_time_ms": round(processing_time, 2),
                "environment": config.environment
            },
            "planning_summary": {
                "gameweek": planning_summary.gameweek,
                "gameweeks_analyzed": planning_summary.gameweeks_analyzed,
                "gameweeks_ahead": planning_summary.gameweeks_ahead,
                "free_transfers": planning_summary.free_transfers,
                "priority_actions": planning_summary.priority_actions,
                "key_insights": planning_summary.key_insights,
                "fixture_run_summary": planning_summary.fixture_run_summary,
                "strategic_recommendation": planning_summary.strategic_recommendation
            },
            "transfer_recommendations": {
                "transfer_out_candidates": [
                    {
                        "player_name": candidate.player_name,
                        "position_type": candidate.position_type,
                        "avg_difficulty": candidate.avg_difficulty,
                        "hard_fixtures_count": candidate.hard_fixtures_count,
                        "fixture_run_quality": candidate.fixture_run_quality.value,
                        "upcoming_fixtures": candidate.upcoming_fixtures,
                        "priority": candidate.priority.value
                    }
                    for candidate in transfer_recommendations["transfer_out_candidates"]
                ],
                "players_to_monitor": [
                    {
                        "player_name": candidate.player_name,
                        "position_type": candidate.position_type,
                        "avg_difficulty": candidate.avg_difficulty,
                        "fixture_run_quality": candidate.fixture_run_quality.value,
                        "priority": candidate.priority.value
                    }
                    for candidate in transfer_recommendations["players_to_monitor"]
                ],
                "transfer_targets": [
                    {
                        "name": target.name,
                        "position": target.position,
                        "price": target.price,
                        "avg_difficulty": target.avg_difficulty,
                        "easy_fixtures": target.easy_fixtures,
                        "fixture_run_quality": target.fixture_run_quality.value
                    }
                    for target in transfer_recommendations["transfer_targets"]
                ],
                "free_transfers": transfer_recommendations["free_transfers"],
                "recommendation": transfer_recommendations["recommendation"]
            },
            "captain_recommendation": {
                "top_captain_pick": {
                    "name": captain_recommendation["top_captain_pick"].name,
                    "position_type": captain_recommendation["top_captain_pick"].position_type,
                    "captain_score": captain_recommendation["top_captain_pick"].captain_score,
                    "next_fixture_difficulty": captain_recommendation["top_captain_pick"].next_fixture_difficulty,
                    "fixture_run_quality": captain_recommendation["top_captain_pick"].fixture_run_quality,
                    "is_template": captain_recommendation["top_captain_pick"].is_template,
                    "reasoning": captain_recommendation["top_captain_pick"].reasoning
                } if captain_recommendation.get("top_captain_pick") else None,
                "alternatives": [
                    {
                        "name": alt.name,
                        "captain_score": alt.captain_score,
                        "fixture_run_quality": alt.fixture_run_quality,
                        "reasoning": alt.reasoning
                    }
                    for alt in captain_recommendation.get("alternatives", [])
                ],
                "captain_rotation_plan": [
                    {
                        "gameweek": plan.gameweek,
                        "recommended_captain": plan.recommended_captain,
                        "opponent": plan.opponent,
                        "difficulty": plan.difficulty,
                        "reasoning": plan.reasoning
                    }
                    for plan in captain_recommendation.get("captain_rotation_plan", [])
                ],
                "recommendation": captain_recommendation.get("recommendation", "")
            },
            "chip_analysis": {
                chip_name: {
                    "recommended": chip_rec.recommended,
                    "timing": chip_rec.timing.value,
                    "reason": chip_rec.reason,
                    "best_candidate": chip_rec.best_candidate,
                    "value_assessment": chip_rec.value_assessment
                }
                for chip_name, chip_rec in chip_analysis.items()
            },
            "formation_optimization": formation_optimization,
            "upcoming_fixtures_overview": {
                "total_fixtures_analyzed": len(all_upcoming_fixtures),
                "gameweeks_covered": analyzed_gameweeks,
                "fixtures_available": bool(all_upcoming_fixtures),
                "teams_analyzed": len(fixture_analysis)
            }
        }
        
    except PlannerServiceException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in gameweek planner: {str(e)}")
        raise PlannerServiceException(f"Internal server error: {str(e)}", 500)

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    await http_client.cleanup()
    logger.info("Planner service shutdown complete")

