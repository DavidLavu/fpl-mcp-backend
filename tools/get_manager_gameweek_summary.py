
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

# Domain Models
# =============

class ChipType(str, Enum):
    """Available FPL chip types"""
    WILDCARD = "wildcard"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"
    FREE_HIT = "freehit"

class PositionType(str, Enum):
    """Player position types"""
    GOALKEEPER = "Goalkeeper"
    DEFENDER = "Defender"
    MIDFIELDER = "Midfielder"
    FORWARD = "Forward"
    UNKNOWN = "Unknown"

class CaptainDecisionQuality(str, Enum):
    """Captain decision quality assessment"""
    OPTIMAL = "Optimal"
    SUBOPTIMAL = "Suboptimal"
    POOR = "Poor"
    EXCELLENT = "Excellent"

@dataclass
class PlayerPick:
    """Enhanced player pick with comprehensive data"""
    id: int
    name: str
    multiplier: int
    position: int
    points: int
    is_captain: bool = False
    is_vice_captain: bool = False
    fixture_difficulty: Optional[int] = None
    position_type: str = PositionType.UNKNOWN.value
    team_id: Optional[int] = None
    
    @property
    def is_starting(self) -> bool:
        """Check if player is in starting XI"""
        return self.position <= 11
    
    @property
    def is_bench(self) -> bool:
        """Check if player is on bench"""
        return self.position > 11
    
    @property
    def total_points(self) -> int:
        """Calculate total points including multiplier"""
        return self.points * self.multiplier
    
    @property
    def fixture_difficulty_rating(self) -> str:
        """Get human-readable fixture difficulty"""
        if self.fixture_difficulty is None:
            return "Unknown"
        
        ratings = {1: "Very Easy", 2: "Easy", 3: "Average", 4: "Hard", 5: "Very Hard"}
        return ratings.get(self.fixture_difficulty, "Unknown")

@dataclass
class TransferMove:
    """Transfer move with enhanced data"""
    element_in: str
    element_out: str
    cost: int
    element_in_id: Optional[int] = None
    element_out_id: Optional[int] = None
    
    @property
    def is_free_transfer(self) -> bool:
        """Check if transfer was free"""
        return self.cost == 0

@dataclass
class CaptainEffectiveness:
    """Captain decision analysis"""
    captain_points: int
    captain_base_points: int
    max_possible_points: int
    difference: int
    efficiency_percentage: float
    verdict: CaptainDecisionQuality
    
    @property
    def points_lost(self) -> int:
        """Points lost due to suboptimal captain choice"""
        return max(0, self.difference)
    
    @property
    def was_optimal(self) -> bool:
        """Check if captain choice was optimal"""
        return self.verdict == CaptainDecisionQuality.OPTIMAL

@dataclass
class DataQuality:
    """Data availability assessment"""
    picks_available: bool = False
    live_data_available: bool = False
    history_available: bool = False
    transfers_available: bool = False
    crowd_data_available: bool = False
    fixtures_available: bool = False
    bootstrap_available: bool = False
    
    @property
    def completeness_score(self) -> float:
        """Calculate data completeness as percentage"""
        total_sources = 7
        available_sources = sum([
            self.picks_available, self.live_data_available, self.history_available,
            self.transfers_available, self.crowd_data_available, 
            self.fixtures_available, self.bootstrap_available
        ])
        return (available_sources / total_sources) * 100

@dataclass
class GameweekSummary:
    """Complete gameweek summary response"""
    manager_id: int
    gameweek: int
    chips_used: List[str]
    picks: List[PlayerPick]
    transfers: List[TransferMove]
    total_transfer_cost: int
    captain_effectiveness: CaptainEffectiveness
    crowd_trends: Dict[str, Any]
    total_points: int
    bench_points: int
    processing_time_ms: float
    data_quality: DataQuality
    
    @property
    def starting_xi(self) -> List[PlayerPick]:
        """Get starting XI players"""
        return [p for p in self.picks if p.is_starting]
    
    @property
    def bench_players(self) -> List[PlayerPick]:
        """Get bench players"""
        return [p for p in self.picks if p.is_bench]
    
    @property
    def has_active_chip(self) -> bool:
        """Check if any chip was used this gameweek"""
        return len(self.chips_used) > 0

# Infrastructure Services
# =======================

class ConfigurationService:
    """Railway-optimized configuration service"""
    
    def __init__(self):
        self.base_url = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
        self.api_timeout = float(os.getenv("API_TIMEOUT", "25.0"))  # Reduced for Railway
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))  # Reduced for Railway
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "6"))
        self.connection_pool_size = int(os.getenv("CONNECTION_POOL_SIZE", "10"))
        
        # Railway-specific settings
        self.port = int(os.getenv("PORT", "8000"))
        self.environment = os.getenv("RAILWAY_ENVIRONMENT", "development")
        self.is_production = self.environment == "production"

class HttpClientService:
    """Railway-optimized HTTP client with advanced error handling"""
    
    def __init__(self, config: ConfigurationService):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with Railway optimizations"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=8.0,  # Railway-optimized
                    read=self.config.api_timeout,
                    write=8.0,
                    pool=5.0
                ),
                limits=httpx.Limits(
                    max_connections=self.config.connection_pool_size,
                    max_keepalive_connections=5
                ),
                headers={
                    "User-Agent": "FPL-Summary-Service/1.0",
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
        
        async with self._semaphore:  # Limit concurrent requests
            for attempt in range(retries + 1):
                try:
                    client = await self.get_client()
                    
                    response = await client.get(
                        url,
                        timeout=timeout,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except Exception as json_error:
                            logger.error(f"JSON parsing error for {context}: {json_error}")
                            return fallback_data
                            
                    elif response.status_code == 404:
                        logger.warning(f"Resource not found ({context}): {url}")
                        return fallback_data
                        
                    elif response.status_code == 429:  # Rate limited
                        if attempt < retries:
                            wait_time = min(2 ** attempt * 2, 30)  # Cap at 30 seconds
                            logger.warning(f"Rate limited ({context}), waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded for {context}: {url}")
                            return fallback_data
                            
                    elif response.status_code >= 500:  # Server error
                        if attempt < retries:
                            wait_time = min(2 ** attempt, 15)
                            logger.warning(f"Server error ({context}), retry {attempt + 1} in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Server error after retries ({context}): {url}")
                            return fallback_data
                    else:
                        logger.warning(f"Unexpected status ({context}): {response.status_code} for {url}")
                        return fallback_data
                        
                except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    if attempt < retries:
                        wait_time = min(2 ** attempt, 10)
                        logger.warning(f"Timeout ({context}) attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Final timeout error ({context}): {str(e)}")
                        return fallback_data
                        
                except (httpx.ConnectError, httpx.NetworkError) as e:
                    if attempt < retries:
                        wait_time = min(2 ** attempt * 1.5, 12)
                        logger.warning(f"Network error ({context}) attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Final network error ({context}): {str(e)}")
                        return fallback_data
                        
                except Exception as e:
                    logger.error(f"Unexpected error ({context}): {str(e)}")
                    return fallback_data
        
        return fallback_data
    
    async def parallel_api_calls(
        self,
        calls: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]]  # (url, context, fallback, params)
    ) -> List[Any]:
        """Execute multiple API calls in parallel with proper error handling"""
        tasks = [
            self.safe_api_call(url, fallback, params=params, context=context)
            for url, context, fallback, params in calls
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {str(result)}")
                    processed_results.append(calls[i][2])  # fallback data
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Parallel API calls failed: {str(e)}")
            return [call[2] for call in calls]  # Return all fallback data
    
    async def cleanup(self):
        """Cleanup HTTP client resources"""
        if self._client:
            await self._client.aclose()
            self._client = None

# Domain Services
# ===============

class CaptainAnalysisService:
    """Service for analyzing captain effectiveness"""
    
    def calculate_enhanced_captain_effectiveness(self, picks: List[PlayerPick]) -> CaptainEffectiveness:
        """Calculate comprehensive captain effectiveness"""
        captain_points = 0
        captain_actual_points = 0
        captain_found = False
        
        # Find captain
        for pick in picks:
            if pick.is_captain:
                captain_points = pick.points
                captain_actual_points = pick.total_points
                captain_found = True
                break
        
        if not captain_found:
            logger.warning("No captain found in picks")
            return CaptainEffectiveness(
                captain_points=0,
                captain_base_points=0,
                max_possible_points=0,
                difference=0,
                efficiency_percentage=0.0,
                verdict=CaptainDecisionQuality.POOR
            )
        
        # Find best possible captain choice
        starting_players = [p for p in picks if p.is_starting]
        max_points = max((p.points for p in starting_players), default=0)
        
        difference = max_points - captain_points
        efficiency = (captain_points / max_points * 100) if max_points > 0 else 0
        
        # Determine verdict
        if captain_points == max_points:
            verdict = CaptainDecisionQuality.OPTIMAL
        elif efficiency >= 80:
            verdict = CaptainDecisionQuality.EXCELLENT
        elif efficiency >= 60:
            verdict = CaptainDecisionQuality.SUBOPTIMAL
        else:
            verdict = CaptainDecisionQuality.POOR
        
        return CaptainEffectiveness(
            captain_points=captain_actual_points,
            captain_base_points=captain_points,
            max_possible_points=max_points,
            difference=difference,
            efficiency_percentage=round(efficiency, 1),
            verdict=verdict
        )

class FixtureDifficultyService:
    """Service for handling fixture difficulty mapping"""
    
    def build_difficulty_map(self, fixtures_data: Dict[str, Any], gameweek: int) -> Tuple[Dict[int, int], bool]:
        """Build fixture difficulty mapping for the gameweek"""
        difficulty_map = {}
        fixtures_available = bool(fixtures_data and fixtures_data.get("fixtures"))
        
        if not fixtures_available:
            logger.warning(f"No fixtures data available for GW{gameweek}")
            return difficulty_map, False
        
        fixtures = fixtures_data.get("fixtures", [])
        mapped_teams = 0
        
        for fixture in fixtures:
            if not (fixture.get("finished") or fixture.get("started")):
                continue
            
            # Handle different fixture data structures
            team_h = fixture.get("team_h", {})
            team_a = fixture.get("team_a", {})
            
            # Extract team IDs (handle both dict and int formats)
            team_h_id = team_h.get("id") if isinstance(team_h, dict) else team_h
            team_a_id = team_a.get("id") if isinstance(team_a, dict) else team_a
            
            # Map difficulties with defaults
            if team_h_id:
                difficulty_map[team_h_id] = fixture.get("team_h_difficulty", 3)
                mapped_teams += 1
            
            if team_a_id:
                difficulty_map[team_a_id] = fixture.get("team_a_difficulty", 3)
                mapped_teams += 1
        
        logger.info(f"Built difficulty map for GW{gameweek}: {mapped_teams} team mappings")
        return difficulty_map, True

class DataEnrichmentService:
    """Service for enriching player data with additional information"""
    
    POSITION_TYPE_MAP = {
        1: PositionType.GOALKEEPER.value,
        2: PositionType.DEFENDER.value,
        3: PositionType.MIDFIELDER.value,
        4: PositionType.FORWARD.value
    }
    
    def __init__(self):
        self.bootstrap_cache: Optional[Dict[str, Any]] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl_minutes = 30  # Cache bootstrap data for 30 minutes
    
    async def get_bootstrap_data(self) -> Dict[str, Any]:
        """Get cached bootstrap data with TTL"""
        now = datetime.utcnow()
        
        if (self.bootstrap_cache is None or 
            self.cache_timestamp is None or 
            (now - self.cache_timestamp).total_seconds() > self.cache_ttl_minutes * 60):
            
            try:
                from utils.bootstrap import get_cached_bootstrap
                self.bootstrap_cache = await get_cached_bootstrap()
                self.cache_timestamp = now
                logger.info("Bootstrap data refreshed")
            except Exception as e:
                logger.error(f"Failed to refresh bootstrap data: {e}")
                if self.bootstrap_cache is None:
                    self.bootstrap_cache = {"elements": []}
        
        return self.bootstrap_cache
    
    def build_player_mappings(self, bootstrap_data: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
        """Build player mapping dictionaries"""
        player_team_map = {}
        player_position_map = {}
        player_name_map = {}
        
        for element in bootstrap_data.get("elements", []):
            player_id = element.get("id")
            if player_id:
                player_team_map[player_id] = element.get("team")
                player_position_map[player_id] = element.get("element_type")
                player_name_map[player_id] = element.get("web_name", f"ID:{player_id}")
        
        return player_team_map, player_position_map, player_name_map
    
    def enrich_picks(
        self,
        picks_data: Dict[str, Any],
        live_points: Dict[int, int],
        difficulty_map: Dict[int, int],
        player_team_map: Dict[int, int],
        player_position_map: Dict[int, int]
    ) -> List[PlayerPick]:
        """Enrich picks with comprehensive data"""
        enriched_picks = []
        
        for pick in picks_data.get("picks", []):
            player_id = pick.get("player_id")
            if player_id is None:
                continue
            
            points = live_points.get(player_id, 0)
            position = pick.get("position", 0)
            
            # Get fixture difficulty
            player_team = player_team_map.get(player_id)
            fixture_difficulty = difficulty_map.get(player_team) if player_team else None
            
            # Get position type
            player_position = player_position_map.get(player_id, 0)
            position_type = self.POSITION_TYPE_MAP.get(player_position, PositionType.UNKNOWN.value)
            
            enriched_pick = PlayerPick(
                id=player_id,
                name=pick.get("player_name", f"ID:{player_id}"),
                multiplier=pick.get("multiplier", 1),
                position=position,
                points=points,
                is_captain=pick.get("is_captain", False),
                is_vice_captain=pick.get("is_vice_captain", False),
                fixture_difficulty=fixture_difficulty,
                position_type=position_type,
                team_id=player_team
            )
            
            enriched_picks.append(enriched_pick)
        
        return enriched_picks
    
    def enrich_transfers(
        self,
        transfers_data: Dict[str, Any],
        player_name_map: Dict[int, str]
    ) -> Tuple[List[TransferMove], int]:
        """Enrich transfer data with player names"""
        transfers_enriched = []
        total_transfer_cost = 0
        
        if not transfers_data or "transfers" not in transfers_data:
            return transfers_enriched, total_transfer_cost
        
        for transfer in transfers_data["transfers"]:
            element_in_id = transfer.get("element_in")
            element_out_id = transfer.get("element_out")
            cost = transfer.get("cost", 0)
            
            transfer_move = TransferMove(
                element_in=player_name_map.get(element_in_id, f"ID:{element_in_id}"),
                element_out=player_name_map.get(element_out_id, f"ID:{element_out_id}"),
                cost=cost,
                element_in_id=element_in_id,
                element_out_id=element_out_id
            )
            
            transfers_enriched.append(transfer_move)
            total_transfer_cost += cost
        
        return transfers_enriched, total_transfer_cost

# Error Handling
# ==============

class SummaryServiceException(Exception):
    """Custom exception for summary service errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

async def summary_exception_handler(request: Request, exc: SummaryServiceException):
    """Global exception handler for summary service"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": str(request.url)
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
captain_service = CaptainAnalysisService()
fixture_service = FixtureDifficultyService()
enrichment_service = DataEnrichmentService()

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for Railway monitoring"""
    return {
        "status": "healthy",
        "service": "gameweek-summary",
        "environment": config.environment,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/tools/get_manager_gameweek_summary/{tid}/{gw}")
async def get_manager_gameweek_summary(tid: int, gw: int) -> Dict[str, Any]:
    """Enhanced gameweek summary with Railway optimizations and comprehensive error handling"""
    start_time = time.time()
    
    try:
        # Input validation
        if not (1 <= gw <= 38):
            raise SummaryServiceException("Invalid gameweek number", 400)
        if tid <= 0:
            raise SummaryServiceException("Invalid team ID", 400)
        
        # Prepare parallel API calls - Railway optimized
        api_calls = [
            (f"{config.base_url}/tools/get_manager_picks/{tid}/{gw}", "picks", {}, None),
            (f"{config.base_url}/tools/get_manager_history/{tid}", "history", {}, None),
            (f"{config.base_url}/tools/get_transfers_by_gw/{tid}/{gw}", "transfers", {}, None),
            (f"{config.base_url}/tools/get_event_live/{gw}", "live", {}, None),
            (f"{config.base_url}/tools/get_crowd_trends_by_gw/{gw}", "crowd", {}, None),
            (f"{config.base_url}/tools/get_fixtures_by_gw", "fixtures", {}, {"gw": gw})
        ]
        
        # Execute all API calls in parallel
        logger.info(f"Starting parallel data fetch for manager {tid}, GW{gw}")
        results = await http_client.parallel_api_calls(api_calls)
        
        picks_data, history_data, transfers_data, live_data, crowd_data, fixtures_data = results
        
        # Validate critical data
        if not picks_data or "picks" not in picks_data:
            raise SummaryServiceException("Manager picks not found", 404)
        
        # Get bootstrap data with caching
        bootstrap_data = await enrichment_service.get_bootstrap_data()
        player_team_map, player_position_map, player_name_map = enrichment_service.build_player_mappings(bootstrap_data)
        
        # Build fixture difficulty mapping
        difficulty_map, fixtures_available = fixture_service.build_difficulty_map(fixtures_data, gw)
        
        # Parse chips used
        chips_used = []
        if history_data and "chips" in history_data:
            chips_used = [
                chip["name"] for chip in history_data["chips"] 
                if chip.get("event") == gw
            ]
        
        # Prepare live points mapping
        live_points = {}
        if live_data and "players" in live_data:
            live_points = {
                e.get("player_id"): e.get("points", 0) 
                for e in live_data["players"] 
                if e.get("player_id")
            }
        
        # Enrich picks data
        picks_enriched = enrichment_service.enrich_picks(
            picks_data, live_points, difficulty_map, player_team_map, player_position_map
        )
        
        # Enrich transfers data
        transfers_enriched, total_transfer_cost = enrichment_service.enrich_transfers(
            transfers_data, player_name_map
        )
        
        # Calculate captain effectiveness
        captain_effectiveness = captain_service.calculate_enhanced_captain_effectiveness(picks_enriched)
        
        # Calculate points totals
        total_points = sum(p.total_points for p in picks_enriched if p.is_starting)
        bench_points = sum(p.points for p in picks_enriched if p.is_bench)
        
        # Assess data quality
        data_quality = DataQuality(
            picks_available=bool(picks_data),
            live_data_available=bool(live_data),
            history_available=bool(history_data),
            transfers_available=bool(transfers_data),
            crowd_data_available=bool(crowd_data),
            fixtures_available=fixtures_available,
            bootstrap_available=bool(bootstrap_data.get("elements"))
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Build comprehensive response
        summary = GameweekSummary(
            manager_id=tid,
            gameweek=gw,
            chips_used=chips_used,
            picks=picks_enriched,
            transfers=transfers_enriched,
            total_transfer_cost=total_transfer_cost,
            captain_effectiveness=captain_effectiveness,
            crowd_trends=crowd_data or {},
            total_points=total_points,
            bench_points=bench_points,
            processing_time_ms=round(processing_time, 2),
            data_quality=data_quality
        )
        
        # Convert to dictionary format for API response
        return {
            "manager_id": summary.manager_id,
            "gameweek": summary.gameweek,
            "chips_used": summary.chips_used,
            "picks": [
                {
                    "id": p.id,
                    "name": p.name,
                    "multiplier": p.multiplier,
                    "is_captain": p.is_captain,
                    "is_vice_captain": p.is_vice_captain,
                    "position": p.position,
                    "points": p.points,
                    "is_starting": p.is_starting,
                    "is_bench": p.is_bench,
                    "fixture_difficulty": p.fixture_difficulty,
                    "position_type": p.position_type,
                    "team_id": p.team_id
                }
                for p in summary.picks
            ],
            "transfers": [
                {
                    "element_in": t.element_in,
                    "element_out": t.element_out,
                    "cost": t.cost
                }
                for t in summary.transfers
            ],
            "total_transfer_cost": summary.total_transfer_cost,
            "captain_effectiveness": {
                "captain_points": summary.captain_effectiveness.captain_points,
                "captain_base_points": summary.captain_effectiveness.captain_base_points,
                "max_possible_points": summary.captain_effectiveness.max_possible_points,
                "difference": summary.captain_effectiveness.difference,
                "efficiency_percentage": summary.captain_effectiveness.efficiency_percentage,
                "verdict": summary.captain_effectiveness.verdict.value
            },
            "crowd_trends": summary.crowd_trends,
            "total_points": summary.total_points,
            "bench_points": summary.bench_points,
            "processing_time_ms": summary.processing_time_ms,
            "data_quality": {
                "picks_available": summary.data_quality.picks_available,
                "live_data_available": summary.data_quality.live_data_available,
                "history_available": summary.data_quality.history_available,
                "transfers_available": summary.data_quality.transfers_available,
                "crowd_data_available": summary.data_quality.crowd_data_available,
                "fixtures_available": summary.data_quality.fixtures_available,
                "bootstrap_available": summary.data_quality.bootstrap_available,
                "completeness_score": summary.data_quality.completeness_score
            }
        }
        
    except SummaryServiceException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in gameweek summary: {str(e)}")
        raise SummaryServiceException(f"Internal server error: {str(e)}", 500)

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    await http_client.cleanup()
    logger.info("Summary service shutdown complete")

