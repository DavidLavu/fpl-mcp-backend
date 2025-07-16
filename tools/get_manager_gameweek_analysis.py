
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx
import time
import logging
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# Domain Models (TypeScript-style structure)
# ==========================================

class PositionType(str, Enum):
    """Player position types with standardized values"""
    GOALKEEPER = "Goalkeeper"
    DEFENDER = "Defender" 
    MIDFIELDER = "Midfielder"
    FORWARD = "Forward"
    UNKNOWN = "Unknown"

class FormationType(str, Enum):
    """Formation tactical analysis types"""
    ATTACKING = "Attacking formation with wing-backs, high risk/reward"
    MIDFIELD_HEAVY = "Midfield-heavy, good for controlling games"
    BALANCED_ATTACKING = "Balanced attacking formation, popular choice"
    CLASSIC_BALANCED = "Classic balanced formation, solid defensive structure"
    DEFENSIVE = "Defensive formation, prioritizes clean sheets"
    VERY_DEFENSIVE = "Very defensive, focuses on defensive returns"
    ULTRA_DEFENSIVE = "Ultra-defensive, minimal attacking threat"
    CUSTOM = "Custom formation"

@dataclass
class PlayerPick:
    """Enhanced player pick with type safety"""
    id: int
    name: str
    points: int
    multiplier: int
    is_starting: bool
    is_captain: bool
    is_bench: bool
    position_type: str
    fixture_difficulty: Optional[int] = None
    
    @property
    def total_points(self) -> int:
        """Calculate total points including multiplier"""
        return self.points * self.multiplier

@dataclass
class ApiResponse:
    """Standardized API response wrapper"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None

# Infrastructure Services
# =======================

class ConfigurationService:
    """Centralized configuration management"""
    
    def __init__(self):
        self.base_url = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
        self.api_timeout = float(os.getenv("API_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Railway-specific configurations
        self.port = int(os.getenv("PORT", "8000"))
        self.environment = os.getenv("RAILWAY_ENVIRONMENT", "development")
        self.is_production = self.environment == "production"

class HttpClientService:
    """Enhanced HTTP client with Railway-optimized settings"""
    
    def __init__(self, config: ConfigurationService):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    @asynccontextmanager
    async def get_client(self):
        """Get properly configured HTTP client"""
        if self._client is None:
            # Railway-optimized client settings
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,  # Reduced for Railway
                    read=self.config.api_timeout,
                    write=10.0,
                    pool=5.0
                ),
                limits=httpx.Limits(
                    max_connections=20,  # Railway connection limits
                    max_keepalive_connections=5
                ),
                # Railway often requires explicit headers
                headers={
                    "User-Agent": "FPL-Analysis-Service/1.0",
                    "Accept": "application/json",
                    "Connection": "keep-alive"
                }
            )
        
        try:
            yield self._client
        finally:
            # Don't close immediately, let it be reused
            pass
    
    async def safe_api_call(
        self, 
        url: str, 
        fallback_data: Any = None,
        retries: int = None
    ) -> Any:
        """Enhanced API call with Railway-specific error handling"""
        retries = retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                async with self.get_client() as client:
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 404:
                        logger.warning(f"Resource not found: {url}")
                        return fallback_data
                    elif response.status_code >= 500:
                        # Server error, retry
                        if attempt < retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Server error after {retries} retries: {url}")
                            return fallback_data
                    else:
                        logger.warning(f"API call failed: {url}, status: {response.status_code}")
                        return fallback_data
                        
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                logger.warning(f"Network error (attempt {attempt + 1}/{retries + 1}): {str(e)}")
                
                if attempt < retries:
                    # Railway networking can be flaky, use progressive backoff
                    await asyncio.sleep(min(2 ** attempt, 10))
                    continue
                else:
                    logger.error(f"Final network error for {url}: {str(e)}")
                    return fallback_data
                    
            except Exception as e:
                logger.error(f"Unexpected error calling {url}: {str(e)}")
                return fallback_data
        
        return fallback_data
    
    async def cleanup(self):
        """Cleanup client resources"""
        if self._client:
            await self._client.aclose()
            self._client = None

# Domain Services
# ===============

class FormationAnalysisService:
    """Service for analyzing team formations with tactical insights"""
    
    FORMATION_MAPPING = {
        "3-4-3": FormationType.ATTACKING,
        "3-5-2": FormationType.MIDFIELD_HEAVY,
        "4-3-3": FormationType.BALANCED_ATTACKING,
        "4-4-2": FormationType.CLASSIC_BALANCED,
        "4-5-1": FormationType.DEFENSIVE,
        "5-3-2": FormationType.VERY_DEFENSIVE,
        "5-4-1": FormationType.ULTRA_DEFENSIVE
    }
    
    POSITION_MAPPING = {
        PositionType.GOALKEEPER: "GKP",
        PositionType.DEFENDER: "DEF",
        PositionType.MIDFIELDER: "MID",
        PositionType.FORWARD: "FWD"
    }
    
    def analyze_formation(self, picks: List[PlayerPick]) -> Dict[str, Any]:
        """Analyze team formation with enhanced tactical insights"""
        starting_players = [p for p in picks if p.is_starting]
        formation_count = Counter()
        
        # Count positions (excluding goalkeeper)
        for player in starting_players:
            try:
                pos_type = PositionType(player.position_type)
            except ValueError:
                pos_type = PositionType.UNKNOWN
                
            mapped_pos = self.POSITION_MAPPING.get(pos_type)
            if mapped_pos and mapped_pos != "GKP":
                formation_count[mapped_pos] += 1
        
        formation_string = f"{formation_count.get('DEF', 0)}-{formation_count.get('MID', 0)}-{formation_count.get('FWD', 0)}"
        formation_type = self.FORMATION_MAPPING.get(formation_string, FormationType.CUSTOM)
        
        return {
            "formation": formation_string,
            "defender_count": formation_count.get('DEF', 0),
            "midfielder_count": formation_count.get('MID', 0),
            "forward_count": formation_count.get('FWD', 0),
            "is_balanced": all(count >= 2 for count in formation_count.values()),
            "formation_type": formation_type.value,
            "tactical_analysis": self._get_tactical_insights(formation_string, formation_count)
        }
    
    def _get_tactical_insights(self, formation: str, counts: Counter) -> Dict[str, Any]:
        """Generate tactical insights based on formation"""
        total_outfield = sum(counts.values())
        
        return {
            "attacking_potential": counts.get('FWD', 0) / total_outfield if total_outfield > 0 else 0,
            "defensive_stability": counts.get('DEF', 0) / total_outfield if total_outfield > 0 else 0,
            "midfield_control": counts.get('MID', 0) / total_outfield if total_outfield > 0 else 0,
            "balance_score": 1 - abs(counts.get('DEF', 0) - counts.get('FWD', 0)) / 5,
            "recommended_captain_position": "Forward" if counts.get('FWD', 0) >= 3 else "Midfielder"
        }

class ValueEfficiencyService:
    """Service for calculating and analyzing player value efficiency"""
    
    def calculate_value_efficiency(
        self, 
        picks: List[PlayerPick], 
        bootstrap_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate points per million with enhanced analytics"""
        try:
            element_prices = {
                el["id"]: el["now_cost"] / 10 
                for el in bootstrap_data.get("elements", [])
            }
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to extract player prices: {e}")
            return self._get_fallback_efficiency_data()
        
        efficiency_data = []
        total_starting_value = 0
        total_starting_points = 0
        
        for player in picks:
            if player.is_starting:
                price = element_prices.get(player.id, 0)
                if price <= 0:
                    continue
                    
                points = player.total_points
                ppm = points / price
                
                efficiency_data.append({
                    "name": player.name,
                    "price": price,
                    "points": points,
                    "points_per_million": round(ppm, 2),
                    "position_type": player.position_type,
                    "value_rating": self._get_value_rating(ppm)
                })
                
                total_starting_value += price
                total_starting_points += points
        
        if not efficiency_data:
            return self._get_fallback_efficiency_data()
        
        # Sort by efficiency
        efficiency_data.sort(key=lambda x: x["points_per_million"], reverse=True)
        
        return {
            "players": efficiency_data,
            "team_average_ppm": round(total_starting_points / total_starting_value, 2) if total_starting_value > 0 else 0,
            "best_value": efficiency_data[0],
            "worst_value": efficiency_data[-1],
            "total_starting_value": round(total_starting_value, 1),
            "value_distribution": self._analyze_value_distribution(efficiency_data)
        }
    
    def _get_value_rating(self, ppm: float) -> str:
        """Rate player value efficiency"""
        if ppm >= 3.0:
            return "Excellent"
        elif ppm >= 2.0:
            return "Good"
        elif ppm >= 1.0:
            return "Average"
        else:
            return "Poor"
    
    def _analyze_value_distribution(self, efficiency_data: List[Dict]) -> Dict[str, Any]:
        """Analyze value distribution across the team"""
        if not efficiency_data:
            return {}
            
        ppms = [p["points_per_million"] for p in efficiency_data]
        return {
            "excellent_value_count": sum(1 for ppm in ppms if ppm >= 3.0),
            "poor_value_count": sum(1 for ppm in ppms if ppm < 1.0),
            "average_ppm": sum(ppms) / len(ppms),
            "ppm_std_dev": (sum((x - sum(ppms)/len(ppms))**2 for x in ppms) / len(ppms))**0.5
        }
    
    def _get_fallback_efficiency_data(self) -> Dict[str, Any]:
        """Fallback data when efficiency calculation fails"""
        return {
            "players": [],
            "team_average_ppm": 0,
            "best_value": None,
            "worst_value": None,
            "total_starting_value": 0,
            "value_distribution": {},
            "error": "Unable to calculate value efficiency"
        }

# Error Handling
# ==============

class FPLAnalysisException(Exception):
    """Custom exception for FPL analysis errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

async def fpl_exception_handler(request: Request, exc: FPLAnalysisException):
    """Global exception handler for FPL analysis errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Router Setup with Railway Optimizations
# =======================================

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Railway captures stdout
)
logger = logging.getLogger(__name__)

# Initialize services
config = ConfigurationService()
http_client = HttpClientService(config)
formation_service = FormationAnalysisService()
value_service = ValueEfficiencyService()

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "environment": config.environment,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/tools/get_manager_gameweek_analysis/{tid}/{gw}")
async def get_manager_gameweek_analysis(tid: int, gw: int) -> ApiResponse:
    """Enhanced gameweek analysis with Railway optimizations"""
    start_time = time.time()
    
    try:
        # Validate inputs
        if not (1 <= gw <= 38):
            raise FPLAnalysisException("Invalid gameweek number", 400)
        if tid <= 0:
            raise FPLAnalysisException("Invalid team ID", 400)
        
        # Get enriched summary data with retries
        summary_url = f"{config.base_url}/tools/get_manager_gameweek_summary/{tid}/{gw}"
        summary = await http_client.safe_api_call(summary_url, {})
        
        if not summary or "picks" not in summary:
            raise FPLAnalysisException("Failed to retrieve gameweek summary", 503)
        
        # Convert to domain objects
        try:
            picks = [
                PlayerPick(
                    id=p.get("id", 0),
                    name=p.get("name", "Unknown"),
                    points=p.get("points", 0),
                    multiplier=p.get("multiplier", 1),
                    is_starting=p.get("is_starting", False),
                    is_captain=p.get("is_captain", False),
                    is_bench=p.get("is_bench", False),
                    position_type=p.get("position_type", "Unknown"),
                    fixture_difficulty=p.get("fixture_difficulty")
                )
                for p in summary.get("picks", [])
            ]
        except Exception as e:
            logger.error(f"Failed to parse picks data: {e}")
            raise FPLAnalysisException("Invalid picks data format", 500)
        
        # Get bootstrap data with error handling
        try:
            # This should be imported from your utils
            from utils.bootstrap import get_cached_bootstrap
            bootstrap = await get_cached_bootstrap()
        except Exception as e:
            logger.error(f"Failed to get bootstrap data: {e}")
            bootstrap = {"elements": []}  # Fallback
        
        # Perform analysis
        formation_analysis = formation_service.analyze_formation(picks)
        value_efficiency = value_service.calculate_value_efficiency(picks, bootstrap)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Build comprehensive response
        analysis_result = {
            "meta": {
                "analysis_type": "enhanced_comprehensive_gameweek_analysis",
                "manager_id": tid,
                "gameweek": gw,
                "generated_at": datetime.utcnow().isoformat(),
                "processing_time_ms": round(processing_time, 2),
                "environment": config.environment,
                "data_quality": {
                    "summary_available": bool(summary),
                    "bootstrap_available": bool(bootstrap.get("elements")),
                    "total_picks": len(picks)
                }
            },
            "total_points": summary.get("total_points", 0),
            "formation_analysis": formation_analysis,
            "value_efficiency": value_efficiency,
            "picks_count": len(picks),
            "starting_xi_count": len([p for p in picks if p.is_starting])
        }
        
        return ApiResponse(
            success=True,
            data=analysis_result,
            processing_time_ms=round(processing_time, 2)
        )
        
    except FPLAnalysisException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in gameweek analysis: {str(e)}")
        raise FPLAnalysisException(f"Internal server error: {str(e)}", 500)

# Cleanup on shutdown
@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    await http_client.cleanup()
    logger.info("FPL Analysis service shutdown complete")

