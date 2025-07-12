import httpx
from datetime import datetime, timedelta

_bootstrap_data = None
_expires_at = None

async def get_cached_bootstrap():
    """Fetches and caches bootstrap-static data for 10 minutes."""
    global _bootstrap_data, _expires_at

    now = datetime.utcnow()
    if _bootstrap_data is None or _expires_at is None or now > _expires_at:
        print("⏬ Fetching new bootstrap data from FPL API")
        async with httpx.AsyncClient() as client:
            response = await client.get("https://fantasy.premierleague.com/api/bootstrap-static/")
            _bootstrap_data = response.json()
            _expires_at = now + timedelta(minutes=10)
    else:
        print("✅ Using cached bootstrap data")

    return _bootstrap_data


async def build_element_map():
    """Returns mapping dicts: player_id → name, team_id → name, and player positions."""
    data = await get_cached_bootstrap()

    player_map = {
        player["id"]: player["web_name"]
        for player in data["elements"]
    }

    team_map = {
        team["id"]: team["name"]
        for team in data["teams"]
    }

    position_map = {
        pos["id"]: pos["singular_name"]
        for pos in data["element_types"]
    }

    return {
        "players": player_map,
        "teams": team_map,
        "positions": position_map
    }
