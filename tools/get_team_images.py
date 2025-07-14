from fastapi import APIRouter
from typing import List, Dict

router = APIRouter()

# Manually define the 2025/26 teams and their FPL 'code' values
TEAMS_2025_26 = [
    {"id": 1, "name": "Arsenal", "short_name": "ARS", "code": 3},
    {"id": 2, "name": "Aston Villa", "short_name": "AVL", "code": 7},
    {"id": 3, "name": "Bournemouth", "short_name": "BOU", "code": 91},
    {"id": 4, "name": "Brentford", "short_name": "BRE", "code": 94},
    {"id": 5, "name": "Brighton", "short_name": "BHA", "code": 36},
    {"id": 6, "name": "Burnley", "short_name": "BUR", "code": 90},
    {"id": 7, "name": "Chelsea", "short_name": "CHE", "code": 8},
    {"id": 8, "name": "Crystal Palace", "short_name": "CRY", "code": 31},
    {"id": 9, "name": "Everton", "short_name": "EVE", "code": 11},
    {"id": 10, "name": "Fulham", "short_name": "FUL", "code": 54},
    {"id": 11, "name": "Liverpool", "short_name": "LIV", "code": 14},
    {"id": 12, "name": "Luton", "short_name": "LUT", "code": 98},
    {"id": 13, "name": "Man City", "short_name": "MCI", "code": 43},
    {"id": 14, "name": "Man Utd", "short_name": "MUN", "code": 1},
    {"id": 15, "name": "Newcastle", "short_name": "NEW", "code": 4},
    {"id": 16, "name": "Nott'm Forest", "short_name": "NFO", "code": 17},
    {"id": 17, "name": "Sheffield Utd", "short_name": "SHU", "code": 49},
    {"id": 18, "name": "Spurs", "short_name": "TOT", "code": 6},
    {"id": 19, "name": "West Ham", "short_name": "WHU", "code": 21},
    {"id": 20, "name": "Wolves", "short_name": "WOL", "code": 39}
]

@router.get("/tools/get_team_images", response_model=List[Dict])
def get_team_images():
    """Returns shirt and badge image URLs for each PL team in 2025/26."""
    enriched = []
    for team in TEAMS_2025_26:
        team_data = {
            "id": team["id"],
            "name": team["name"],
            "short_name": team["short_name"],
            "code": team["code"],
            "shirt_url": f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{team['code']}-220.webp",
            "badge_url": f"https://resources.premierleague.com/premierleague/badges/25/t{team['code']}.png"
        }
        enriched.append(team_data)
    return enriched
