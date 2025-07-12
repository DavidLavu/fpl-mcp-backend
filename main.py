from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Tool imports
from tools import (
    get_bootstrap_data, get_fixtures, get_event_live, get_dream_team, get_fixtures_by_gw,
    get_manager_info, get_manager_history, get_manager_picks, get_transfers_by_gw,
    get_set_piece_notes, get_top_team_values, get_player_history, get_player_fixtures,
    get_player_profile, get_crowd_trends_by_gw, get_classic_league_standings,
    get_league_captains, get_manager_gameweek_summary, get_manager_gameweek_analysis,
    get_upcoming_gameweek_planner, get_rival_comparison
)

app = FastAPI()

# Enable CORS for frontend and ChatGPT access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development
        "http://localhost:3000",
        "https://localhost:3000"
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Register your routers
app.include_router(get_bootstrap_data.router)
app.include_router(get_fixtures.router)
app.include_router(get_event_live.router)
app.include_router(get_dream_team.router)
app.include_router(get_fixtures_by_gw.router)
app.include_router(get_manager_info.router)
app.include_router(get_manager_history.router)
app.include_router(get_manager_picks.router)
app.include_router(get_transfers_by_gw.router)
app.include_router(get_set_piece_notes.router)
app.include_router(get_top_team_values.router)
app.include_router(get_player_history.router)
app.include_router(get_player_fixtures.router)
app.include_router(get_player_profile.router)
app.include_router(get_crowd_trends_by_gw.router)
app.include_router(get_classic_league_standings.router)
app.include_router(get_league_captains.router)
app.include_router(get_manager_gameweek_summary.router)
app.include_router(get_manager_gameweek_analysis.router)
app.include_router(get_upcoming_gameweek_planner.router)
app.include_router(get_rival_comparison.router)


# Run app if executed directly (used locally or in Procfile)
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)