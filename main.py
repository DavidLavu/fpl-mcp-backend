from fastapi import FastAPI
from tools import get_bootstrap_data, get_fixtures, get_event_live, get_dream_team, get_fixtures_by_gw
from tools import get_manager_info
from tools import get_manager_history
from tools import get_manager_picks
from tools import get_transfers_by_gw
from tools import get_set_piece_notes
from tools import get_top_team_values
from tools import get_player_history
from tools import get_player_fixtures
from tools import get_player_profile
from tools import get_crowd_trends_by_gw
from tools import get_classic_league_standings
from tools import get_league_captains
from tools import get_manager_gameweek_summary
from tools import get_manager_gameweek_analysis
from tools import get_upcoming_gameweek_planner
from tools import get_rival_comparison




app = FastAPI()

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