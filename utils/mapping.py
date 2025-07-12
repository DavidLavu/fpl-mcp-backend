# utils/mapping.py

from typing import Dict, List, Any

def build_team_map(data: Dict[str, Any]) -> Dict[int, str]:
    """
    Build a mapping of team IDs to team names.
    :param data: The full bootstrap-static JSON.
    :return: { team_id: team_name }
    """
    return {team["id"]: team["name"] for team in data.get("teams", [])}


def build_position_map(data: Dict[str, Any]) -> Dict[int, str]:
    """
    Build a mapping of element_type IDs to their singular position names.
    :param data: The full bootstrap-static JSON.
    :return: { element_type_id: position_name }
    """
    return {etype["id"]: etype["singular_name"] for etype in data.get("element_types", [])}


def map_team(team_id: int, team_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Turn a raw team_id into a dict with id+name.
    :param team_id: The numeric team ID.
    :param team_map: The { id: name } map.
    :return: {"id": team_id, "name": team_name}
    """
    return {"id": team_id, "name": team_map.get(team_id, "Unknown")}


def map_position(pos_id: int, position_map: Dict[int, str]) -> str:
    """
    Look up the human‐readable position name.
    :param pos_id: The numeric element_type.
    :param position_map: The { element_type_id: position_name } map.
    :return: e.g. "Goalkeeper", "Midfielder", or "Unknown".
    """
    return position_map.get(pos_id, "Unknown")


def build_player_map(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Build a rich player info map from bootstrap data.
    :param data: The full bootstrap-static JSON.
    :return: 
      {
        player_id: {
          "id": player_id,
          "first_name": ...,
          "second_name": ...,
          "web_name": ...,
          "element_type": ...,
          "team": ...
        },
        ...
      }
    """
    result: Dict[int, Dict[str, Any]] = {}
    for p in data.get("elements", []):
        result[p["id"]] = {
            "id": p["id"],
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "web_name": p["web_name"],
            "element_type": p["element_type"],
            "team": p["team"],
        }
    return result
