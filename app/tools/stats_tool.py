import pandas as pd
from langsmith import traceable
from statsbombpy import sb

COMPETITION_NAME_TO_ID: dict[str, int] = {
    "FIFA Women's World Cup": 72,
    "FA Women's Super League": 37,
    "NWSL": 49,
    "UEFA Women's Euro": 53,
}


def _latest_season(comps: pd.DataFrame, competition_id: int) -> tuple[int, str] | None:
    """Return (season_id, season_name) for the most recent season of a competition.

    Args:
        comps: Full competitions DataFrame from statsbombpy.
        competition_id: StatsBomb competition ID.

    Returns:
        Tuple of (season_id, season_name), or None if not found.
    """
    subset = comps[comps["competition_id"] == competition_id].sort_values(
        "season_name", ascending=False
    )
    if subset.empty:
        return None
    row = subset.iloc[0]
    return int(row["season_id"]), str(row["season_name"])


def _team_name(match_row: pd.Series, side: str) -> str:
    """Extract team name from a match row.

    Args:
        match_row: A single row from the matches DataFrame.
        side: Either 'home' or 'away'.

    Returns:
        Team name string.
    """
    val = match_row.get(f"{side}_team")
    if val is not None and str(val) not in ("nan", ""):
        return str(val)
    return "?"


def _format_matches(
    matches: pd.DataFrame, competition_name: str, season_name: str
) -> str:
    """Format a matches DataFrame into a readable text block.

    Args:
        matches: DataFrame of matches from statsbombpy.
        competition_name: Name of the competition.
        season_name: Name of the season.

    Returns:
        Multi-line string with one match result per line.
    """
    lines = [f"{competition_name} {season_name}:"]
    for _, row in matches.iterrows():
        home = _team_name(row, "home")
        away = _team_name(row, "away")
        hs = row.get("home_score", "?")
        as_ = row.get("away_score", "?")
        date = row.get("match_date", "?")
        lines.append(f"  {date}: {home} {hs}-{as_} {away}")
    return "\n".join(lines)


@traceable(name="tool:search_stats")
def search_stats(query: str, competition: str | None = None) -> str:
    """Fetch live women's football match results from StatsBomb open data.

    Args:
        query: Natural language query (used to determine scope, not for semantic search).
        competition: Optional competition name to filter to one competition.
                     Must be one of the keys in COMPETITION_NAME_TO_ID.

    Returns:
        Formatted string of match results, or an error message.
    """
    try:
        all_comps = sb.competitions()
        women = all_comps[all_comps["competition_gender"] == "female"]

        if competition and competition in COMPETITION_NAME_TO_ID:
            cid = COMPETITION_NAME_TO_ID[competition]
            target_ids = [cid]
        else:
            target_ids = list(COMPETITION_NAME_TO_ID.values())

        blocks: list[str] = []
        for cid in target_ids:
            result = _latest_season(women, cid)
            if result is None:
                continue
            sid, sname = result
            cname_rows = women[women["competition_id"] == cid]
            cname = (
                str(cname_rows.iloc[0]["competition_name"])
                if not cname_rows.empty
                else str(cid)
            )
            try:
                matches = sb.matches(competition_id=cid, season_id=sid)
                if not matches.empty:
                    blocks.append(_format_matches(matches, cname, sname))
            except Exception:
                continue

        if not blocks:
            return "No match statistics found."
        return "\n\n".join(blocks)

    except Exception as exc:
        return f"Error fetching stats: {exc}"
