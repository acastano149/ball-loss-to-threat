import matplotlib
matplotlib.use('Agg')

import asyncio
import functools
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from statsbombpy import sb
from mplsoccer import Pitch
import shinyswatch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PITCH_LENGTH = 120
PITCH_WIDTH = 80
BOX_ENTRY_X = 102

PLAY_PATTERN_CHOICES = {
    "Regular Play": "Juego Regular",
    "From Counter": "Contraataque",
    "From Keeper": "Desde Portero",
    "From Corner": "Córner",
    "From Free Kick": "Falta",
    "From Throw In": "Saque de Banda",
    "From Goal Kick": "Saque de Puerta",
    "From Kick Off": "Saque de Centro",
}

DANGER_CRITERIA_CHOICES = {
    "shot": "Tiro",
    "box_entry": "Entrada al área",
    "from_counter": "From Counter",
}

DANGER_COMBINE_CHOICES = {
    "any": "Cualquiera de los criterios (OR)",
    "all": "Todos los criterios seleccionados (AND)",
}

DANGER_SCOPE_CHOICES = {
    "possession": "Posesión completa",
    "window": "Posesión + ventana temporal",
}

COLS_NEEDED = [
    "id", "index", "period", "timestamp", "minute", "second",
    "type", "play_pattern", "possession", "possession_team", "team",
    "location", "under_pressure",
    "shot_outcome", "shot_statsbomb_xg",
    "ball_recovery_recovery_failure",
    "interception_outcome", "duel_type", "duel_outcome",
    "pass_outcome", "pass_length", "pass_angle", "pass_height", "pass_end_location", "pass_type",
    "carry_end_location",
    "dribble_outcome",
    "ball_receipt_outcome",
    "miscontrol_aerial_won",
    "counterpress",
]

CATEGORY_COLS = [
    "type", "play_pattern", "possession_team", "team", "shot_outcome",
    "interception_outcome", "duel_type", "duel_outcome", "pass_outcome",
    "pass_height", "pass_type", "dribble_outcome", "ball_receipt_outcome",
]

TRANSITION_RECOVERY_SUCCESS = ["Won", "Success", "Success In Play", "In Play"]
TACKLE_SUCCESS = ["Won", "Success", "Success In Play"]
LOSS_CANDIDATE_TYPES = ["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed", "Ball Receipt*"]

# ─────────────────────────────────────────────
# HELPERS DE DATOS
# ─────────────────────────────────────────────

def _safe_list(value):
    return isinstance(value, (list, tuple)) and len(value) >= 2


def _x_from_location(value):
    return float(value[0]) if _safe_list(value) else np.nan


def _y_from_location(value):
    return float(value[1]) if _safe_list(value) else np.nan


def _end_location(row):
    for col in ["pass_end_location", "carry_end_location"]:
        value = row.get(col, None)
        if _safe_list(value):
            return float(value[0]), float(value[1])
    return np.nan, np.nan


def _end_x(row):
    x_end, _ = _end_location(row)
    return x_end


def _timestamp_to_seconds(value):
    if pd.isna(value):
        return np.nan
    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
    try:
        parts = str(value).split(":")
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
    except Exception:
        return np.nan
    return np.nan


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    for col in COLS_NEEDED:
        if col not in events.columns:
            events[col] = pd.NA

    if events["index"].isna().all():
        events["event_order"] = np.arange(len(events))
    else:
        events["event_order"] = pd.to_numeric(events["index"], errors="coerce")
        events["event_order"] = events["event_order"].fillna(pd.Series(np.arange(len(events)), index=events.index))

    if not events["minute"].isna().all() and not events["second"].isna().all():
        events["event_seconds"] = (
            pd.to_numeric(events["minute"], errors="coerce").fillna(0) * 60
            + pd.to_numeric(events["second"], errors="coerce").fillna(0)
        )
    else:
        events["event_seconds"] = events["timestamp"].apply(_timestamp_to_seconds)

    events["period"] = pd.to_numeric(events["period"], errors="coerce").fillna(0).astype(int)
    events["possession"] = pd.to_numeric(events["possession"], errors="coerce")
    events["x"] = events["location"].apply(_x_from_location)
    events["y"] = events["location"].apply(_y_from_location)

    for col in CATEGORY_COLS:
        if col in events.columns:
            try:
                events[col] = events[col].astype("category")
            except Exception:
                pass

    return events.sort_values(["period", "event_order"]).reset_index(drop=True)


@functools.lru_cache(maxsize=256)
def _fetch_events_cached(match_id: int) -> pd.DataFrame:
    events = sb.events(match_id=int(match_id))
    keep = [c for c in COLS_NEEDED if c in events.columns]
    return _prepare_events(events[keep].copy())


def _possession_meta(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["match_id", "period", "possession"] if "match_id" in events.columns else ["period", "possession"]
    meta = (
        events.dropna(subset=["possession"])
        .sort_values(["period", "event_order"])
        .groupby(group_cols, observed=True)
        .agg(
            possession_team=("possession_team", "first"),
            play_pattern=("play_pattern", "first"),
            start_order=("event_order", "min"),
            end_order=("event_order", "max"),
            start_seconds=("event_seconds", "min"),
            end_seconds=("event_seconds", "max"),
        )
        .reset_index()
    )
    return meta


def _next_possession_for_team(meta: pd.DataFrame, match_id: int, period: int, order: float, team: str):
    candidates = meta[
        (meta["match_id"] == match_id)
        & (meta["period"] == period)
        & (meta["possession_team"] == team)
        & (meta["start_order"] >= order)
    ].sort_values("start_order")
    if candidates.empty:
        return np.nan, None
    row = candidates.iloc[0]
    return row["possession"], row["play_pattern"]


def _next_opponent_possession(meta: pd.DataFrame, match_id: int, period: int, order: float, team: str):
    candidates = meta[
        (meta["match_id"] == match_id)
        & (meta["period"] == period)
        & (meta["possession_team"] != team)
        & (meta["start_order"] > order)
    ].sort_values("start_order")
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _box_entry_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    loc_entry = df["x"].fillna(-1) >= BOX_ENTRY_X
    end_entry = df.apply(lambda r: _end_x(r), axis=1).fillna(-1) >= BOX_ENTRY_X
    return loc_entry | end_entry


def _combine_danger_flags(flags: dict, criteria, combine: str) -> int:
    criteria = list(criteria or [])
    allowed = {"shot", "box_entry", "from_counter"}
    criteria = [c for c in criteria if c in allowed]
    if not criteria:
        return 0
    values = [bool(flags.get(f"danger_{c}", 0)) for c in criteria]
    return int(all(values) if combine == "all" else any(values))


def _danger_definition_text(criteria=None, combine="any") -> str:
    criteria = list(criteria or [])
    names = {"shot": "tiro", "box_entry": "entrada al área", "from_counter": "From Counter"}
    if not criteria:
        return "sin criterios activos"
    separator = " y " if combine == "all" else " o "
    return separator.join(names.get(c, c) for c in criteria)


def _target_events_by_scope(target_events: pd.DataFrame, start_seconds: float, scope: str, window: int) -> pd.DataFrame:
    if target_events.empty:
        return target_events
    if scope != "window":
        return target_events
    if pd.isna(start_seconds):
        return target_events
    return target_events[
        (target_events["event_seconds"] >= float(start_seconds))
        & (target_events["event_seconds"] <= float(start_seconds) + int(window))
    ].copy()


def _danger_flags_for_target(target_events: pd.DataFrame, target_play_pattern) -> dict:
    return {
        "danger_shot": int((target_events["type"] == "Shot").any()) if not target_events.empty else 0,
        "danger_box_entry": int(_box_entry_mask(target_events).any()) if not target_events.empty else 0,
        "danger_from_counter": int(str(target_play_pattern) == "From Counter"),
    }


def _build_transition_starts(
    events: pd.DataFrame,
    team_set: frozenset,
    pattern_set: frozenset,
    danger_criteria: tuple,
    danger_combine: str,
    danger_scope: str,
    danger_window: int,
) -> pd.DataFrame:
    ev = events[events["team"].isin(team_set)].copy()
    if ev.empty:
        return pd.DataFrame()

    meta = _possession_meta(events)
    if meta.empty:
        return pd.DataFrame()

    mask_recovery = (ev["type"] == "Ball Recovery") & (ev["ball_recovery_recovery_failure"].isna())
    mask_interception = (
        (ev["type"] == "Interception")
        & (
            ev["interception_outcome"].astype("object").isin(TRANSITION_RECOVERY_SUCCESS)
            | ev["interception_outcome"].isna()
        )
    )
    mask_tackle = (
        (ev["type"] == "Duel")
        & (ev["duel_type"] == "Tackle")
        & (ev["duel_outcome"].astype("object").isin(TACKLE_SUCCESS))
    )

    starts = ev[mask_recovery | mask_interception | mask_tackle].dropna(subset=["x", "y"]).copy()
    if starts.empty:
        return pd.DataFrame()

    rows = []
    for _, row in starts.iterrows():
        match_id = int(row["match_id"])
        period = int(row["period"])
        team = row["team"]

        if row.get("possession_team") == row.get("team"):
            transition_possession = row.get("possession")
            transition_pattern = row.get("play_pattern")
        else:
            transition_possession, transition_pattern = _next_possession_for_team(
                meta, match_id, period, float(row["event_order"]), team
            )

        if pd.isna(transition_possession):
            continue
        if pattern_set and transition_pattern not in pattern_set:
            continue

        target_events = events[
            (events["match_id"] == match_id)
            & (events["period"] == period)
            & (events["possession"] == transition_possession)
            & (events["team"] == team)
        ].copy()
        target_events = _target_events_by_scope(target_events, row.get("event_seconds", np.nan), danger_scope, danger_window)
        flags = _danger_flags_for_target(target_events, transition_pattern)
        dangerous = _combine_danger_flags(flags, danger_criteria, danger_combine)
        ends_in_goal = int(((target_events["type"] == "Shot") & (target_events["shot_outcome"] == "Goal")).any()) if not target_events.empty else 0

        rows.append({
            "match_id": match_id,
            "period": period,
            "team": team,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "start_type": row["type"],
            "transition_possession": transition_possession,
            "transition_play_pattern": transition_pattern,
            "danger_shot": flags["danger_shot"],
            "danger_box_entry": flags["danger_box_entry"],
            "danger_from_counter": flags["danger_from_counter"],
            "dangerous_action": dangerous,
            "ends_in_shot": flags["danger_shot"],
            "ends_in_goal": ends_in_goal,
        })

    return pd.DataFrame(rows)


def _build_loss_events(
    events: pd.DataFrame,
    team_set: frozenset,
    danger_criteria: tuple,
    danger_combine: str,
    danger_scope: str,
    danger_window: int,
    filter_losses_by_pattern: bool,
    pattern_set: frozenset,
) -> pd.DataFrame:
    meta = _possession_meta(events)
    if meta.empty:
        return pd.DataFrame()

    own_possessions = meta[meta["possession_team"].isin(team_set)].copy()
    if filter_losses_by_pattern and pattern_set:
        own_possessions = own_possessions[own_possessions["play_pattern"].isin(pattern_set)]
    if own_possessions.empty:
        return pd.DataFrame()

    rows = []
    for _, poss in own_possessions.iterrows():
        match_id = int(poss["match_id"])
        team = poss["possession_team"]
        period = int(poss["period"])
        possession_id = poss["possession"]

        possession_events = events[
            (events["match_id"] == match_id)
            & (events["period"] == period)
            & (events["possession"] == possession_id)
            & (events["team"] == team)
        ].copy()

        candidates = possession_events[
            possession_events["type"].astype("object").isin(LOSS_CANDIDATE_TYPES)
            & possession_events["x"].notna()
            & possession_events["y"].notna()
        ].copy()
        if candidates.empty:
            continue

        loss_event = candidates.sort_values("event_order").iloc[-1]
        next_poss = _next_opponent_possession(meta, match_id, period, float(poss["end_order"]), team)
        if next_poss is None:
            continue

        opponent = next_poss["possession_team"]
        next_events = events[
            (events["match_id"] == match_id)
            & (events["period"] == period)
            & (events["possession"] == next_poss["possession"])
            & (events["team"] == opponent)
        ].copy()

        target_events = _target_events_by_scope(next_events, loss_event.get("event_seconds", np.nan), danger_scope, danger_window)
        flags = _danger_flags_for_target(target_events, next_poss["play_pattern"])
        dangerous = _combine_danger_flags(flags, danger_criteria, danger_combine)

        rows.append({
            "match_id": match_id,
            "period": period,
            "team": team,
            "opponent": opponent,
            "possession": possession_id,
            "next_opponent_possession": next_poss["possession"],
            "play_pattern": poss["play_pattern"],
            "action_type": loss_event["type"],
            "x": float(loss_event["x"]),
            "y": float(loss_event["y"]),
            "danger_shot": flags["danger_shot"],
            "danger_box_entry": flags["danger_box_entry"],
            "danger_from_counter": flags["danger_from_counter"],
            "dangerous_loss": dangerous,
            "opponent_shot": flags["danger_shot"],
            "opponent_box_entry": flags["danger_box_entry"],
            "opponent_counter": flags["danger_from_counter"],
        })

    return pd.DataFrame(rows)


def _process_single_match(
    match_id: int,
    team_set: frozenset,
    pattern_set: frozenset,
    danger_criteria: tuple,
    danger_combine: str,
    danger_scope: str,
    danger_window: int,
    filter_losses_by_pattern: bool,
) -> dict:
    events = _fetch_events_cached(int(match_id)).copy()
    events["match_id"] = int(match_id)

    transitions = _build_transition_starts(
        events=events,
        team_set=team_set,
        pattern_set=pattern_set,
        danger_criteria=danger_criteria,
        danger_combine=danger_combine,
        danger_scope=danger_scope,
        danger_window=danger_window,
    )
    losses = _build_loss_events(
        events=events,
        team_set=team_set,
        danger_criteria=danger_criteria,
        danger_combine=danger_combine,
        danger_scope=danger_scope,
        danger_window=danger_window,
        filter_losses_by_pattern=filter_losses_by_pattern,
        pattern_set=pattern_set,
    )
    return {"transitions": transitions, "losses": losses}

# ─────────────────────────────────────────────
# HELPERS DE VISUALIZACIÓN
# ─────────────────────────────────────────────

def _make_custom_grid(figheight=8):
    from matplotlib.gridspec import GridSpec
    figwidth = figheight * 1.6
    fig = plt.figure(figsize=(figwidth, figheight), facecolor="white")
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 15, 1], width_ratios=[25, 1],
        left=0.05, right=0.95, top=0.95, bottom=0.05,
        wspace=0.02, hspace=0.05,
    )
    axs = {
        "title": fig.add_subplot(gs[0, 0]),
        "pitch": fig.add_subplot(gs[1, 0]),
        "cbar": fig.add_subplot(gs[1, 1]),
        "endnote": fig.add_subplot(gs[2, 0]),
    }
    for k in ["title", "endnote"]:
        axs[k].axis("off")
    return fig, axs


def _empty_pitch_message(message: str):
    fig, axs = _make_custom_grid(figheight=8)
    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
    pitch.draw(ax=axs["pitch"])
    axs["pitch"].text(60, 40, message, color="#94a3b8", ha="center", fontsize=14)
    return fig


def _bin_stats_transitions(data: pd.DataFrame):
    data = data.dropna(subset=["x", "y"])
    if data.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb")
    bin_rec = pitch.bin_statistic(data.x, data.y, statistic="count", bins=(12, 8))
    bin_danger = pitch.bin_statistic(data.x, data.y, values=data.dangerous_action, statistic="sum", bins=(12, 8))
    bin_shot = pitch.bin_statistic(data.x, data.y, values=data.danger_shot, statistic="sum", bins=(12, 8))
    bin_goal = pitch.bin_statistic(data.x, data.y, values=data.ends_in_goal, statistic="sum", bins=(12, 8))
    bin_pct = pitch.bin_statistic(data.x, data.y, values=data.dangerous_action, statistic="mean", bins=(12, 8))
    bin_pct["statistic"] = np.nan_to_num(bin_pct["statistic"] * 100)
    return {"rec": bin_rec, "danger": bin_danger, "shot": bin_shot, "goal": bin_goal, "pct": bin_pct}


def _bin_stats_losses(data: pd.DataFrame):
    data = data.dropna(subset=["x", "y"])
    if data.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb")
    bin_loss = pitch.bin_statistic(data.x, data.y, statistic="count", bins=(12, 8))
    bin_danger = pitch.bin_statistic(data.x, data.y, values=data.dangerous_loss, statistic="sum", bins=(12, 8))
    bin_shot = pitch.bin_statistic(data.x, data.y, values=data.danger_shot, statistic="sum", bins=(12, 8))
    bin_pct = pitch.bin_statistic(data.x, data.y, values=data.dangerous_loss, statistic="mean", bins=(12, 8))
    bin_pct["statistic"] = np.nan_to_num(bin_pct["statistic"] * 100)
    return {"loss": bin_loss, "danger": bin_danger, "shot": bin_shot, "pct": bin_pct}


def _team_quadrant_summary(transitions: pd.DataFrame, losses: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(
        set(transitions["team"].dropna().unique() if transitions is not None and not transitions.empty else [])
        | set(losses["team"].dropna().unique() if losses is not None and not losses.empty else [])
    )
    if not teams:
        return pd.DataFrame()

    summary = pd.DataFrame({"team": teams})

    if transitions is not None and not transitions.empty:
        off = transitions.groupby("team", observed=True).agg(
            offensive_actions=("dangerous_action", "count"),
            offensive_dangerous=("dangerous_action", "sum"),
        ).reset_index()
        summary = summary.merge(off, on="team", how="left")
    else:
        summary["offensive_actions"] = 0
        summary["offensive_dangerous"] = 0

    if losses is not None and not losses.empty:
        deff = losses.groupby("team", observed=True).agg(
            defensive_losses=("dangerous_loss", "count"),
            defensive_dangerous=("dangerous_loss", "sum"),
        ).reset_index()
        summary = summary.merge(deff, on="team", how="left")
    else:
        summary["defensive_losses"] = 0
        summary["defensive_dangerous"] = 0

    for col in ["offensive_actions", "offensive_dangerous", "defensive_losses", "defensive_dangerous"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

    summary["offensive_danger_rate"] = np.where(
        summary["offensive_actions"] > 0,
        summary["offensive_dangerous"] / summary["offensive_actions"],
        0,
    )
    summary["defensive_danger_rate"] = np.where(
        summary["defensive_losses"] > 0,
        summary["defensive_dangerous"] / summary["defensive_losses"],
        0,
    )
    return summary.sort_values("offensive_danger_rate", ascending=False)


def _plot_team_quadrant(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "Sin datos para comparar", ha="center", va="center", transform=ax.transAxes, color="#94a3b8")
        ax.axis("off")
        return fig
    x = summary["defensive_danger_rate"] * 100
    y = summary["offensive_danger_rate"] * 100
    sizes = np.maximum(summary["offensive_actions"].fillna(0) + summary["defensive_losses"].fillna(0), 20)
    sizes = 40 + 4 * np.sqrt(sizes)
    ax.scatter(x, y, s=sizes, alpha=0.75)
    mx, my = x.mean(), y.mean()
    ax.axvline(mx, linestyle="--", linewidth=1, alpha=0.4)
    ax.axhline(my, linestyle="--", linewidth=1, alpha=0.4)
    for _, r in summary.iterrows():
        ax.annotate(str(r["team"]), (r["defensive_danger_rate"] * 100, r["offensive_danger_rate"] * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("% pérdidas que conceden peligro")
    ax.set_ylabel("% recuperaciones que generan peligro")
    ax.set_title("Comparador transicional de equipos", fontsize=15, fontweight="bold")
    ax.text(0.02, 0.98, "Controlado / fuerte", transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#64748b")
    ax.text(0.98, 0.98, "Vertical / caótico", transform=ax.transAxes, va="top", ha="right", fontsize=9, color="#64748b")
    ax.text(0.98, 0.02, "Vulnerable", transform=ax.transAxes, va="bottom", ha="right", fontsize=9, color="#64748b")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
        ui.tags.link(href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap", rel="stylesheet"),
        ui.tags.style("""
            :root { --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; --text: #0f172a; --muted: #64748b; --primary: #1e40af; --radius: 12px; }
            * { font-family: 'DM Sans', sans-serif !important; }
            body { background: var(--bg); color: var(--text); }
            h1, h2, h3, h4, h5, h6, .card-header { font-weight: 700 !important; letter-spacing: -0.02em !important; }
            .card { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important; margin-bottom: 1.5rem; background: white; }
            .card-header { background-color: transparent !important; border-bottom: 1px solid var(--border) !important; font-weight: 600 !important; padding: 1rem 1.25rem !important; }
            .sidebar { background: white !important; border-right: 1px solid var(--border) !important; padding: 1.5rem !important; }
            .btn-primary { background: var(--primary) !important; border-color: var(--primary) !important; border-radius: 8px !important; font-weight: 600; padding: 0.6rem 1rem; }
            .kpi-value { font-weight: 800; line-height: 1; margin-bottom: 0.5rem; }
            .kpi-label { font-size: 0.75rem; font-weight: 500; color: var(--muted); text-transform: uppercase; }
            .table-container { margin-top: 10px; border-radius: 8px; overflow: auto; border: 1px solid var(--border); max-height: 500px; }
            .table { width: 100%; margin-bottom: 0; font-size: 0.85rem; background: white; }
            .table th { background-color: #f8fafc; color: #475569; font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); position: sticky; top: 0; }
            .table td { padding: 10px 12px; vertical-align: middle; border-bottom: 1px solid var(--border); }
            .table tr:hover { background-color: #f1f5f9; }
            .small-help { color: var(--muted); font-size: 0.82rem; line-height: 1.35; margin-top: 0.5rem; }
        """)
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.h2("BeyondPlay", style="font-size: 1.4rem; font-weight: 800; color: #1e40af; margin-bottom: 0.25rem;"),
                ui.p("Transiciones y pérdidas peligrosas", style="font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem;"),
            ),
            ui.output_ui("competition_selector"),
            ui.output_ui("season_selector"),
            ui.output_ui("team_selector"),
            ui.input_selectize(
                "play_patterns", "Filtrar transiciones ofensivas por patrón",
                choices=PLAY_PATTERN_CHOICES,
                selected=[],
                multiple=True,
            ),
            ui.div("Sin selección = no filtrar por patrón de juego.", class_="small-help"),
            ui.input_switch("filter_losses_by_pattern", "Filtrar pérdidas por esos patrones", False),
            ui.input_checkbox_group(
                "danger_criteria", "Criterios de peligro",
                choices=DANGER_CRITERIA_CHOICES,
                selected=["shot"],
            ),
            ui.input_select("danger_combine", "Combinación de criterios", choices=DANGER_COMBINE_CHOICES, selected="any"),
            ui.input_select("danger_scope", "Alcance del análisis", choices=DANGER_SCOPE_CHOICES, selected="possession"),
            ui.input_numeric("danger_window", "Ventana si aplica (segundos)", value=15, min=5, max=45, step=1),
            ui.div("La misma definición de peligro se aplica a transiciones ofensivas y pérdidas defensivas. No se calculan variables extra como centralidad, distancias, presión, longitud o progresión.", class_="small-help"),
            ui.div(style="height: 20px;"),
            ui.input_action_button("analyze", "Actualizar análisis", class_="btn-primary w-100"),
            width=330,
        ),
        ui.navset_underline(
            ui.nav_panel(
                "Transiciones",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Volumen de inicios de transición y acciones peligrosas"),
                        ui.output_plot("field_transition_volume", height="760px"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header("Eficacia: % de transiciones peligrosas"),
                        ui.output_plot("field_transition_efficiency", height="760px"),
                        full_screen=True,
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Pérdidas peligrosas",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Volumen de pérdidas y pérdidas peligrosas"),
                        ui.output_plot("field_loss_volume", height="760px"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header("Riesgo empírico de pérdida peligrosa"),
                        ui.output_plot("field_loss_risk", height="760px"),
                        full_screen=True,
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Cuadrante transicional",
                ui.card(
                    ui.card_header("Cuadrante transicional"),
                    ui.output_plot("team_quadrant", height="700px"),
                    full_screen=True,
                ),
            ),
            id="tabs",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Metodología"),
                ui.markdown("""
                **Definición única de peligro**: puedes elegir si peligro significa **tiro**, **entrada al área**,
                **From Counter** o combinaciones de esos criterios. La combinación puede ser OR o AND.

                **Alcance del análisis**: con *posesión completa*, se revisa toda la posesión asociada. Con
                *posesión + ventana temporal*, se revisan solo los eventos posteriores a la recuperación o pérdida
                dentro de la ventana seleccionada.

                **Transiciones ofensivas**: se consideran inicios de transición las acciones de recuperación por
                **Ball Recovery** exitosa, **Interception** exitosa y **Duel - Tackle** ganado. Después se etiqueta
                la misma posesión propia con la definición de peligro seleccionada.

                **Pérdidas peligrosas**: se identifica la última acción ofensiva propia antes de que la posesión pase
                al rival. Después se etiqueta la siguiente posesión rival con la misma definición de peligro.

                No se calcula regresión logística ni variables extra como centralidad, distancias a porterías,
                presión, longitud de acción o progresión.
                """),
                style="max-height: 600px; overflow-y: auto;",
            ),
            ui.card(
                ui.card_header("Resumen"),
                ui.output_ui("analysis_summary"),
                style="max-height: 600px; overflow-y: auto;",
            ),
            col_widths=[5, 7],
        ),
    ),
    title="BeyondPlay: Transiciones y Pérdidas Peligrosas",
    theme=shinyswatch.theme.litera,
)

# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────

def server(input, output, session):

    @reactive.calc
    def get_competitions():
        return sb.competitions()

    @output
    @render.ui
    def competition_selector():
        comps = get_competitions()
        choices = {"all": "Todas las competiciones"}
        choices.update({str(row["competition_id"]): f"{row['competition_name']}" for _, row in comps.iterrows()})
        return ui.input_selectize("competition_ids", "Competiciones", choices, multiple=True)

    def _competition_season_pairs(comps: pd.DataFrame, selected_competitions, selected_seasons) -> pd.DataFrame:
        if not selected_competitions or not selected_seasons:
            return pd.DataFrame(columns=["competition_id", "season_id"])

        relevant = comps.copy() if "all" in selected_competitions else comps[
            comps["competition_id"].isin([int(c) for c in selected_competitions])
        ].copy()

        if "all" in selected_seasons:
            return relevant[["competition_id", "season_id"]].drop_duplicates()

        explicit_pairs = []
        legacy_seasons = []
        for value in selected_seasons:
            parts = str(value).split(":", 1)
            if len(parts) == 2:
                explicit_pairs.append({"competition_id": int(parts[0]), "season_id": int(parts[1])})
            else:
                legacy_seasons.append(int(value))

        selected = pd.DataFrame(explicit_pairs) if explicit_pairs else pd.DataFrame(columns=["competition_id", "season_id"])
        if legacy_seasons:
            legacy = relevant[relevant["season_id"].isin(legacy_seasons)][["competition_id", "season_id"]]
            selected = pd.concat([selected, legacy], ignore_index=True)

        if selected.empty:
            return selected

        return selected.merge(
            relevant[["competition_id", "season_id"]].drop_duplicates(),
            on=["competition_id", "season_id"],
            how="inner",
        ).drop_duplicates()

    @output
    @render.ui
    def season_selector():
        if not input.competition_ids():
            return ui.input_selectize("season_ids", "Ligas / temporadas", {}, multiple=True)
        comps = get_competitions()
        sel = input.competition_ids()
        relevant = comps if "all" in sel else comps[comps["competition_id"].isin([int(c) for c in sel])]
        relevant = relevant.drop_duplicates(["competition_id", "season_id"]).sort_values(["competition_name", "season_name"])
        choices = {"all": "Todas las ligas / temporadas seleccionadas"}
        choices.update({
            f"{int(row['competition_id'])}:{int(row['season_id'])}": f"{row['competition_name']} — {row['season_name']}"
            for _, row in relevant.iterrows()
        })
        return ui.input_selectize("season_ids", "Ligas / temporadas", choices, multiple=True)

    @reactive.calc
    def get_matches():
        if not input.competition_ids() or not input.season_ids():
            return None
        comps = get_competitions()
        pairs = _competition_season_pairs(comps, input.competition_ids(), input.season_ids())
        if pairs.empty:
            return None

        all_matches = []
        for _, pair in pairs.iterrows():
            try:
                matches = sb.matches(competition_id=int(pair["competition_id"]), season_id=int(pair["season_id"]))
                matches["source_competition_id"] = int(pair["competition_id"])
                matches["source_season_id"] = int(pair["season_id"])
                all_matches.append(matches)
            except Exception:
                continue
        return pd.concat(all_matches).drop_duplicates("match_id") if all_matches else None

    @output
    @render.ui
    def team_selector():
        m = get_matches()
        if m is None or m.empty:
            return ui.input_selectize("team_names", "Equipos", {}, multiple=True)
        teams = sorted(list(set(m["home_team"].tolist() + m["away_team"].tolist())))
        choices = {"all": "Todos los equipos"}
        choices.update({t: t for t in teams})
        return ui.input_selectize("team_names", "Equipos", choices, multiple=True)

    @reactive.calc
    @reactive.event(input.analyze)
    async def process_data():
        m = get_matches()
        if m is None or not input.team_names():
            return None

        sel_t = list(input.team_names())
        relevant = m if "all" in sel_t else m[(m["home_team"].isin(sel_t)) | (m["away_team"].isin(sel_t))]
        if relevant.empty:
            return None

        if "all" in sel_t:
            all_teams = set(relevant["home_team"].tolist() + relevant["away_team"].tolist())
            team_set = frozenset(all_teams)
        else:
            team_set = frozenset(sel_t)

        pattern_set = frozenset(input.play_patterns() or [])
        m_ids = relevant["match_id"].tolist()
        danger_criteria = tuple(input.danger_criteria() or [])
        danger_combine = str(input.danger_combine() or "any")
        danger_scope = str(input.danger_scope() or "possession")
        danger_window = int(input.danger_window() or 15)
        filter_losses_by_pattern = bool(input.filter_losses_by_pattern())

        all_transitions = []
        all_losses = []

        with ui.Progress(min=0, max=len(m_ids)) as p:
            p.set(message=f"Procesando {len(m_ids)} partidos...")
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=8) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor,
                        functools.partial(
                            _process_single_match,
                            int(mid), team_set, pattern_set,
                            danger_criteria, danger_combine, danger_scope, danger_window,
                            filter_losses_by_pattern,
                        ),
                    )
                    for mid in m_ids
                ]
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    try:
                        res = await task
                    except Exception:
                        res = {"transitions": pd.DataFrame(), "losses": pd.DataFrame()}
                    p.set(i + 1, detail=f"Progreso: {i + 1}/{len(m_ids)}")
                    if res["transitions"] is not None and not res["transitions"].empty:
                        all_transitions.append(res["transitions"])
                    if res["losses"] is not None and not res["losses"].empty:
                        all_losses.append(res["losses"])

        transitions = pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame()
        losses = pd.concat(all_losses, ignore_index=True) if all_losses else pd.DataFrame()
        return {
            "transitions": transitions,
            "losses": losses,
            "danger_text": _danger_definition_text(danger_criteria, danger_combine),
            "danger_scope": danger_scope,
            "danger_window": danger_window,
        }

    @reactive.calc
    async def transition_stats():
        data = await process_data()
        if data is None:
            return None
        return _bin_stats_transitions(data["transitions"])

    @reactive.calc
    async def loss_stats():
        data = await process_data()
        if data is None:
            return None
        return _bin_stats_losses(data["losses"])

    @reactive.calc
    async def team_quadrant_summary():
        data = await process_data()
        if data is None:
            return pd.DataFrame()
        return _team_quadrant_summary(data["transitions"], data["losses"])

    @output
    @render.plot
    async def team_quadrant():
        summary = await team_quadrant_summary()
        return _plot_team_quadrant(summary)

    @output
    @render.plot
    async def field_transition_volume():
        data = await process_data()
        stats = await transition_stats()
        if stats is None:
            return _empty_pitch_message("Sin datos de transición")
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        pcm = pitch.heatmap(stats["rec"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7)
        cx = stats["rec"]["cx"].flatten()
        cy = stats["rec"]["cy"].flatten()
        recs = np.nan_to_num(stats["rec"]["statistic"].flatten())
        danger = np.nan_to_num(stats["danger"]["statistic"].flatten())
        for i in range(len(cx)):
            if recs[i] > 0:
                pitch.text(cx[i], cy[i], f"R:{int(recs[i])}\nD:{int(danger[i])}", ax=axs["pitch"], color="black", fontsize=8, ha="center", va="center", fontweight="bold")
        cb = plt.colorbar(pcm, cax=axs["cbar"])
        cb.set_label("Inicios de transición", size=9)
        axs["title"].text(0, 0.7, "Volumen de inicios de transición", fontsize=18, fontweight="bold", color="#0f172a")
        axs["title"].text(0, 0.2, f"R = recuperaciones/intercepciones/tackles; D = peligrosas ({danger_text})", fontsize=11, color="#64748b")
        axs["endnote"].text(0.5, 0.5, "Dirección de ataque ➜", ha="center", va="center", fontsize=10, fontstyle="italic", color="#64748b")
        return fig

    @output
    @render.plot
    async def field_transition_efficiency():
        data = await process_data()
        stats = await transition_stats()
        if stats is None:
            return _empty_pitch_message("Sin datos de transición")
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        pcm = pitch.heatmap(stats["pct"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7, vmin=0, vmax=100)
        cx = stats["pct"]["cx"].flatten()
        cy = stats["pct"]["cy"].flatten()
        pcts = np.nan_to_num(stats["pct"]["statistic"].flatten())
        goals = np.nan_to_num(stats["goal"]["statistic"].flatten())
        recs = np.nan_to_num(stats["rec"]["statistic"].flatten())
        for i in range(len(cx)):
            if recs[i] > 0:
                pitch.text(cx[i], cy[i], f"{pcts[i]:.1f}%\nG:{int(goals[i])}", ax=axs["pitch"], color="black", fontsize=8, ha="center", va="center", fontweight="bold")
        cb = plt.colorbar(pcm, cax=axs["cbar"])
        cb.set_label("% transiciones peligrosas", size=9)
        axs["title"].text(0, 0.7, "Eficacia tras recuperación", fontsize=18, fontweight="bold", color="#0f172a")
        axs["title"].text(0, 0.2, f"Porcentaje de inicios de transición peligrosos ({danger_text}); G = goles", fontsize=11, color="#64748b")
        axs["endnote"].text(0.5, 0.5, "Dirección de ataque ➜", ha="center", va="center", fontsize=10, fontstyle="italic", color="#64748b")
        return fig

    @output
    @render.plot
    async def field_loss_volume():
        data = await process_data()
        stats = await loss_stats()
        if stats is None:
            return _empty_pitch_message("Sin datos de pérdidas")
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        pcm = pitch.heatmap(stats["loss"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7)
        cx = stats["loss"]["cx"].flatten()
        cy = stats["loss"]["cy"].flatten()
        losses = np.nan_to_num(stats["loss"]["statistic"].flatten())
        danger = np.nan_to_num(stats["danger"]["statistic"].flatten())
        for i in range(len(cx)):
            if losses[i] > 0:
                pitch.text(cx[i], cy[i], f"P:{int(losses[i])}\nD:{int(danger[i])}", ax=axs["pitch"], color="black", fontsize=8, ha="center", va="center", fontweight="bold")
        cb = plt.colorbar(pcm, cax=axs["cbar"])
        cb.set_label("Pérdidas", size=9)
        axs["title"].text(0, 0.7, "Volumen de pérdidas", fontsize=18, fontweight="bold", color="#0f172a")
        axs["title"].text(0, 0.2, f"P = pérdidas; D = peligrosas ({danger_text})", fontsize=11, color="#64748b")
        axs["endnote"].text(0.5, 0.5, "Dirección de ataque ➜", ha="center", va="center", fontsize=10, fontstyle="italic", color="#64748b")
        return fig

    @output
    @render.plot
    async def field_loss_risk():
        data = await process_data()
        stats = await loss_stats()
        if stats is None:
            return _empty_pitch_message("Sin datos de pérdidas")
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        pcm = pitch.heatmap(stats["pct"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7, vmin=0, vmax=100)
        cx = stats["pct"]["cx"].flatten()
        cy = stats["pct"]["cy"].flatten()
        pcts = np.nan_to_num(stats["pct"]["statistic"].flatten())
        shots = np.nan_to_num(stats["shot"]["statistic"].flatten())
        losses = np.nan_to_num(stats["loss"]["statistic"].flatten())
        for i in range(len(cx)):
            if losses[i] > 0:
                pitch.text(cx[i], cy[i], f"{pcts[i]:.1f}%\nT:{int(shots[i])}", ax=axs["pitch"], color="black", fontsize=8, ha="center", va="center", fontweight="bold")
        cb = plt.colorbar(pcm, cax=axs["cbar"])
        cb.set_label("% pérdidas peligrosas", size=9)
        axs["title"].text(0, 0.7, "Riesgo empírico de pérdida peligrosa", fontsize=18, fontweight="bold", color="#0f172a")
        axs["title"].text(0, 0.2, f"% = pérdidas peligrosas ({danger_text}); T = tiros rivales", fontsize=11, color="#64748b")
        axs["endnote"].text(0.5, 0.5, "Dirección de ataque ➜", ha="center", va="center", fontsize=10, fontstyle="italic", color="#64748b")
        return fig

    @output
    @render.ui
    async def analysis_summary():
        data = await process_data()
        if data is None:
            return ui.p("Selecciona competición, temporada y equipos; después pulsa 'Actualizar análisis'.", style="color: var(--muted); padding: 1.5rem; text-align: center;")

        transitions = data["transitions"]
        losses = data["losses"]
        danger_text = data.get("danger_text", "peligro")
        scope_label = "posesión completa" if data.get("danger_scope") == "possession" else f"posesión + {data.get('danger_window', 15)}s"

        n_transitions = len(transitions)
        trans_danger = int(transitions["dangerous_action"].sum()) if not transitions.empty else 0
        trans_shots = int(transitions["danger_shot"].sum()) if not transitions.empty else 0
        trans_goals = int(transitions["ends_in_goal"].sum()) if not transitions.empty else 0
        trans_eff = trans_danger / n_transitions * 100 if n_transitions else 0

        n_losses = len(losses)
        danger = int(losses["dangerous_loss"].sum()) if not losses.empty else 0
        opp_shots = int(losses["danger_shot"].sum()) if not losses.empty else 0
        danger_rate = danger / n_losses * 100 if n_losses else 0

        if not losses.empty:
            losses_for_table = losses.copy()
            losses_for_table["tercio"] = pd.cut(losses_for_table["x"], bins=[0, 40, 80, 120], labels=["Defensivo", "Intermedio", "Ofensivo"])
            losses_for_table["carril"] = pd.cut(losses_for_table["y"], bins=[0, 26.66, 53.33, 80], labels=["Izquierdo", "Central", "Derecho"])
            zone_stats = losses_for_table.groupby(["tercio", "carril"], observed=True).agg(
                Perdidas=("x", "count"),
                Peligrosas=("dangerous_loss", "sum"),
                Tiros_Rivales=("danger_shot", "sum"),
            ).reset_index()
            zone_stats["% Peligro"] = (zone_stats["Peligrosas"] / zone_stats["Perdidas"] * 100).fillna(0).map("{:.1f}%".format)
        else:
            zone_stats = pd.DataFrame(columns=["tercio", "carril", "Perdidas", "Peligrosas", "Tiros_Rivales", "% Peligro"])

        return ui.div(
            ui.div(
                ui.div(ui.div(f"{n_transitions:,}", class_="kpi-value", style="color: #1e40af; font-size: 1.6rem;"), ui.div("Transiciones", class_="kpi-label"), style="flex:1; text-align:center;"),
                ui.div(ui.div(f"{trans_eff:.1f}%", class_="kpi-value", style="color: #0891b2; font-size: 1.6rem;"), ui.div("Transición peligrosa", class_="kpi-label"), style="flex:1; text-align:center;"),
                ui.div(ui.div(f"{n_losses:,}", class_="kpi-value", style="color: #be123c; font-size: 1.6rem;"), ui.div("Pérdidas", class_="kpi-label"), style="flex:1; text-align:center;"),
                ui.div(ui.div(f"{danger_rate:.1f}%", class_="kpi-value", style="color: #9f1239; font-size: 1.6rem;"), ui.div("Pérdidas peligrosas", class_="kpi-label"), style="flex:1; text-align:center;"),
                style="display: flex; justify-content: space-around; padding: 15px 0; border-bottom: 1px solid #eee; margin-bottom: 15px;"
            ),
            ui.markdown(f"""
            **Definición de peligro:** {danger_text}  
            **Alcance:** {scope_label}  
            **Transiciones peligrosas:** {trans_danger:,}  
            **Transiciones con tiro:** {trans_shots:,}  
            **Goles tras transición:** {trans_goals:,}  
            **Tiros rivales tras pérdida:** {opp_shots:,}
            """),
            ui.h6("Pérdidas por zona", style="font-weight: bold; margin-top: 10px; color: #475569;"),
            ui.div(ui.HTML(zone_stats.to_html(classes="table", index=False, escape=False)), class_="table-container"),
        )


app = App(app_ui, server)
