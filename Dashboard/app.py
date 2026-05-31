import matplotlib
matplotlib.use('Agg')

import asyncio
import functools
import io
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from statsbombpy import sb
from mplsoccer import Pitch
import shinyswatch

try:
    from shinywidgets import output_widget, render_widget
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

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
    "location", "under_pressure", "out",
    "shot_outcome", "shot_statsbomb_xg",
    "ball_recovery_recovery_failure",
    "ball_recovery_offensive",
    "block_deflection", "block_offensive", "block_save_block",
    "interception_outcome", "duel_type", "duel_outcome",
    "50_50_outcome",
    "clearance_aerial_won", "clearance_body_part",
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
    "50_50_outcome",
]

TRANSITION_RECOVERY_SUCCESS = ["Won", "Success", "Success In Play", "In Play"]
TACKLE_SUCCESS = ["Won", "Success", "Success In Play"]
OPEN_PLAY_NEXT_PATTERNS = {"Regular Play", "From Counter"}
DEFENSIVE_PRIORITY = {"Block": 1, "Interception": 2, "Duel": 3, "50/50": 4, "Clearance": 5}
ATTACKING_PRIORITY = {
    "Miscontrol": 1,
    "Dispossessed": 2,
    "Dribble": 3,
    "Pass": 4,
    "Carry": 5,
    "Ball Receipt*": 6,
}
VALID_TACKLE_OUTCOMES = {"Won", "Success", "Success In Play"}
VALID_INTERCEPTION_OUTCOMES = {"Won", "Success", "In Play", "Success In Play"}
VALID_50_50_OUTCOMES = {"Won", "Success To Team"}
INCOMPLETE_PASS_OUTCOMES = {"Incomplete", "Out", "Pass Offside", "Unknown", "Injury Clearance"}
INCOMPLETE_DRIBBLE_OUTCOMES = {"Incomplete"}
INCOMPLETE_RECEIPT_OUTCOMES = {"Incomplete"}

TEAM_COLOR_PALETTE = [
    {"bg": "#E6F1FB", "border": "#378ADD", "text": "#0C447C", "hex": "#185FA5", "cmap": "Blues"},
    {"bg": "#EAF3DE", "border": "#639922", "text": "#27500A", "hex": "#3B6D11", "cmap": "Greens"},
    {"bg": "#FAECE7", "border": "#D85A30", "text": "#712B13", "hex": "#993C1D", "cmap": "Oranges"},
    {"bg": "#FAEEDA", "border": "#BA7517", "text": "#633806", "hex": "#854F0B", "cmap": "YlOrBr"},
    {"bg": "#EEEDFE", "border": "#7F77DD", "text": "#26215C", "hex": "#534AB7", "cmap": "Purples"},
]

# ─────────────────────────────────────────────
# CACHÉ GLOBAL Y PREFETCH EN SEGUNDO PLANO
# ─────────────────────────────────────────────
_EVENTS_CACHE: dict = {}          # {match_id -> DataFrame}  persiste entre sesiones
_EVENTS_LOCK  = threading.Lock()
_BG_EXECUTOR  = ThreadPoolExecutor(max_workers=6, thread_name_prefix="bp_bg")

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


def _get_events(match_id: int) -> pd.DataFrame:
    """Obtiene y cachea eventos de un partido. Thread-safe. No usa lru_cache para evitar GIL en ThreadPoolExecutor."""
    mid = int(match_id)
    cached = _EVENTS_CACHE.get(mid)           # lectura sin lock (GIL protege dict reads en CPython)
    if cached is not None:
        return cached
    events = sb.events(match_id=mid)
    keep = [c for c in COLS_NEEDED if c in events.columns]
    prepared = _prepare_events(events[keep].copy())
    with _EVENTS_LOCK:
        _EVENTS_CACHE[mid] = prepared
    return prepared


def _prefetch_match_events(match_ids: list) -> None:
    """Descarga eventos en segundo plano sin bloquear la UI.
    Se lanza tan pronto como el usuario elige competición + temporada,
    para que cuando pulse 'Actualizar análisis' los datos ya estén en caché."""
    def _safe_fetch(mid: int):
        try:
            _get_events(mid)
        except Exception:
            pass
    for mid in match_ids:
        if int(mid) not in _EVENTS_CACHE:
            _BG_EXECUTOR.submit(_safe_fetch, int(mid))


def _possession_meta(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["match_id", "period", "possession"] if "match_id" in events.columns else ["period", "possession"]
    sort_cols = ["match_id", "period", "event_order"] if "match_id" in events.columns else ["period", "event_order"]
    out_sort_cols = ["match_id", "period", "start_order"] if "match_id" in events.columns else ["period", "start_order"]
    meta = (
        events.dropna(subset=["possession"])
        .sort_values(sort_cols)
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
        .sort_values(out_sort_cols)
        .reset_index(drop=True)
    )
    return meta


def _add_next_possession_columns(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return meta.copy()
    sort_cols = ["match_id", "period", "start_order"] if "match_id" in meta.columns else ["period", "start_order"]
    group_cols = ["match_id", "period"] if "match_id" in meta.columns else ["period"]
    pairs = meta.sort_values(sort_cols).reset_index(drop=True).copy()
    next_cols = ["possession", "possession_team", "play_pattern", "start_order", "start_seconds"]
    grouped = pairs.groupby(group_cols, observed=True, sort=False)
    for col in next_cols:
        pairs[f"next_{col}"] = grouped[col].shift(-1)
    return pairs[pairs["next_possession"].notna()].copy()


def _pair_to_possession_series(pair: pd.Series):
    current_poss = pd.Series({
        "match_id": pair.get("match_id", pd.NA),
        "period": int(pair["period"]),
        "possession": pair["possession"],
        "possession_team": pair["possession_team"],
        "play_pattern": pair["play_pattern"],
        "start_order": pair["start_order"],
        "end_order": pair.get("end_order", pd.NA),
        "start_seconds": pair["start_seconds"],
        "end_seconds": pair.get("end_seconds", pd.NA),
    })
    next_poss = pd.Series({
        "match_id": pair.get("match_id", pd.NA),
        "period": int(pair["period"]),
        "possession": pair["next_possession"],
        "possession_team": pair["next_possession_team"],
        "play_pattern": pair["next_play_pattern"],
        "start_order": pair["next_start_order"],
        "start_seconds": pair["next_start_seconds"],
    })
    return current_poss, next_poss


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


def _is_true(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "si", "sí"}
    return bool(value)


def _is_restart_or_out(previous_events: pd.DataFrame, next_possession: pd.Series) -> bool:
    if next_possession["play_pattern"] not in OPEN_PLAY_NEXT_PATTERNS:
        return True
    if previous_events.empty:
        return True
    last_event = previous_events.sort_values("event_order").iloc[-1]
    if (previous_events["type"] == "Shot").any():
        return True
    if str(last_event.get("type", "")) == "Goal Keeper":
        return True
    if _is_true(last_event.get("out", pd.NA)):
        return True
    if str(last_event.get("pass_outcome", "")) in {"Out", "Pass Offside", "Injury Clearance"}:
        return True
    return False


def _valid_defensive_event(event: pd.Series) -> bool:
    event_type = event.get("type")
    if event_type == "Duel":
        return event.get("duel_type") == "Tackle" and str(event.get("duel_outcome")) in VALID_TACKLE_OUTCOMES
    if event_type == "Interception":
        return str(event.get("interception_outcome")) in VALID_INTERCEPTION_OUTCOMES or pd.isna(event.get("interception_outcome"))
    if event_type == "50/50":
        return str(event.get("50_50_outcome")) in VALID_50_50_OUTCOMES
    return True


def _has_attacking_loss_signal(event: pd.Series) -> bool:
    event_type = event.get("type")
    if event_type in {"Miscontrol", "Dispossessed"}:
        return True
    if event_type == "Ball Receipt*":
        return str(event.get("ball_receipt_outcome")) in INCOMPLETE_RECEIPT_OUTCOMES
    if event_type == "Dribble":
        return str(event.get("dribble_outcome")) in INCOMPLETE_DRIBBLE_OUTCOMES
    if event_type == "Pass":
        return str(event.get("pass_outcome")) in INCOMPLETE_PASS_OUTCOMES
    if event_type == "Duel":
        return "Lost" in str(event.get("duel_outcome"))
    return False


def _defensive_candidates(previous_events: pd.DataFrame, transition_team: str) -> pd.DataFrame:
    candidates = previous_events[
        (previous_events["team"] == transition_team)
        & (previous_events["type"].astype("object").isin(DEFENSIVE_PRIORITY))
        & previous_events["x"].notna()
        & previous_events["y"].notna()
    ].copy()
    if candidates.empty:
        return candidates
    candidates["candidate_source"] = "defensive_event"
    candidates["priority"] = candidates["type"].map(DEFENSIVE_PRIORITY)
    candidates["valid_candidate"] = candidates.apply(_valid_defensive_event, axis=1)
    return candidates[candidates["valid_candidate"]].copy()


def _attacking_candidates(previous_events: pd.DataFrame, losing_team: str) -> pd.DataFrame:
    candidates = previous_events[
        (previous_events["team"] == losing_team)
        & (previous_events["type"].astype("object").isin(ATTACKING_PRIORITY) | (previous_events["type"] == "Duel"))
        & previous_events["x"].notna()
        & previous_events["y"].notna()
    ].copy()
    if candidates.empty:
        return candidates
    candidates["candidate_source"] = "attacking_event"
    candidates["priority"] = candidates["type"].map(ATTACKING_PRIORITY).fillna(7)
    candidates["valid_candidate"] = candidates.apply(_has_attacking_loss_signal, axis=1)
    return candidates[candidates["valid_candidate"]].copy()


def _first_next_possession_recovery(events: pd.DataFrame, next_possession: pd.Series, transition_team: str):
    mask = (
        (events["period"] == int(next_possession["period"]))
        & (events["possession"] == next_possession["possession"])
    )
    if "match_id" in events.columns and pd.notna(next_possession.get("match_id", pd.NA)):
        mask = mask & (events["match_id"] == int(next_possession["match_id"]))
    next_events = events[mask].sort_values("event_order")
    if next_events.empty:
        return None
    first_event = next_events.iloc[0].copy()
    if (
        first_event.get("type") == "Ball Recovery"
        and first_event.get("team") == transition_team
        and pd.isna(first_event.get("ball_recovery_recovery_failure"))
        and pd.notna(first_event.get("x"))
        and pd.notna(first_event.get("y"))
    ):
        return first_event
    return None


def _event_location_for_loss(event: pd.Series):
    if event.get("type") == "Pass":
        pass_end = event.get("pass_end_location", None)
        if _safe_list(pass_end):
            return float(pass_end[0]), float(pass_end[1])
    return float(event["x"]), float(event["y"])


def _mirror_x(x):
    return PITCH_LENGTH - float(x) if pd.notna(x) else np.nan


def _mirror_y(y):
    return PITCH_WIDTH - float(y) if pd.notna(y) else np.nan


def _choose_turnover_event(events: pd.DataFrame, current_possession: pd.Series, next_possession: pd.Series):
    losing_team = current_possession["possession_team"]
    transition_team = next_possession["possession_team"]
    previous_mask = (
        (events["period"] == int(current_possession["period"]))
        & (events["possession"] == current_possession["possession"])
    )
    if "match_id" in events.columns and pd.notna(current_possession.get("match_id", pd.NA)):
        previous_mask = previous_mask & (events["match_id"] == int(current_possession["match_id"]))
    previous_events = events[previous_mask].copy()

    defensive = _defensive_candidates(previous_events, transition_team)
    if not defensive.empty:
        chosen = defensive.sort_values(["priority", "event_order"]).iloc[0].copy()
        return chosen, "defensive_event", previous_events

    attacking = _attacking_candidates(previous_events, losing_team)
    recovery = _first_next_possession_recovery(events, next_possession, transition_team)
    if not attacking.empty and recovery is not None:
        recovery["candidate_source"] = "next_possession_recovery"
        recovery["priority"] = 0
        recovery["valid_candidate"] = True
        return recovery, "next_possession_recovery", previous_events

    if not attacking.empty:
        chosen = attacking.sort_values(["event_order", "priority"], ascending=[False, True]).iloc[0].copy()
        return chosen, "attacking_event", previous_events

    return None, None, previous_events


def _normalised_loss_coordinates(event: pd.Series, source: str) -> dict:
    turnover_x, turnover_y = _event_location_for_loss(event)
    if source in {"defensive_event", "next_possession_recovery"}:
        return {
            "x": _mirror_x(turnover_x),
            "y": _mirror_y(turnover_y),
            "transition_x": turnover_x,
            "transition_y": turnover_y,
            "turnover_x": turnover_x,
            "turnover_y": turnover_y,
        }
    return {
        "x": turnover_x,
        "y": turnover_y,
        "transition_x": _mirror_x(turnover_x),
        "transition_y": _mirror_y(turnover_y),
        "turnover_x": turnover_x,
        "turnover_y": turnover_y,
    }


def _transition_start_coordinates(
    events: pd.DataFrame,
    current_possession: pd.Series,
    next_possession: pd.Series,
    turnover_event: pd.Series,
    source: str,
):
    """Coordenadas de inicio de transición (perspectiva del equipo que transiciona)."""
    coords = _normalised_loss_coordinates(turnover_event, source)

    # En acciones defensivas o recuperación inicial, el turnover ya representa el inicio.
    if source in {"defensive_event", "next_possession_recovery"}:
        return (
            coords["transition_x"],
            coords["transition_y"],
            turnover_event.get("type", pd.NA),
            turnover_event.get("event_seconds", np.nan),
            "turnover_event",
        )

    # Si solo hay señal atacante de pérdida, anclamos al primer evento localizable de la posesión B.
    mask = (
        (events["period"] == int(next_possession["period"]))
        & (events["possession"] == next_possession["possession"])
        & (events["team"] == next_possession["possession_team"])
    )
    if "match_id" in events.columns and pd.notna(next_possession.get("match_id", pd.NA)):
        mask = mask & (events["match_id"] == int(next_possession["match_id"]))

    next_events = events[mask].sort_values("event_order")
    first_loc = next_events[next_events["x"].notna() & next_events["y"].notna()]
    if not first_loc.empty:
        row = first_loc.iloc[0]
        return (
            float(row["x"]),
            float(row["y"]),
            row.get("type", pd.NA),
            row.get("event_seconds", np.nan),
            "next_possession_first_event",
        )

    # Fallback: turnover espejado en perspectiva del equipo que transiciona.
    return (
        coords["transition_x"],
        coords["transition_y"],
        turnover_event.get("type", pd.NA),
        turnover_event.get("event_seconds", np.nan),
        "mirrored_turnover_event",
    )


def _build_transition_starts(
    events: pd.DataFrame,
    team_set: frozenset,
    pattern_set: frozenset,
    danger_criteria: tuple,
    danger_combine: str,
    danger_scope: str,
    danger_window: int,
) -> pd.DataFrame:
    _COLS = ["match_id","period","team","x","y","start_type","transition_possession",
             "transition_play_pattern","danger_shot","danger_box_entry","danger_from_counter",
             "dangerous_action","ends_in_shot","ends_in_goal","previous_play_pattern",
             "turnover_event_source","start_context_source"]

    meta = _possession_meta(events)
    if meta.empty:
        return pd.DataFrame(columns=_COLS)

    possession_pairs = _add_next_possession_columns(meta)
    if possession_pairs.empty:
        return pd.DataFrame(columns=_COLS)

    rows = []
    for _, pair in possession_pairs.iterrows():
        current_poss, next_poss = _pair_to_possession_series(pair)
        transition_team = next_poss["possession_team"]   # B
        losing_team = current_poss["possession_team"]    # A

        if transition_team not in team_set:
            continue
        if transition_team == losing_team:
            continue

        # Estándar: la jugada B solo puede ser open play o contraataque.
        transition_pattern = next_poss["play_pattern"]
        if transition_pattern not in OPEN_PLAY_NEXT_PATTERNS:
            continue

        # El selector de patrón en transiciones filtra la jugada A.
        previous_pattern = current_poss["play_pattern"]
        if pattern_set and previous_pattern not in pattern_set:
            continue

        turnover_event, source, previous_events = _choose_turnover_event(events, current_poss, next_poss)
        if turnover_event is None:
            continue
        if _is_restart_or_out(previous_events, next_poss):
            continue

        x, y, start_type, start_seconds, start_context = _transition_start_coordinates(
            events=events,
            current_possession=current_poss,
            next_possession=next_poss,
            turnover_event=turnover_event,
            source=source,
        )
        if pd.isna(x) or pd.isna(y):
            continue

        match_id = int(next_poss["match_id"])
        period = int(next_poss["period"])
        transition_possession = next_poss["possession"]

        target_events = events[
            (events["match_id"] == match_id)
            & (events["period"] == period)
            & (events["possession"] == transition_possession)
            & (events["team"] == transition_team)
        ].copy()
        target_events = _target_events_by_scope(
            target_events, start_seconds, danger_scope, danger_window
        )
        flags     = _danger_flags_for_target(target_events, transition_pattern)
        dangerous = _combine_danger_flags(flags, danger_criteria, danger_combine)
        ends_in_goal = int(
            ((target_events["type"] == "Shot") & (target_events["shot_outcome"] == "Goal")).any()
        ) if not target_events.empty else 0

        rows.append({
            "match_id":               match_id,
            "period":                 period,
            "team":                   transition_team,
            "x":                      float(x),
            "y":                      float(y),
            "start_type":             start_type,
            "transition_possession":  transition_possession,
            "transition_play_pattern":transition_pattern,
            "danger_shot":            flags["danger_shot"],
            "danger_box_entry":       flags["danger_box_entry"],
            "danger_from_counter":    flags["danger_from_counter"],
            "dangerous_action":       dangerous,
            "ends_in_shot":           flags["danger_shot"],
            "ends_in_goal":           ends_in_goal,
            "previous_play_pattern":  previous_pattern,
            "turnover_event_source":  source,
            "start_context_source":   start_context,
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)
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
    _COLS = ["match_id","period","team","opponent","possession","next_opponent_possession",
             "play_pattern","action_type","x","y","danger_shot","danger_box_entry",
             "danger_from_counter","dangerous_loss","opponent_shot","opponent_box_entry","opponent_counter",
             "transition_play_pattern","turnover_event_source","turnover_event_id","turnover_event_team",
             "turnover_event_order","turnover_event_seconds","turnover_x","turnover_y",
             "transition_x","transition_y","target_start_seconds"]

    meta = _possession_meta(events)
    if meta.empty:
        return pd.DataFrame(columns=_COLS)

    possession_pairs = _add_next_possession_columns(meta)
    if possession_pairs.empty:
        return pd.DataFrame(columns=_COLS)

    rows = []
    for _, pair in possession_pairs.iterrows():
        current_poss, next_poss = _pair_to_possession_series(pair)
        match_id = int(current_poss["match_id"])
        period = int(current_poss["period"])
        team = current_poss["possession_team"]
        opponent = next_poss["possession_team"]

        if team not in team_set:
            continue
        if team == opponent:
            continue
        if filter_losses_by_pattern and pattern_set and current_poss["play_pattern"] not in pattern_set:
            continue

        loss_event, source, previous_events = _choose_turnover_event(events, current_poss, next_poss)
        if loss_event is None:
            continue
        if _is_restart_or_out(previous_events, next_poss):
            continue

        coords = _normalised_loss_coordinates(loss_event, source)
        next_events = events[
            (events["match_id"] == match_id)
            & (events["period"] == period)
            & (events["possession"] == next_poss["possession"])
            & (events["team"] == opponent)
        ].copy()

        target_events = _target_events_by_scope(
            next_events, next_poss.get("start_seconds", np.nan), danger_scope, danger_window
        )
        flags     = _danger_flags_for_target(target_events, next_poss["play_pattern"])
        dangerous = _combine_danger_flags(flags, danger_criteria, danger_combine)

        rows.append({
            "match_id":                match_id,
            "period":                  period,
            "team":                    team,
            "opponent":                opponent,
            "possession":              current_poss["possession"],
            "next_opponent_possession":next_poss["possession"],
            "play_pattern":            current_poss["play_pattern"],
            "action_type":             loss_event["type"],
            "x":                       coords["x"],
            "y":                       coords["y"],
            "danger_shot":             flags["danger_shot"],
            "danger_box_entry":        flags["danger_box_entry"],
            "danger_from_counter":     flags["danger_from_counter"],
            "dangerous_loss":          dangerous,
            "opponent_shot":           flags["danger_shot"],
            "opponent_box_entry":      flags["danger_box_entry"],
            "opponent_counter":        flags["danger_from_counter"],
            "transition_play_pattern":  next_poss["play_pattern"],
            "turnover_event_source":    source,
            "turnover_event_id":        loss_event.get("id", pd.NA),
            "turnover_event_team":      loss_event.get("team", pd.NA),
            "turnover_event_order":     float(loss_event["event_order"]) if pd.notna(loss_event.get("event_order", np.nan)) else np.nan,
            "turnover_event_seconds":   float(loss_event["event_seconds"]) if pd.notna(loss_event.get("event_seconds", np.nan)) else np.nan,
            "turnover_x":               coords["turnover_x"],
            "turnover_y":               coords["turnover_y"],
            "transition_x":             coords["transition_x"],
            "transition_y":             coords["transition_y"],
            "target_start_seconds":     float(next_poss["start_seconds"]) if pd.notna(next_poss.get("start_seconds", np.nan)) else np.nan,
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)
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
    events = _get_events(int(match_id)).copy()   # usa caché global
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
    # Guard: DataFrame vacío o sin las columnas necesarias
    required = {"x", "y", "dangerous_action", "danger_shot", "ends_in_goal"}
    if data is None or data.empty or not required.issubset(data.columns):
        return None
    data = data.dropna(subset=["x", "y"])
    if data.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb")
    bin_rec    = pitch.bin_statistic(data.x, data.y, statistic="count", bins=(12, 8))
    bin_danger = pitch.bin_statistic(data.x, data.y, values=data.dangerous_action, statistic="sum", bins=(12, 8))
    bin_shot   = pitch.bin_statistic(data.x, data.y, values=data.danger_shot,      statistic="sum", bins=(12, 8))
    bin_goal   = pitch.bin_statistic(data.x, data.y, values=data.ends_in_goal,     statistic="sum", bins=(12, 8))
    bin_pct    = pitch.bin_statistic(data.x, data.y, values=data.dangerous_action, statistic="mean", bins=(12, 8))
    bin_pct["statistic"] = np.nan_to_num(bin_pct["statistic"] * 100)
    return {"rec": bin_rec, "danger": bin_danger, "shot": bin_shot, "goal": bin_goal, "pct": bin_pct}


def _bin_stats_losses(data: pd.DataFrame):
    required = {"x", "y", "dangerous_loss", "danger_shot"}
    if data is None or data.empty or not required.issubset(data.columns):
        return None
    data = data.dropna(subset=["x", "y"])
    if data.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb")
    bin_loss   = pitch.bin_statistic(data.x, data.y, statistic="count", bins=(12, 8))
    bin_danger = pitch.bin_statistic(data.x, data.y, values=data.dangerous_loss, statistic="sum", bins=(12, 8))
    bin_shot   = pitch.bin_statistic(data.x, data.y, values=data.danger_shot,    statistic="sum", bins=(12, 8))
    bin_pct    = pitch.bin_statistic(data.x, data.y, values=data.dangerous_loss, statistic="mean", bins=(12, 8))
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
    summary["total_actions"] = summary["offensive_actions"] + summary["defensive_losses"]

    # Add goals scored after transitions
    if transitions is not None and not transitions.empty and "ends_in_goal" in transitions.columns:
        goals = transitions.groupby("team", observed=True)["ends_in_goal"].sum().reset_index()
        goals.columns = ["team", "transition_goals"]
        summary = summary.merge(goals, on="team", how="left")
    else:
        summary["transition_goals"] = 0
    summary["transition_goals"] = pd.to_numeric(summary["transition_goals"], errors="coerce").fillna(0)

    return summary.sort_values("offensive_danger_rate", ascending=False)


def _plot_team_quadrant(summary: pd.DataFrame, color_by: str = "volume"):
    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "Sin datos para comparar", ha="center", va="center",
                transform=ax.transAxes, color="#94a3b8")
        ax.axis("off")
        return fig

    x = (summary["defensive_danger_rate"] * 100).values
    y = (summary["offensive_danger_rate"] * 100).values
    sizes = np.maximum(summary["total_actions"].fillna(0), 20)
    sizes = 50 + 5 * np.sqrt(sizes)

    COLOR_OPTIONS = {
        "volume":     ("total_actions",           "Volumen total de acciones",    "Blues"),
        "offensive":  ("offensive_danger_rate_pct","% Transición peligrosa",      "Greens"),
        "defensive":  ("defensive_danger_rate_pct","% Pérdidas peligrosas",       "Reds"),
        "goals":      ("transition_goals",         "Goles tras transición",        "Purples"),
    }
    summary = summary.copy()
    summary["offensive_danger_rate_pct"]  = summary["offensive_danger_rate"]  * 100
    summary["defensive_danger_rate_pct"] = summary["defensive_danger_rate"] * 100

    col_name, cbar_label, cmap_name = COLOR_OPTIONS.get(color_by, COLOR_OPTIONS["volume"])
    color_values = summary[col_name].fillna(0).values
    vmin, vmax = color_values.min(), color_values.max()
    if vmin == vmax:
        vmin, vmax = 0, max(vmax, 1)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(x, y, s=sizes, c=color_values, cmap=cmap_name, norm=norm,
                    alpha=0.82, zorder=5, edgecolors="white", linewidths=0.8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label(cbar_label, size=9, color="#475569")
    cbar.ax.tick_params(labelsize=8)

    mx, my = x.mean(), y.mean()
    ax.axvline(mx, linestyle="--", linewidth=1, alpha=0.35, color="#94a3b8")
    ax.axhline(my, linestyle="--", linewidth=1, alpha=0.35, color="#94a3b8")

    # ── Label placement with overlap avoidance ─────────────────────
    labels = [str(r["team"]) for _, r in summary.iterrows()]
    texts = [ax.text(xi, yi, lbl, fontsize=8, color="#334155",
                     ha="left", va="bottom", zorder=6)
             for xi, yi, lbl in zip(x, y, labels)]
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=x, y=y, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.6),
            expand_text=(1.15, 1.3),
            expand_points=(1.2, 1.4),
            force_text=(0.4, 0.6),
        )
    except ImportError:
        # Fallback: push labels away from their dot with a small offset
        x_range = x.max() - x.min() if x.max() != x.min() else 1
        y_range = y.max() - y.min() if y.max() != y.min() else 1
        for t, xi, yi in zip(texts, x, y):
            ox = x_range * 0.015
            oy = y_range * 0.02
            t.set_position((xi + ox, yi + oy))

    # ── Quadrant labels (outside data area, near axes) ─────────────
    pad_x = (x.max() - x.min()) * 0.02 if x.max() != x.min() else 0.1
    pad_y = (y.max() - y.min()) * 0.02 if y.max() != y.min() else 0.1
    ax.text(x.min() + pad_x, y.max() - pad_y, "Controlado / fuerte",
            va="top", ha="left", fontsize=9, color="#64748b",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.text(x.max() - pad_x, y.max() - pad_y, "Vertical / caótico",
            va="top", ha="right", fontsize=9, color="#64748b",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.text(x.min() + pad_x, y.min() + pad_y, "Sólido / ineficaz",
            va="bottom", ha="left", fontsize=9, color="#64748b",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.text(x.max() - pad_x, y.min() + pad_y, "Vulnerable",
            va="bottom", ha="right", fontsize=9, color="#64748b",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax.set_xlabel("% pérdidas que conceden peligro", fontsize=10, color="#475569")
    ax.set_ylabel("% recuperaciones que generan peligro", fontsize=10, color="#475569")
    ax.set_title("Comparador transicional de equipos", fontsize=15, fontweight="bold", color="#0f172a", pad=14)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#e2e8f0")
    ax.grid(alpha=0.15, color="#cbd5e1")
    fig.tight_layout()
    return fig


def _build_quadrant_figure(summary: pd.DataFrame, color_by: str = "volume",
                            selected_teams: list | None = None) -> "go.FigureWidget":
    """Versión interactiva del cuadrante usando Plotly FigureWidget."""
    if not HAS_PLOTLY:
        return None

    COLOR_OPTIONS = {
        "volume":    ("total_actions",             "Volumen total",           "Blues"),
        "offensive": ("offensive_danger_rate_pct",  "% Transición peligrosa",  "Greens"),
        "defensive": ("defensive_danger_rate_pct",  "% Riesgo defensivo",      "Reds"),
        "goals":     ("transition_goals",           "Goles tras transición",   "Purples"),
    }
    summary = summary.copy()
    summary["offensive_danger_rate_pct"]  = summary["offensive_danger_rate"] * 100
    summary["defensive_danger_rate_pct"] = summary["defensive_danger_rate"] * 100
    col_name, cbar_label, cscale = COLOR_OPTIONS.get(color_by, COLOR_OPTIONS["volume"])

    x = summary["defensive_danger_rate"].values * 100
    y = summary["offensive_danger_rate"].values * 100
    c = summary[col_name].fillna(0).values
    sizes = 12 + 2.5 * np.sqrt(np.maximum(summary["total_actions"].fillna(0), 4))
    teams = summary["team"].astype(str).tolist()
    selected = set(selected_teams or [])

    # Border color: highlight if selected
    borders = ["#185FA5" if t in selected else "white" for t in teams]
    border_w = [3 if t in selected else 1 for t in teams]

    fig = go.FigureWidget(data=[go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=teams,
        textposition="top right",
        textfont=dict(size=9, color="#334155"),
        marker=dict(
            size=sizes,
            color=c,
            colorscale=cscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=cbar_label, side="right", font=dict(size=10)),
                thickness=12, len=0.65, tickfont=dict(size=8),
            ),
            line=dict(color=borders, width=border_w),
            opacity=0.88,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Peligro concedido: %{x:.1f}%<br>"
            "Peligro generado: %{y:.1f}%<br>"
            f"{cbar_label}: " + "%{marker.color:.1f}"
            "<extra></extra>"
        ),
        customdata=teams,
    )])

    # Quadrant lines
    mx, my = float(x.mean()), float(y.mean())
    fig.add_hline(y=my, line_dash="dot", line_color="#94a3b8", line_width=1, opacity=0.55)
    fig.add_vline(x=mx, line_dash="dot", line_color="#94a3b8", line_width=1, opacity=0.55)

    # Labels — placed at extremes, not over data
    xr = (float(x.min()), float(x.max()))
    yr = (float(y.min()), float(y.max()))
    px_ = (xr[1] - xr[0]) * 0.02 or 0.1
    py_ = (yr[1] - yr[0]) * 0.02 or 0.1
    for tx, ty, txt, xa, ya in [
        (xr[0]+px_, yr[1]-py_, "Controlado / fuerte",  "left",  "top"),
        (xr[1]-px_, yr[1]-py_, "Vertical / caótico",   "right", "top"),
        (xr[0]+px_, yr[0]+py_, "Sólido / ineficaz",    "left",  "bottom"),
        (xr[1]-px_, yr[0]+py_, "Vulnerable",            "right", "bottom"),
    ]:
        fig.add_annotation(
            x=tx, y=ty, text=f"<i>{txt}</i>", showarrow=False,
            font=dict(size=9, color="#94a3b8"),
            xanchor=xa, yanchor=ya,
            bgcolor="rgba(255,255,255,0.75)", borderpad=3,
        )

    fig.update_layout(
        title=dict(text="Comparador transicional de equipos",
                   font=dict(size=14, color="#0f172a"), x=0.5),
        xaxis=dict(title="% pérdidas que conceden peligro",
                   gridcolor="#f1f5f9", tickfont=dict(size=9), zeroline=False),
        yaxis=dict(title="% recuperaciones que generan peligro",
                   gridcolor="#f1f5f9", tickfont=dict(size=9), zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=90, t=55, b=55),
        hoverlabel=dict(bgcolor="white", font_size=12),
        font=dict(family="DM Sans, sans-serif"),
        height=660,
        dragmode="pan",
    )
    return fig


def _team_color_map(team_list):
    """Asigna colores en orden de selección."""
    return {t: TEAM_COLOR_PALETTE[i % len(TEAM_COLOR_PALETTE)] for i, t in enumerate(team_list)}


def _scatter_transitions(ax_pitch, pitch, df, colors_map):
    """Dibuja inicios de transición como puntos coloreados por equipo."""
    for team, ci in colors_map.items():
        tdf = df[df["team"] == team] if not df.empty else df
        safe = tdf[tdf["dangerous_action"] == 0]
        dng = tdf[tdf["dangerous_action"] == 1]
        if not safe.empty:
            pitch.scatter(safe["x"], safe["y"], ax=ax_pitch, s=22, color=ci["hex"], alpha=0.35, zorder=3)
        if not dng.empty:
            pitch.scatter(dng["x"], dng["y"], ax=ax_pitch, s=60, color="#E24B4A", alpha=0.78,
                         edgecolors=ci["hex"], linewidths=1.5, zorder=4)


def _scatter_losses(ax_pitch, pitch, df, colors_map):
    """Dibuja pérdidas como puntos coloreados por equipo."""
    for team, ci in colors_map.items():
        tdf = df[df["team"] == team] if not df.empty else df
        safe = tdf[tdf["dangerous_loss"] == 0]
        dng = tdf[tdf["dangerous_loss"] == 1]
        if not safe.empty:
            pitch.scatter(safe["x"], safe["y"], ax=ax_pitch, s=22, color=ci["hex"], alpha=0.35, zorder=3)
        if not dng.empty:
            pitch.scatter(dng["x"], dng["y"], ax=ax_pitch, s=60, color="#E24B4A", alpha=0.78,
                         edgecolors=ci["hex"], linewidths=1.5, zorder=4)


def _add_scatter_legend(ax_pitch, colors_map):
    """Añade leyenda de equipos + peligroso al eje del campo."""
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=ci["hex"], label=team, alpha=0.7) for team, ci in colors_map.items()]
    handles.append(mpatches.Patch(color="#E24B4A", label="Peligrosa", alpha=0.78))
    ax_pitch.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
        ui.tags.link(href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap", rel="stylesheet"),
        ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.4.0/dist/tabler-icons.min.css"),
        ui.tags.style("""
            :root {
                --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0;
                --text: #0f172a; --muted: #64748b; --primary: #185FA5; --radius: 12px;
            }
            * { font-family: 'DM Sans', sans-serif !important; }
            body { background: var(--bg); color: var(--text); }
            h1,h2,h3,h4,h5,h6,.card-header { font-weight: 700 !important; letter-spacing: -0.02em !important; }
            .card { border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
                    box-shadow: 0 4px 6px -1px rgb(0 0 0/0.07) !important; margin-bottom: 1.5rem; background: white; }
            .card-header { background-color: transparent !important; border-bottom: 1px solid var(--border) !important;
                           font-weight: 600 !important; padding: 0.75rem 1rem !important;
                           display: flex !important; align-items: center !important; justify-content: space-between !important; }
            .sidebar { background: white !important; border-right: 1px solid var(--border) !important; padding: 1.5rem !important; }
            .btn-primary { background: var(--primary) !important; border-color: var(--primary) !important;
                           border-radius: 8px !important; font-weight: 600; padding: 0.6rem 1rem; }
            .section-label { font-size: 0.72rem; font-weight: 600; color: var(--muted);
                              text-transform: uppercase; letter-spacing: .05em; margin-bottom: 5px; }
            .kpi-value { font-weight: 800; line-height: 1; margin-bottom: 0.25rem; }
            .kpi-label { font-size: 0.72rem; font-weight: 500; color: var(--muted); text-transform: uppercase; }
            .kpi-delta { font-size: 0.72rem; font-weight: 500; margin-top: 4px; display: flex; align-items: center; gap: 3px; }
            .kpi-delta.up { color: #16a34a; } .kpi-delta.down { color: #dc2626; } .kpi-delta.neutral { color: var(--muted); }
            .table-container { margin-top: 10px; border-radius: 8px; overflow: auto; border: 1px solid var(--border); max-height: 500px; }
            .table { width: 100%; margin-bottom: 0; font-size: 0.85rem; background: white; }
            .table th { background-color: #f8fafc; color: #475569; font-weight: 600; text-align: left;
                         padding: 10px 12px; border-bottom: 2px solid var(--border); position: sticky; top: 0; }
            .table td { padding: 10px 12px; vertical-align: middle; border-bottom: 1px solid var(--border); }
            .table tr:hover { background-color: #f1f5f9; }
            .small-help { color: var(--muted); font-size: 0.82rem; line-height: 1.35; margin-top: 0.5rem; }

            /* ── Danger criteria icon buttons ── */
            .danger-group > label { font-size: 0.72rem !important; font-weight: 600 !important; color: var(--muted) !important;
                                    text-transform: uppercase !important; letter-spacing: .05em !important;
                                    margin-bottom: 6px !important; display: block !important; }
            .danger-group .shiny-options-group { display: grid !important; grid-template-columns: 1fr 1fr 1fr !important; gap: 6px !important; }
            .danger-group .checkbox { margin: 0 !important; }
            .danger-group .checkbox label { display: flex !important; flex-direction: column !important;
                align-items: center !important; justify-content: center !important; padding: 10px 4px 8px !important;
                gap: 3px !important; border: 0.5px solid var(--border) !important; border-radius: 8px !important;
                cursor: pointer !important; font-size: 11px !important; font-weight: 500 !important;
                background: var(--surface) !important; color: var(--muted) !important;
                min-height: 62px !important; width: 100% !important; margin: 0 !important; transition: all .15s !important; }
            .danger-group .checkbox label:hover { background: #f1f5f9 !important; }
            .danger-group .checkbox input[type=checkbox] { display: none !important; }
            .danger-group .checkbox:has(input:checked) label { background: #E6F1FB !important;
                border-color: #378ADD !important; color: #0C447C !important; font-weight: 600 !important; }
            .danger-group .checkbox .ti { font-size: 20px !important; display: block !important; }

            /* ── Chart mode pills ── */
            .chart-mode-pills > label { display: none !important; }
            .chart-mode-pills .shiny-options-group { display: flex !important; gap: 6px !important; }
            .chart-mode-pills .radio { margin: 0 !important; }
            .chart-mode-pills .radio label { padding: 5px 12px !important; border-radius: 20px !important;
                font-size: 12px !important; border: 0.5px solid var(--border) !important; cursor: pointer !important;
                background: var(--surface) !important; color: var(--muted) !important;
                margin: 0 !important; font-weight: 400 !important; transition: all .15s !important; }
            .chart-mode-pills .radio input[type=radio] { display: none !important; }
            .chart-mode-pills .radio:has(input:checked) label { background: #E6F1FB !important;
                border-color: #378ADD !important; color: #0C447C !important; font-weight: 600 !important; }

            /* ── Team legend bar ── */
            .legend-bar { display: flex; align-items: center; gap: 8px; padding: 7px 16px;
                           background: var(--surface); border-bottom: 1px solid var(--border); flex-wrap: wrap; min-height: 38px; }
            .legend-bar .legend-label { font-size: 11px; color: var(--muted); font-weight: 500; margin-right: 4px; }

            /* ── Download button ── */
            .dl-btn { padding: 3px 8px !important; font-size: 11px !important;
                      border: 0.5px solid var(--border) !important; background: var(--surface) !important;
                      border-radius: 6px !important; color: var(--muted) !important;
                      display: inline-flex !important; align-items: center !important; gap: 4px !important; }
            .dl-btn:hover { background: #f1f5f9 !important; color: var(--text) !important; }
            .dl-btn .ti { font-size: 13px !important; }

            /* ── Card header with action row ── */
            .chart-header { display: flex; align-items: center; justify-content: space-between; width: 100%; }
            .chart-header-title { font-size: 0.9rem; font-weight: 600; color: var(--text); }
        """)
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.h2(
                    ui.tags.i(class_="ti ti-hexagon-filled", style="color:#185FA5; margin-right:6px; font-size:1.2rem;"),
                    "BeyondPlay",
                    style="font-size:1.4rem; font-weight:800; color:#1e40af; margin-bottom:0.25rem; display:flex; align-items:center;"
                ),
                ui.p("Transiciones y pérdidas peligrosas", style="font-size:0.85rem; color:#64748b; margin-bottom:1.5rem;"),
            ),
            ui.output_ui("competition_selector"),
            ui.output_ui("season_selector"),
            ui.output_ui("team_selector"),
            ui.div(
                ui.div("Patrón de juego (transiciones)", class_="section-label"),
                ui.input_selectize(
                    "play_patterns", None,
                    choices=PLAY_PATTERN_CHOICES,
                    selected=[],
                    multiple=True,
                ),
                ui.div("Sin selección = todos los patrones.", class_="small-help"),
            ),
            ui.input_switch("filter_losses_by_pattern", "Filtrar pérdidas por esos patrones", False),
            ui.tags.hr(style="border-color: var(--border); margin: 8px 0;"),
            ui.div(
                ui.input_checkbox_group(
                    "danger_criteria", "Criterios de peligro",
                    choices={
                        "shot": ui.tags.span(ui.tags.i(class_="ti ti-arrow-right-circle"), ui.tags.br(), "Tiro"),
                        "box_entry": ui.tags.span(ui.tags.i(class_="ti ti-box"), ui.tags.br(), "Área"),
                        "from_counter": ui.tags.span(ui.tags.i(class_="ti ti-bolt"), ui.tags.br(), "Contra"),
                    },
                    selected=["shot"],
                ),
                class_="danger-group",
            ),
            ui.input_select("danger_combine", "Combinación", choices=DANGER_COMBINE_CHOICES, selected="any"),
            ui.input_select("danger_scope", "Alcance", choices=DANGER_SCOPE_CHOICES, selected="possession"),
            ui.input_numeric("danger_window", "Ventana temporal (s)", value=15, min=5, max=45, step=1),
            ui.tags.hr(style="border-color: var(--border); margin: 8px 0;"),
            ui.div(
                ui.div("Modo de visualización", class_="section-label"),
                ui.div(
                    ui.input_radio_buttons(
                        "chart_mode", None,
                        choices={"heatmap": ui.tags.span(ui.tags.i(class_="ti ti-flame", style="font-size:13px; margin-right:3px;"), "Mapa de calor"),
                                 "scatter": ui.tags.span(ui.tags.i(class_="ti ti-circles-relation", style="font-size:13px; margin-right:3px;"), "Puntos")},
                        selected="heatmap",
                        inline=True,
                    ),
                    class_="chart-mode-pills",
                ),
            ),
            ui.div(style="height: 16px;"),
            ui.input_action_button("analyze", ui.tags.span(ui.tags.i(class_="ti ti-player-play", style="margin-right:5px;"), "Actualizar análisis"), class_="btn-primary w-100"),
            width=330,
        ),
        ui.output_ui("team_legend_bar"),
        ui.navset_underline(
            ui.nav_panel(
                "Transiciones",
                ui.layout_columns(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.span("Volumen de inicios de transición", class_="chart-header-title"),
                                ui.download_button("dl_transition_volume", ui.tags.span(ui.tags.i(class_="ti ti-download"), " PNG"), class_="dl-btn"),
                                class_="chart-header",
                            )
                        ),
                        ui.output_plot("field_transition_volume", height="720px"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.span("Eficacia: % transiciones peligrosas", class_="chart-header-title"),
                                ui.download_button("dl_transition_efficiency", ui.tags.span(ui.tags.i(class_="ti ti-download"), " PNG"), class_="dl-btn"),
                                class_="chart-header",
                            )
                        ),
                        ui.output_plot("field_transition_efficiency", height="720px"),
                        full_screen=True,
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Pérdidas peligrosas",
                ui.layout_columns(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.span("Volumen de pérdidas y pérdidas peligrosas", class_="chart-header-title"),
                                ui.download_button("dl_loss_volume", ui.tags.span(ui.tags.i(class_="ti ti-download"), " PNG"), class_="dl-btn"),
                                class_="chart-header",
                            )
                        ),
                        ui.output_plot("field_loss_volume", height="720px"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header(
                            ui.div(
                                ui.span("Riesgo empírico de pérdida peligrosa", class_="chart-header-title"),
                                ui.download_button("dl_loss_risk", ui.tags.span(ui.tags.i(class_="ti ti-download"), " PNG"), class_="dl-btn"),
                                class_="chart-header",
                            )
                        ),
                        ui.output_plot("field_loss_risk", height="720px"),
                        full_screen=True,
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Cuadrante transicional",
                ui.card(
                    ui.card_header(
                        ui.div(
                            ui.span("Cuadrante transicional de equipos", class_="chart-header-title"),
                            ui.div(
                                ui.span(
                                    ui.tags.i(class_="ti ti-hand-click", style="font-size:13px; margin-right:4px;"),
                                    "Clic en un equipo para seleccionar/deseleccionar",
                                    style="font-size:11px; color:var(--muted); margin-right:12px;",
                                ),
                                ui.download_button("dl_quadrant", ui.tags.span(ui.tags.i(class_="ti ti-download"), " PNG"), class_="dl-btn"),
                                style="display:flex; align-items:center;",
                            ),
                            class_="chart-header",
                        )
                    ),
                    ui.div(
                        ui.span("Colorear por:", style="font-size:12px; font-weight:600; color:var(--muted); margin-right:10px;"),
                        ui.div(
                            ui.input_radio_buttons(
                                "quadrant_color_by", None,
                                choices={
                                    "volume":    ui.tags.span(ui.tags.i(class_="ti ti-circles", style="font-size:13px; margin-right:3px;"), "Volumen"),
                                    "offensive": ui.tags.span(ui.tags.i(class_="ti ti-trending-up", style="font-size:13px; margin-right:3px;"), "% Ataque peligroso"),
                                    "defensive": ui.tags.span(ui.tags.i(class_="ti ti-shield", style="font-size:13px; margin-right:3px;"), "% Riesgo defensivo"),
                                    "goals":     ui.tags.span(ui.tags.i(class_="ti ti-ball-football", style="font-size:13px; margin-right:3px;"), "Goles tras transición"),
                                },
                                selected="volume",
                                inline=True,
                            ),
                            class_="chart-mode-pills",
                        ),
                        style="display:flex; align-items:center; padding:8px 14px; border-bottom:1px solid var(--border); background:#fafafa;",
                    ),
                    output_widget("team_quadrant_widget") if HAS_PLOTLY else ui.output_plot("team_quadrant", height="700px"),
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
                ui.card_header("Resumen comparativo"),
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

    # ── Pre-fetch en segundo plano al cargar partidos ────────────────
    @reactive.effect
    def _trigger_prefetch():
        m = get_matches()
        if m is not None and not m.empty:
            match_ids = m["match_id"].tolist()
            # Lanza descarga background — cuando el usuario pulse "Actualizar"
            # los eventos ya estarán en caché
            _BG_EXECUTOR.submit(_prefetch_match_events, match_ids)

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
    def team_colors():
        sel_t = list(input.team_names()) if input.team_names() else []
        if not sel_t:
            return {}
        if "all" in sel_t:
            m = get_matches()
            if m is not None:
                all_teams = sorted(set(m["home_team"].tolist() + m["away_team"].tolist()))
                return {t: TEAM_COLOR_PALETTE[i % len(TEAM_COLOR_PALETTE)] for i, t in enumerate(all_teams)}
            return {}
        return {t: TEAM_COLOR_PALETTE[i % len(TEAM_COLOR_PALETTE)] for i, t in enumerate(sel_t)}

    @output
    @render.ui
    def team_legend_bar():
        colors = team_colors()
        if not colors:
            return ui.div(style="height:0;")
        chips = [ui.span("Equipos:", class_="legend-label")]
        for team, ci in colors.items():
            chips.append(
                ui.span(
                    team,
                    style=(
                        f"display:inline-flex; align-items:center; padding:3px 11px; border-radius:20px;"
                        f"font-size:12px; font-weight:600; border:1px solid {ci['border']};"
                        f"background:{ci['bg']}; color:{ci['text']};"
                    )
                )
            )
        chips.append(
            ui.span(
                ui.tags.span("●", style="color:#E24B4A; margin-right:4px; font-size:14px;"),
                "Acción peligrosa",
                style="font-size:12px; color:#64748b; font-weight:500; display:inline-flex; align-items:center;"
            )
        )
        return ui.div(*chips, class_="legend-bar")

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
        df = data["transitions"]
        # FIX: filtrar por equipo seleccionado para que cada equipo muestre sus propios datos
        sel_t = list(input.team_names()) if input.team_names() else []
        if "all" not in sel_t and not df.empty and "team" in df.columns and sel_t:
            df = df[df["team"].isin(sel_t)]
        return _bin_stats_transitions(df)

    @reactive.calc
    async def loss_stats():
        data = await process_data()
        if data is None:
            return None
        df = data["losses"]
        # FIX: filtrar por equipo seleccionado para que cada equipo muestre sus propios datos
        sel_t = list(input.team_names()) if input.team_names() else []
        if "all" not in sel_t and not df.empty and "team" in df.columns and sel_t:
            df = df[df["team"].isin(sel_t)]
        return _bin_stats_losses(df)

    @reactive.calc
    async def team_quadrant_summary():
        data = await process_data()
        if data is None:
            return pd.DataFrame()
        return _team_quadrant_summary(data["transitions"], data["losses"])

    @output
    @render.plot
    async def team_quadrant():
        """Fallback matplotlib — sólo se usa si shinywidgets no está disponible."""
        summary = await team_quadrant_summary()
        color_by = input.quadrant_color_by() if input.quadrant_color_by() else "volume"
        return _plot_team_quadrant(summary, color_by=color_by)

    if HAS_PLOTLY:
        @render_widget
        async def team_quadrant_widget():
            summary = await team_quadrant_summary()
            color_by = input.quadrant_color_by() if input.quadrant_color_by() else "volume"
            sel = list(input.team_names() or [])
            selected = [] if "all" in sel else sel

            if summary is None or summary.empty:
                empty = go.FigureWidget()
                empty.add_annotation(text="Ejecuta el análisis para ver el cuadrante",
                                     x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(size=13, color="#94a3b8"))
                empty.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                    height=660, xaxis_visible=False, yaxis_visible=False)
                return empty

            # Captura local para el callback (evita cierre sobre referencia mutable)
            summary_snap = summary.reset_index(drop=True)

            fig = _build_quadrant_figure(summary_snap, color_by=color_by, selected_teams=selected)

            def _on_click(trace, points, selector):
                if not points.point_inds:
                    return
                team = summary_snap.iloc[points.point_inds[0]]["team"]
                current = list(input.team_names() or [])
                if "all" in current:
                    current = [team]
                elif team in current:
                    current.remove(team)
                else:
                    current.append(team)
                ui.update_selectize("team_names", selected=current, session=session)

            fig.data[0].on_click(_on_click)
            return fig

    @output
    @render.plot
    async def field_transition_volume():
        data = await process_data()
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        mode = input.chart_mode()
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        if mode == "scatter":
            if data is None or data["transitions"].empty:
                return _empty_pitch_message("Sin datos de transición")
            df = data["transitions"]
            sel_t = list(input.team_names()) if input.team_names() else []
            if "all" not in sel_t and not df.empty:
                df = df[df["team"].isin(sel_t)]
            cmap = team_colors()
            _scatter_transitions(axs["pitch"], pitch, df, cmap)
            _add_scatter_legend(axs["pitch"], cmap)
            axs["cbar"].axis("off")
        else:
            stats = await transition_stats()
            if stats is None:
                return _empty_pitch_message("Sin datos de transición")
            pcm = pitch.heatmap(stats["rec"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7)
            cx = stats["rec"]["cx"].flatten()
            cy = stats["rec"]["cy"].flatten()
            recs = np.nan_to_num(stats["rec"]["statistic"].flatten())
            dng = np.nan_to_num(stats["danger"]["statistic"].flatten())
            for i in range(len(cx)):
                if recs[i] > 0:
                    pitch.text(cx[i], cy[i], f"R:{int(recs[i])}\nD:{int(dng[i])}", ax=axs["pitch"],
                               color="black", fontsize=8, ha="center", va="center", fontweight="bold")
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
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        mode = input.chart_mode()
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        if mode == "scatter":
            if data is None or data["transitions"].empty:
                return _empty_pitch_message("Sin datos de transición")
            df = data["transitions"]
            sel_t = list(input.team_names()) if input.team_names() else []
            if "all" not in sel_t and not df.empty:
                df = df[df["team"].isin(sel_t)]
            cmap = team_colors()
            _scatter_transitions(axs["pitch"], pitch, df, cmap)
            _add_scatter_legend(axs["pitch"], cmap)
            axs["cbar"].axis("off")
        else:
            stats = await transition_stats()
            if stats is None:
                return _empty_pitch_message("Sin datos de transición")
            pcm = pitch.heatmap(stats["pct"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7, vmin=0, vmax=100)
            cx = stats["pct"]["cx"].flatten()
            cy = stats["pct"]["cy"].flatten()
            pcts = np.nan_to_num(stats["pct"]["statistic"].flatten())
            goals = np.nan_to_num(stats["goal"]["statistic"].flatten())
            recs = np.nan_to_num(stats["rec"]["statistic"].flatten())
            for i in range(len(cx)):
                if recs[i] > 0:
                    pitch.text(cx[i], cy[i], f"{pcts[i]:.1f}%\nG:{int(goals[i])}", ax=axs["pitch"],
                               color="black", fontsize=8, ha="center", va="center", fontweight="bold")
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
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        mode = input.chart_mode()
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        if mode == "scatter":
            if data is None or data["losses"].empty:
                return _empty_pitch_message("Sin datos de pérdidas")
            df = data["losses"]
            sel_t = list(input.team_names()) if input.team_names() else []
            if "all" not in sel_t and not df.empty:
                df = df[df["team"].isin(sel_t)]
            cmap = team_colors()
            _scatter_losses(axs["pitch"], pitch, df, cmap)
            _add_scatter_legend(axs["pitch"], cmap)
            axs["cbar"].axis("off")
        else:
            stats = await loss_stats()
            if stats is None:
                return _empty_pitch_message("Sin datos de pérdidas")
            pcm = pitch.heatmap(stats["loss"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7)
            cx = stats["loss"]["cx"].flatten()
            cy = stats["loss"]["cy"].flatten()
            losses_v = np.nan_to_num(stats["loss"]["statistic"].flatten())
            dng = np.nan_to_num(stats["danger"]["statistic"].flatten())
            for i in range(len(cx)):
                if losses_v[i] > 0:
                    pitch.text(cx[i], cy[i], f"P:{int(losses_v[i])}\nD:{int(dng[i])}", ax=axs["pitch"],
                               color="black", fontsize=8, ha="center", va="center", fontweight="bold")
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
        danger_text = data.get("danger_text", "peligro") if data else "peligro"
        mode = input.chart_mode()
        fig, axs = _make_custom_grid(figheight=8)
        pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white", line_zorder=2)
        pitch.draw(ax=axs["pitch"])
        if mode == "scatter":
            if data is None or data["losses"].empty:
                return _empty_pitch_message("Sin datos de pérdidas")
            df = data["losses"]
            sel_t = list(input.team_names()) if input.team_names() else []
            if "all" not in sel_t and not df.empty:
                df = df[df["team"].isin(sel_t)]
            cmap = team_colors()
            _scatter_losses(axs["pitch"], pitch, df, cmap)
            _add_scatter_legend(axs["pitch"], cmap)
            axs["cbar"].axis("off")
        else:
            stats = await loss_stats()
            if stats is None:
                return _empty_pitch_message("Sin datos de pérdidas")
            pcm = pitch.heatmap(stats["pct"], ax=axs["pitch"], cmap="Reds", edgecolor="black", linewidth=0.5, alpha=0.7, vmin=0, vmax=100)
            cx = stats["pct"]["cx"].flatten()
            cy = stats["pct"]["cy"].flatten()
            pcts = np.nan_to_num(stats["pct"]["statistic"].flatten())
            shots = np.nan_to_num(stats["shot"]["statistic"].flatten())
            losses_v = np.nan_to_num(stats["loss"]["statistic"].flatten())
            for i in range(len(cx)):
                if losses_v[i] > 0:
                    pitch.text(cx[i], cy[i], f"{pcts[i]:.1f}%\nT:{int(shots[i])}", ax=axs["pitch"],
                               color="black", fontsize=8, ha="center", va="center", fontweight="bold")
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
            return ui.p("Selecciona competición, temporada y equipos; después pulsa 'Actualizar análisis'.",
                        style="color: var(--muted); padding: 1.5rem; text-align: center;")

        transitions = data["transitions"]
        losses = data["losses"]
        danger_text = data.get("danger_text", "peligro")
        scope_label = "posesión completa" if data.get("danger_scope") == "possession" else f"posesión + {data.get('danger_window', 15)}s"
        colors = team_colors()
        sel_t = [t for t in colors.keys()]

        def _team_stats(team=None):
            t = transitions[transitions["team"] == team] if team and not transitions.empty else transitions
            l = losses[losses["team"] == team] if team and not losses.empty else losses
            nt = len(t)
            nl = len(l)
            td = int(t["dangerous_action"].sum()) if not t.empty else 0
            ld = int(l["dangerous_loss"].sum()) if not l.empty else 0
            tg = int(t["ends_in_goal"].sum()) if not t.empty else 0
            ts = int(t["danger_shot"].sum()) if not t.empty else 0
            return {
                "n_t": nt, "n_l": nl,
                "t_eff": td / nt * 100 if nt else 0,
                "l_rate": ld / nl * 100 if nl else 0,
                "goals": tg, "shots": ts,
            }

        def _delta_arrow(val, better="up"):
            if abs(val) < 0.01:
                return ui.span("—", class_="kpi-delta neutral")
            up = val > 0
            arrow = "▲" if up else "▼"
            cls = "up" if (up and better == "up") or (not up and better == "down") else "down"
            return ui.span(f"{arrow} {abs(val):.1f}", class_=f"kpi-delta {cls}")

        def _kpi_card(val_str, label, color, delta_el=None):
            return ui.div(
                ui.div(val_str, class_="kpi-value", style=f"color:{color}; font-size:1.5rem;"),
                ui.div(label, class_="kpi-label"),
                delta_el or ui.div(),
                style="flex:1; text-align:center; padding: 8px 4px;",
            )

        # Build per-team KPI rows when exactly 2 teams
        kpi_rows = []
        if len(sel_t) == 2:
            t_a, t_b = sel_t[0], sel_t[1]
            s_a, s_b = _team_stats(t_a), _team_stats(t_b)
            ci_a, ci_b = colors[t_a], colors[t_b]
            for team, s, ci in [(t_a, s_a, ci_a), (t_b, s_b, ci_b)]:
                delta_t = s["t_eff"] - (s_b["t_eff"] if team == t_a else s_a["t_eff"])
                delta_l = s["l_rate"] - (s_b["l_rate"] if team == t_a else s_a["l_rate"])
                kpi_rows.append(
                    ui.div(
                        ui.div(
                            ui.span(team, style=(
                                f"display:inline-block; padding:2px 10px; border-radius:20px; font-size:12px;"
                                f"font-weight:600; border:1px solid {ci['border']}; background:{ci['bg']}; color:{ci['text']};"
                            )),
                            style="margin-bottom:8px;",
                        ),
                        ui.div(
                            _kpi_card(f"{s['n_t']:,}", "Transiciones", ci["hex"]),
                            _kpi_card(f"{s['t_eff']:.1f}%", "% peligrosas", "#0891b2", _delta_arrow(delta_t, "up")),
                            _kpi_card(f"{s['n_l']:,}", "Pérdidas", "#be123c"),
                            _kpi_card(f"{s['l_rate']:.1f}%", "% peligrosas", "#9f1239", _delta_arrow(-delta_l, "down")),
                            style="display:flex; justify-content:space-around;",
                        ),
                        style="padding:12px 0; border-bottom:1px solid #eee;",
                    )
                )
        else:
            s = _team_stats()
            kpi_rows.append(
                ui.div(
                    ui.div(f"{s['n_t']:,}", class_="kpi-value", style="color:#1e40af; font-size:1.6rem;"),
                    ui.div("Transiciones", class_="kpi-label"), style="flex:1; text-align:center;"
                )
            )

        # Zone table (all selected teams combined)
        if not losses.empty:
            lft = losses.copy()
            lft["tercio"] = pd.cut(lft["x"], bins=[0, 40, 80, 120], labels=["Defensivo", "Intermedio", "Ofensivo"])
            lft["carril"] = pd.cut(lft["y"], bins=[0, 26.66, 53.33, 80], labels=["Izquierdo", "Central", "Derecho"])
            zone_stats = lft.groupby(["tercio", "carril"], observed=True).agg(
                Perdidas=("x", "count"),
                Peligrosas=("dangerous_loss", "sum"),
                Tiros_Rivales=("danger_shot", "sum"),
            ).reset_index()
            zone_stats["% Peligro"] = (zone_stats["Peligrosas"] / zone_stats["Perdidas"] * 100).fillna(0).map("{:.1f}%".format)
        else:
            zone_stats = pd.DataFrame(columns=["tercio", "carril", "Perdidas", "Peligrosas", "Tiros_Rivales", "% Peligro"])

        return ui.div(
            *kpi_rows,
            ui.div(
                ui.markdown(f"**Peligro:** {danger_text} · **Alcance:** {scope_label}"),
                style="font-size:0.82rem; color:var(--muted); padding: 8px 0 4px;"
            ),
            ui.h6("Pérdidas por zona", style="font-weight:bold; margin-top:10px; color:#475569;"),
            ui.div(ui.HTML(zone_stats.to_html(classes="table", index=False, escape=False)), class_="table-container"),
        )

    # ── Download handlers ──────────────────────────────────────────

    async def _fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @render.download(filename=lambda: "transition_volume.png")
    async def dl_transition_volume():
        fig = await field_transition_volume()
        if fig is None:
            yield b""
            return
        yield await _fig_to_bytes(fig)

    @render.download(filename=lambda: "transition_efficiency.png")
    async def dl_transition_efficiency():
        fig = await field_transition_efficiency()
        if fig is None:
            yield b""
            return
        yield await _fig_to_bytes(fig)

    @render.download(filename=lambda: "loss_volume.png")
    async def dl_loss_volume():
        fig = await field_loss_volume()
        if fig is None:
            yield b""
            return
        yield await _fig_to_bytes(fig)

    @render.download(filename=lambda: "loss_risk.png")
    async def dl_loss_risk():
        fig = await field_loss_risk()
        if fig is None:
            yield b""
            return
        yield await _fig_to_bytes(fig)

    @render.download(filename=lambda: "quadrant.png")
    async def dl_quadrant():
        fig = await team_quadrant()
        if fig is None:
            yield b""
            return
        yield await _fig_to_bytes(fig)


app = App(app_ui, server)
