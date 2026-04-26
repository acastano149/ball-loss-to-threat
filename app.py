import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from shiny import reactive, render
from shiny.express import input, ui
from statsbombpy import sb
from mplsoccer import Pitch
import shinyswatch

# Configuración de página con tema Litera
ui.page_opts(
    title="StatsBomb: Análisis de Recuperaciones en Contraataque", 
    fillable=True,
    theme=shinyswatch.theme.litera
)

# ─────────────────────────────────────────────
# CAPA DE CACHÉ
# ─────────────────────────────────────────────

# Columnas mínimas necesarias para el análisis
COLS_NEEDED = [
    'type', 'play_pattern', 'possession', 'team', 'location', 'shot_outcome',
    'ball_recovery_recovery_failure', 'interception_outcome', 'duel_type', 'duel_outcome'
]

@functools.lru_cache(maxsize=256)
def _fetch_events_cached(match_id: int) -> pd.DataFrame:
    events = sb.events(match_id=match_id)
    cols = [c for c in COLS_NEEDED if c in events.columns]
    events = events[cols].copy()
    for col in ['type', 'play_pattern', 'team', 'shot_outcome']:
        if col in events.columns:
            events[col] = events[col].astype('category')
    return events


def _process_single_match(match_id: int, team_set: frozenset, pattern_set: frozenset) -> pd.DataFrame | None:
    events = _fetch_events_cached(int(match_id))
    events = events[events['team'].isin(team_set)]

    # Identificar tiros y goles en los patrones seleccionados
    shots = events[
        (events['type'] == 'Shot') &
        (events['play_pattern'].isin(pattern_set))
    ]
    posessions_with_shot = frozenset(shots['possession'].unique())

    # Goles
    goals = shots[shots['shot_outcome'] == 'Goal']
    posessions_with_goal = frozenset(goals['possession'].unique())

    mask_rec = (events['type'] == 'Ball Recovery') & (events['ball_recovery_recovery_failure'].isna())

    success_interception = ['Won', 'Success', 'In Play']
    mask_int = (events['type'] == 'Interception') & (
            (events['interception_outcome'].isin(success_interception)) | (events['interception_outcome'].isna())
    )

    mask_tackle = (events['type'] == 'Duel') & (events['duel_type'] == 'Tackle') & (
        events['duel_outcome'].isin(['Won', 'Success'])
    )

    recoveries = events[mask_rec | mask_int | mask_tackle].copy()
    
    if recoveries.empty:
        return None

    recoveries['ends_in_shot'] = recoveries['possession'].isin(posessions_with_shot).astype('int8')
    recoveries['ends_in_goal'] = recoveries['possession'].isin(posessions_with_goal).astype('int8')
    recoveries['x'] = recoveries['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    recoveries['y'] = recoveries['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    return recoveries[['x', 'y', 'ends_in_shot', 'ends_in_goal', 'team']]


# ─────────────────────────────────────────────
# CARGA INICIAL DE COMPETICIONES
# ─────────────────────────────────────────────

@reactive.calc
def get_competitions():
    return sb.competitions()

# ─────────────────────────────────────────────
# UI SIDEBAR
# ─────────────────────────────────────────────

with ui.sidebar(width=350, bg="#f8f9fa"):
    ui.h4("Configuración", style="color: #2c3e50; font-weight: bold;")
    ui.hr()
    
    @render.ui
    def competition_selector():
        comps = get_competitions()
        choices = {"all": "--- TODAS ---"}
        choices.update({str(row['competition_id']): f"{row['competition_name']} ({row['country_name']})" 
                   for _, row in comps.iterrows()})
        return ui.input_selectize("competition_ids", "Competiciones", choices, multiple=True)

    @render.ui
    def season_selector():
        if not input.competition_ids():
            return ui.input_selectize("season_ids", "Temporadas", {}, multiple=True)
        
        comps = get_competitions()
        selected_comps_raw = input.competition_ids()
        
        if "all" in selected_comps_raw:
            relevant_seasons = comps
        else:
            selected_comps = [int(cid) for cid in selected_comps_raw]
            relevant_seasons = comps[comps['competition_id'].isin(selected_comps)]
            
        choices = {"all": "--- TODAS ---"}
        choices.update({str(row['season_id']): row['season_name'] for _, row in relevant_seasons.iterrows()})
        return ui.input_selectize("season_ids", "Temporadas", choices, multiple=True)

    @render.ui
    def team_selector():
        matches = get_matches()
        if matches is None or matches.empty:
            return ui.input_selectize("team_names", "Equipos", {}, multiple=True)
        
        teams = sorted(list(set(matches['home_team'].tolist() + matches['away_team'].tolist())))
        choices = {"all": "--- TODOS LOS EQUIPOS ---"}
        choices.update({t: t for t in teams})
        return ui.input_selectize("team_names", "Equipos", choices, multiple=True)

    ui.input_selectize(
        "play_patterns", "Patrones de Juego",
        choices={
            "Regular Play": "Juego Regular",
            "From Counter": "Contraataque",
            "From Keeper": "Desde Portero",
            "From Corner": "Córner",
            "From Free Kick": "Falta",
            "From Throw In": "Saque de Banda",
            "From Goal Kick": "Saque de Puerta",
            "From Kick Off": "Saque de Centro"
        },
        selected=["From Counter"],
        multiple=True
    )
    ui.input_switch("use_360", "Enriquecer con datos 360", False)
    ui.input_action_button("analyze", "Analizar Datos", class_="btn-primary w-100", style="margin-top: 10px;")

# ─────────────────────────────────────────────
# LÓGICA DE DATOS
# ─────────────────────────────────────────────

@reactive.calc
def get_matches():
    if not input.competition_ids() or not input.season_ids():
        return None

    selected_comps = input.competition_ids()
    selected_seasons = input.season_ids()
    
    comp_ids = (
        [int(cid) for cid in get_competitions()['competition_id'].unique()]
        if "all" in selected_comps
        else [int(cid) for cid in selected_comps]
    )
        
    all_matches = []
    for comp_id in comp_ids:
        if "all" in selected_seasons:
            comps = get_competitions()
            season_ids = comps[comps['competition_id'] == comp_id]['season_id'].unique()
        else:
            season_ids = [int(sid) for sid in selected_seasons]
            
        for season_id in season_ids:
            try:
                m = sb.matches(competition_id=int(comp_id), season_id=int(season_id))
                all_matches.append(m)
            except:
                continue
    
    if not all_matches:
        return None
    return pd.concat(all_matches).drop_duplicates('match_id')


@reactive.calc
@reactive.event(input.analyze)
def process_data():
    matches = get_matches()
    if matches is None or not input.team_names():
        return None
    
    team_selection = list(input.team_names())
    
    if "all" in team_selection:
        relevant_matches = matches
        team_list = sorted(list(set(matches['home_team'].tolist() + matches['away_team'].tolist())))
    else:
        team_list = team_selection
        relevant_matches = matches[
            (matches['home_team'].isin(team_list)) |
            (matches['away_team'].isin(team_list))
        ]
    
    if relevant_matches.empty:
        return None
    
    team_set = frozenset(team_list)
    pattern_set = frozenset(input.play_patterns())
    match_ids = relevant_matches['match_id'].tolist()
    all_recoveries = []
    
    with ui.Progress(min=0, max=len(match_ids)) as p:
        p.set(message=f"Procesando {len(match_ids)} partidos en paralelo...")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_process_single_match, mid, team_set, pattern_set): mid
                for mid in match_ids
            }
            for i, future in enumerate(as_completed(futures)):
                p.set(i + 1, detail=f"Completados: {i + 1} / {len(match_ids)}")
                result = future.result()
                if result is not None:
                    all_recoveries.append(result)

    if not all_recoveries:
        return None
    return pd.concat(all_recoveries, ignore_index=True)


@reactive.calc
def get_heatmap_stats():
    data = process_data()
    if data is None or data.empty:
        return None
    
    data = data.dropna(subset=['x', 'y'])
    pitch = Pitch(pitch_type='statsbomb')
    
    # 1. Recuperaciones
    bin_rec = pitch.bin_statistic(data.x, data.y, statistic='count', bins=(12, 8))
    
    # 2. Tiros (suma de booleanos)
    bin_shot = pitch.bin_statistic(data.x, data.y, values=data.ends_in_shot, statistic='sum', bins=(12, 8))
    
    # 3. Goles (suma de booleanos)
    bin_goal = pitch.bin_statistic(data.x, data.y, values=data.ends_in_goal, statistic='sum', bins=(12, 8))
    
    # 4. Porcentaje (media de 0/1)
    bin_pct = pitch.bin_statistic(data.x, data.y, values=data.ends_in_shot, statistic='mean', bins=(12, 8))
    bin_pct['statistic'] = bin_pct['statistic'] * 100
    
    return {
        'rec': bin_rec,
        'shot': bin_shot,
        'goal': bin_goal,
        'pct': bin_pct
    }

# ─────────────────────────────────────────────
# ÁREA PRINCIPAL
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# HELPER: Figura compatible con Shiny (usa GridSpec)
# ─────────────────────────────────────────────

def _make_pitch_figure(figheight=8):
    """
    Crea una figura con GridSpec para que TODOS los ejes tengan SubplotSpec válido.
    Shiny requiere esto para poder mapear coordenadas de la figura.
    
    Layout (columnas, proporción ancho):
      col 0: campo (ancho 20) | col 1: colorbar (ancho 1)
    Layout (filas, proporción alto):
      fila 0: título (alto 1) | fila 1: campo (alto 14) | fila 2: nota (alto 1)
    
    Para ajustar el layout manualmente:
      - figheight: alto total en pulgadas
      - width_ratios: [ancho_campo, ancho_colorbar]
      - height_ratios: [alto_titulo, alto_campo, alto_nota]
    """
    from matplotlib.gridspec import GridSpec

    # ── Parámetros de layout (modifica aquí) ──────────────────────
    width_ratios  = [20, 1]   # campo vs colorbar
    height_ratios = [1, 14, 1] # titulo, campo, nota
    fig_aspect    = 68 / 105   # proporción del campo StatsBomb
    # ──────────────────────────────────────────────────────────────

    field_w = (figheight * sum(height_ratios) / height_ratios[1]) * (width_ratios[0] / sum(width_ratios))
    figwidth = figheight / fig_aspect * (width_ratios[0] / (width_ratios[0]))  # estimación
    figwidth = figheight * 1.85  # relación campo ancho/alto + espacio colorbar

    fig = plt.figure(figsize=(figwidth, figheight), facecolor='#1a2732')
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.05, wspace=0.02,
        left=0.01, right=0.99, top=0.97, bottom=0.04
    )

    ax_title   = fig.add_subplot(gs[0, 0])   # fila 0, col 0
    ax_pitch   = fig.add_subplot(gs[1, 0])   # fila 1, col 0 — CAMPO PRINCIPAL
    ax_cbar    = fig.add_subplot(gs[1, 1])   # fila 1, col 1 — COLORBAR
    ax_endnote = fig.add_subplot(gs[2, 0])   # fila 2, col 0

    # Limpiar ejes auxiliares
    for ax in [ax_title, ax_endnote]:
        ax.axis('off')
        ax.set_facecolor('#1a2732')

    ax_cbar.set_facecolor('#1a2732')
    ax_pitch.set_facecolor('#22312b')

    return fig, ax_pitch, ax_cbar, ax_title, ax_endnote


# ─────────────────────────────────────────────
# 1. SECCIÓN SUPERIOR: Campo
# ─────────────────────────────────────────────
with ui.navset_tab(id="tabs"):
    with ui.nav_panel("Campo 1: Volumen"):
        with ui.card(height="850px"):
            ui.card_header("SECCIÓN CAMPO — Volumen de Recuperaciones y Tiros", style="font-weight: bold;")
            @render.plot
            def field_volumen():
                stats = get_heatmap_stats()

                # figheight controla el alto total en pulgadas
                fig, ax, ax_cbar, ax_title, ax_endnote = _make_pitch_figure(figheight=8)

                pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                              line_color='#c7d5cc', goal_type='box')
                pitch.draw(ax=ax)

                if stats is None:
                    ax.text(60, 40, "Sin datos analizados", color='white', ha='center', fontsize=13)
                    return fig

                pcm = pitch.heatmap(stats['rec'], ax=ax, cmap='Blues',
                                    edgecolors='#22312b', lw=0.6, alpha=0.85)

                cx = stats['rec']['cx'].flatten()
                cy = stats['rec']['cy'].flatten()
                recs = stats['rec']['statistic'].flatten()
                shots = stats['shot']['statistic'].flatten()

                for i in range(len(cx)):
                    if recs[i] > 0:
                        pitch.text(cx[i], cy[i], f"R:{int(recs[i])}\nT:{int(shots[i])}",
                                   ax=ax, color='white', fontsize=8.5,
                                   ha='center', va='center', fontweight='bold')

                cb = fig.colorbar(pcm, cax=ax_cbar)
                cb.ax.tick_params(colors='white', labelsize=9)
                cb.set_label('Nº Recuperaciones', color='white', size=10)

                ax_title.text(0.5, 0.4, "Volumen de Recuperaciones (R) y Tiros generados (T)",
                              color='white', ha='center', va='center', fontsize=13, fontweight='bold',
                              transform=ax_title.transAxes)

                ax_endnote.text(0.5, 0.6, "▶  DIRECCIÓN DE ATAQUE  (portería rival →)",
                                color='#c7d5cc', ha='center', va='center', fontsize=10, fontstyle='italic',
                                transform=ax_endnote.transAxes)
                return fig

    with ui.nav_panel("Campo 2: Eficacia y Goles"):
        with ui.card(height="850px"):
            ui.card_header("SECCIÓN CAMPO — Eficacia de Recuperaciones y Goles", style="font-weight: bold;")
            @render.plot
            def field_eficacia():
                stats = get_heatmap_stats()

                fig, ax, ax_cbar, ax_title, ax_endnote = _make_pitch_figure(figheight=8)

                pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                              line_color='#c7d5cc', goal_type='box')
                pitch.draw(ax=ax)

                if stats is None:
                    ax.text(60, 40, "Sin datos analizados", color='white', ha='center', fontsize=13)
                    return fig

                pcm = pitch.heatmap(stats['pct'], ax=ax, cmap='YlOrRd',
                                    edgecolors='#22312b', lw=0.6, alpha=0.85)

                cx = stats['pct']['cx'].flatten()
                cy = stats['pct']['cy'].flatten()
                pcts = stats['pct']['statistic'].flatten()
                goals = stats['goal']['statistic'].flatten()
                recs = stats['rec']['statistic'].flatten()
                max_pct = pcts.max() if pcts.max() > 0 else 1

                for i in range(len(cx)):
                    if recs[i] > 0:
                        g_rel = goals[i] / recs[i] * 100
                        txt = f"{pcts[i]:.1f}%\nG:{int(goals[i])} ({g_rel:.1f}%)"
                        txt_color = 'white' if pcts[i] < (max_pct * 0.5) else 'black'
                        pitch.text(cx[i], cy[i], txt, ax=ax,
                                   color=txt_color, fontsize=8.5,
                                   ha='center', va='center', fontweight='bold')

                cb = fig.colorbar(pcm, cax=ax_cbar)
                cb.ax.tick_params(colors='white', labelsize=9)
                cb.set_label('% Eficacia (Tiro)', color='white', size=10)

                ax_title.text(0.5, 0.4, "Eficacia de Transición (%) y Conversión a Goles (G)",
                              color='white', ha='center', va='center', fontsize=13, fontweight='bold',
                              transform=ax_title.transAxes)

                ax_endnote.text(0.5, 0.6, "▶  DIRECCIÓN DE ATAQUE  (portería rival →)",
                                color='#c7d5cc', ha='center', va='center', fontsize=10, fontstyle='italic',
                                transform=ax_endnote.transAxes)
                return fig

# 2. SECCIONES INFERIORES: Metodología y Resultados
# col_widths: suma 12. Ej: [5,7], [6,6], [4,8]. Para el alto usa max-height en px.
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card(style="max-height: 350px; overflow-y: auto;"):
        ui.card_header("SECCIÓN METODOLOGÍA", style="font-weight: bold; background-color: #f8f9fa;")
        ui.markdown("""
        **Origen de datos**: StatsBomb Open Data.
        
        **Campos de Análisis**:
        *   **Campo 1 (Volumen)**: Número total de recuperaciones (R) y tiros generados (T). El color indica densidad.
        *   **Campo 2 (Eficacia)**: % de recuperaciones que acaban en tiro, goles (G) y su porcentaje relativo.
        
        **Lógica**: Recuperaciones cuya posesión termina en tiro/gol bajo los **patrones de juego seleccionados**.
        """)

    with ui.card(style="max-height: 350px; overflow-y: auto;"):
        ui.card_header("SECCIÓN RESULTADOS", style="font-weight: bold; background-color: #f8f9fa;")
        @render.ui
        def analysis_summary():
            data = process_data()
            if data is None or data.empty:
                return ui.p("Sin datos analizados.")
            
            total = len(data)
            shots = int(data['ends_in_shot'].sum())
            goals = int(data['ends_in_goal'].sum())
            pct_shot = (shots / total) * 100
            pct_goal = (goals / total) * 100
            
            return ui.div(
                ui.div(
                    ui.h3(f"{total:,}", style="margin:0; color:#3498db;"),
                    ui.p("Recuperaciones totales", style="margin:0; font-size:0.85em; color:#7f8c8d;"),
                    style="margin-bottom: 10px;"
                ),
                ui.div(
                    ui.h3(f"{pct_shot:.2f}%", style="margin:0; color:#e67e22;"),
                    ui.p("Eficacia (Transición a Tiro)", style="margin:0; font-size:0.85em; color:#7f8c8d;"),
                    style="margin-bottom: 10px;"
                ),
                ui.div(
                    ui.h3(f"{goals:,} Goles", style="margin:0; color:#e74c3c;"),
                    ui.p(f"Conversión total ({pct_goal:.2f}% de rec.)", style="margin:0; font-size:0.85em; color:#7f8c8d;"),
                    style="margin-bottom: 10px;"
                )
            )
