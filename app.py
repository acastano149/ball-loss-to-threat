import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from shiny import reactive, render
from shiny.express import input, ui
from statsbombpy import sb
from mplsoccer import Pitch
import shinyswatch

# Configuración de página
ui.page_opts(
    title="Soccer Intelligence: Recuperaciones", 
    fillable=True,
    theme=shinyswatch.theme.litera
)

ui.head_content(
    ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
    ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
    ui.tags.link(href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Raleway:wght@700;800&display=swap", rel="stylesheet"),
    ui.tags.style("""
        :root {
          --bg: #f8fafc;
          --surface: #ffffff;
          --border: #e2e8f0;
          --text: #0f172a;
          --muted: #64748b;
          --primary: #1e40af;
          --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
          --radius: 12px;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4, .card-header {
          font-family: 'Raleway', sans-serif;
          letter-spacing: -0.01em;
        }

        .card {
          border: 1px solid var(--border) !important;
          border-radius: var(--radius) !important;
          box-shadow: var(--shadow) !important;
          margin-bottom: 1.5rem;
        }

        .card-header {
          background-color: transparent !important;
          border-bottom: 1px solid var(--border) !important;
          font-weight: 600 !important;
          color: var(--text) !important;
          padding: 1rem 1.25rem !important;
        }

        .sidebar {
          border-right: 1px solid var(--border) !important;
          padding: 1.5rem !important;
        }

        .section-title {
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 1.25rem;
          color: var(--text);
        }

        .btn-primary {
          background: var(--primary) !important;
          border-color: var(--primary) !important;
          border-radius: 8px !important;
          font-weight: 600;
          padding: 0.6rem 1rem;
          transition: all 0.2s;
        }

        .btn-primary:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(30, 64, 175, 0.25);
        }

        .nav-underline .nav-link.active {
          color: var(--primary) !important;
          font-weight: 600;
        }

        .kpi-card {
          text-align: center;
          padding: 1.5rem !important;
        }

        .kpi-value {
          font-size: 2.25rem;
          font-weight: 800;
          line-height: 1;
          margin-bottom: 0.5rem;
          letter-spacing: -0.02em;
        }

        .kpi-label {
          font-size: 0.875rem;
          font-weight: 500;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.025em;
        }

        .control-label {
          font-size: 0.8rem !important;
          font-weight: 600 !important;
          color: var(--muted) !important;
          margin-bottom: 0.4rem !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    """)
)

# ─────────────────────────────────────────────
# CAPA DE CACHÉ
# ─────────────────────────────────────────────

# Columnas mínimas necesarias para el análisis
COLS_NEEDED = ['type', 'play_pattern', 'possession', 'team', 'location', 'shot_outcome']

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

    recoveries = events[events['type'] == 'Ball Recovery'].copy()
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

with ui.sidebar(width=300, bg="white"):
    ui.div(
        ui.h2("Soccer Intelligence", style="font-size: 1.4rem; font-weight: 800; color: #1e40af; margin-bottom: 0.25rem;"),
        ui.p("Análisis de recuperaciones", style="font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem;"),
        class_="sidebar-header"
    )
    
    with ui.div(class_="filter-container"):
        @render.ui
        def competition_selector():
            comps = get_competitions()
            choices = {"all": "Todas las competiciones"}
            choices.update({str(row['competition_id']): f"{row['competition_name']}" 
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
                
            choices = {"all": "Todas las temporadas"}
            choices.update({str(row['season_id']): row['season_name'] for _, row in relevant_seasons.iterrows()})
            return ui.input_selectize("season_ids", "Temporadas", choices, multiple=True)

        @render.ui
        def team_selector():
            matches = get_matches()
            if matches is None or matches.empty:
                return ui.input_selectize("team_names", "Equipos", {}, multiple=True)
            
            teams = sorted(list(set(matches['home_team'].tolist() + matches['away_team'].tolist())))
            choices = {"all": "Todos los equipos"}
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
        
        ui.input_switch("use_360", "Datos 360 enriquecidos", False)
        
    ui.div(style="height: 20px;")
    ui.input_action_button("analyze", "Actualizar análisis", class_="btn-primary w-100")

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
async def process_data():
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
    pattern_set = frozenset(input.play_patterns() or [])
    match_ids = relevant_matches['match_id'].tolist()
    all_recoveries = []
    
    with ui.Progress(min=0, max=len(match_ids)) as p:
        p.set(message=f"Procesando {len(match_ids)} partidos en paralelo...")
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [
                loop.run_in_executor(executor, _process_single_match, mid, team_set, pattern_set)
                for mid in match_ids
            ]
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                p.set(i + 1, detail=f"Completados: {i + 1} / {len(match_ids)}")
                if result is not None:
                    all_recoveries.append(result)

    if not all_recoveries:
        return None
    return pd.concat(all_recoveries, ignore_index=True)


@reactive.calc
async def get_heatmap_stats():
    data = await process_data()
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

def _make_custom_grid(figheight=8):
    """
    Crea una figura con GridSpec para que TODOS los ejes sean SubplotSpec válidos.
    Esto evita el error 'NoneType object has no attribute rowspan' en Shiny.
    """
    from matplotlib.gridspec import GridSpec
    figwidth = figheight * 1.6
    fig = plt.figure(figsize=(figwidth, figheight), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, 
                  height_ratios=[1, 15, 1], 
                  width_ratios=[40, 1],
                  left=0.05, right=0.95, top=0.95, bottom=0.05,
                  wspace=0.02, hspace=0.05)
    
    axs = {
        'title': fig.add_subplot(gs[0, 0]),
        'pitch': fig.add_subplot(gs[1, 0]),
        'cbar': fig.add_subplot(gs[1, 1]),
        'endnote': fig.add_subplot(gs[2, 0])
    }
    for k in ['title', 'endnote']:
        axs[k].axis('off')
    return fig, axs


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# SECCIÓN DE VISUALIZACIÓN
# ─────────────────────────────────────────────
with ui.navset_underline(id="tabs"):
    with ui.nav_panel("Volumen de recuperaciones"):
        with ui.card(height="850px", full_screen=True):
            ui.card_header("Mapa de densidad: Volumen de recuperaciones y tiros")
            @render.plot
            async def field_volumen():
                stats = await get_heatmap_stats()
                fig, axs = _make_custom_grid(figheight=8)
                pitch = Pitch(pitch_type='statsbomb', line_color='black', pitch_color='white', line_zorder=2)
                pitch.draw(ax=axs['pitch'])

                if stats is None:
                    axs['pitch'].text(60, 40, "Esperando análisis...", color='#94a3b8', ha='center', fontsize=14)
                    return fig

                pcm = pitch.heatmap(stats['rec'], ax=axs['pitch'], cmap='Reds', edgecolor='black', linewidth=0.5, alpha=0.7)
                cx, cy = stats['rec']['cx'].flatten(), stats['rec']['cy'].flatten()
                recs, shots = stats['rec']['statistic'].flatten(), stats['shot']['statistic'].flatten()

                for i in range(len(cx)):
                    if recs[i] > 0:
                        pitch.text(cx[i], cy[i], f"R:{int(recs[i])}\nT:{int(shots[i])}",
                                   ax=axs['pitch'], color='black', fontsize=8, ha='center', va='center', fontweight='bold')

                cb = plt.colorbar(pcm, cax=axs['cbar'])
                cb.ax.tick_params(labelsize=8)
                cb.set_label('Intensidad de recuperaciones', size=9)

                axs['title'].text(0, 0.7, "Volumen de Recuperaciones (R) y Tiros (T)", fontsize=18, fontweight='800', color='#0f172a')
                axs['title'].text(0, 0.2, "Distribución espacial de recuperaciones de balón y su finalización en tiro", fontsize=11, color='#64748b')
                axs['endnote'].text(1, 0.5, "Dirección de ataque ➜", ha='right', va='center', fontsize=10, fontstyle='italic', color='#64748b')
                return fig

    with ui.nav_panel("Eficacia y goles"):
        with ui.card(height="850px", full_screen=True):
            ui.card_header("Mapa de eficacia: Probabilidad de éxito tras recuperación")
            @render.plot
            async def field_eficacia():
                stats = await get_heatmap_stats()
                fig, axs = _make_custom_grid(figheight=8)
                pitch = Pitch(pitch_type='statsbomb', line_color='black', pitch_color='white', line_zorder=2)
                pitch.draw(ax=axs['pitch'])

                if stats is None:
                    axs['pitch'].text(60, 40, "Esperando análisis...", color='#94a3b8', ha='center', fontsize=14)
                    return fig

                pcm = pitch.heatmap(stats['pct'], ax=axs['pitch'], cmap='Reds', edgecolor='black', linewidth=0.5, alpha=0.7)
                cx, cy = stats['pct']['cx'].flatten(), stats['pct']['cy'].flatten()
                pcts, goals, recs = stats['pct']['statistic'].flatten(), stats['goal']['statistic'].flatten(), stats['rec']['statistic'].flatten()

                for i in range(len(cx)):
                    if recs[i] > 0:
                        txt = f"{pcts[i]:.1f}%\nG:{int(goals[i])}"
                        pitch.text(cx[i], cy[i], txt, ax=axs['pitch'], color='black', fontsize=8, ha='center', va='center', fontweight='bold')

                cb = plt.colorbar(pcm, cax=axs['cbar'])
                cb.ax.tick_params(labelsize=8)
                cb.set_label('% Eficacia (Transición a Tiro)', size=9)

                axs['title'].text(0, 0.7, "Eficacia de Transición (%) y Goles (G)", fontsize=18, fontweight='800', color='#0f172a')
                axs['title'].text(0, 0.2, "Porcentaje de recuperaciones que terminan en tiro por zona", fontsize=11, color='#64748b')
                axs['endnote'].text(1, 0.5, "Dirección de ataque ➜", ha='right', va='center', fontsize=10, fontstyle='italic', color='#64748b')
                return fig

# ─────────────────────────────────────────────
# SECCIONES INFERIORES: Metodología y Resultados
# ─────────────────────────────────────────────
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card(style="max-height: 350px; overflow-y: auto;"):
        ui.card_header("Metodología de Análisis")
        ui.markdown("""
        Se analizan los eventos de **Ball Recovery** y se rastrea la posesión hasta su finalización.
        
        *   **Volumen**: Densidad de recuperaciones por zona.
        *   **Eficacia**: % de recuperaciones que terminan en tiro.
        
        **Origen**: StatsBomb Open Data.
        """)

    with ui.card(style="max-height: 350px; overflow-y: auto;"):
        ui.card_header("Resumen de Resultados")
        @render.ui
        async def analysis_summary():
            data = await process_data()
            if data is None or data.empty:
                return ui.p("Esperando selección de datos...", style="color: var(--muted); padding: 1rem;")
            
            total = len(data)
            shots = int(data['ends_in_shot'].sum())
            goals = int(data['ends_in_goal'].sum())
            pct_shot = (shots / total) * 100
            pct_goal = (goals / total) * 100
            
            with ui.layout_columns(col_widths=[4, 4, 4]):
                with ui.div(class_="kpi-item"):
                    ui.div(f"{total:,}", class_="kpi-value", style="color: #1e40af; font-size: 1.5rem;")
                    ui.div("Recuperaciones", class_="kpi-label", style="font-size: 0.7rem;")
                with ui.div(class_="kpi-item"):
                    ui.div(f"{pct_shot:.1f}%", class_="kpi-value", style="color: #0891b2; font-size: 1.5rem;")
                    ui.div("Eficacia", class_="kpi-label", style="font-size: 0.7rem;")
                with ui.div(class_="kpi-item"):
                    ui.div(f"{goals}", class_="kpi-value", style="color: #be123c; font-size: 1.5rem;")
                    ui.div("Goles", class_="kpi-label", style="font-size: 0.7rem;")


