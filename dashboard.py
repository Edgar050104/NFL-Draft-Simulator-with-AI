import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nfl_bayesian_fuzzy import NFLDraftSimulator, NFLBayesianNetwork, NFLFuzzySystem

# Configuración de la página
st.set_page_config(
    page_title="Dashboard del Simulador de Draft NFL",
    page_icon="🏈",
    layout="wide"
)

# Título y descripción
st.title("🏈 Dashboard Interactivo del Simulador de Draft NFL")
st.markdown("""
Este dashboard te permite visualizar y analizar los resultados de la simulación del draft de la NFL
utilizando técnicas avanzadas de inteligencia artificial como redes bayesianas y lógica difusa.
""")

# Inicializar simulador
@st.cache_resource
def load_simulator():
    try:
        simulator = NFLDraftSimulator(
            prospects_file="prospects.csv",
            team_needs_file="team_needs.csv",
            trends_file="trends.csv",
            draft_order_file="draft_order.csv"
        )
        return simulator, None
    except Exception as e:
        return None, str(e)

# Cargar el simulador
simulator, error = load_simulator()

if error:
    st.error(f"Error al cargar el simulador: {error}")
    st.warning("Asegúrate de que los archivos CSV necesarios estén disponibles en el directorio actual.")
    st.stop()

# Sidebar para configuración
st.sidebar.header("Configuración de Simulación")

# Opciones de simulación
simulation_limit = st.sidebar.slider(
    "Número de picks a simular",
    min_value=10,
    max_value=len(simulator.draft_order),
    value=32,
    step=1
)

# Personalización de ponderaciones
st.sidebar.subheader("Personalizar Ponderaciones")
use_custom_weights = st.sidebar.checkbox("Personalizar factores de decisión")

need_weight = 0.6
rank_weight = 0.3
hist_weight = 0.1

if use_custom_weights:
    need_weight = st.sidebar.slider("Peso de necesidad del equipo", 0.1, 0.9, 0.6, 0.1)
    rank_weight = st.sidebar.slider("Peso de ranking del jugador", 0.1, 0.9, 0.3, 0.1)
    # Ajustar para que la suma sea 1
    hist_weight = 1.0 - need_weight - rank_weight
    hist_weight = max(0.1, min(hist_weight, 0.9))  # Limitar entre 0.1 y 0.9
    st.sidebar.write(f"Peso de tendencia histórica: {hist_weight:.1f}")

# Botón para iniciar simulación
simulate_button = st.sidebar.button("Simular Draft", type="primary")

# Configuración para editar necesidades de equipo
st.sidebar.header("Modificar Necesidades de Equipo")
teams = sorted(simulator.team_needs['TEAM'].unique())
selected_team = st.sidebar.selectbox("Seleccionar equipo", teams)

# Mostrar necesidades actuales
team_row = simulator.team_needs[simulator.team_needs['TEAM'] == selected_team]
positions = [col for col in simulator.team_needs.columns 
             if col not in ['TEAM', 'MUST-HAVES', 'PARTIAL NEEDS', 'TOTAL PICKS', 'T-100 PICKS']]

st.sidebar.subheader(f"Necesidades de {selected_team}")
position_to_modify = st.sidebar.selectbox("Posición a modificar", positions)

# Obtener valor actual
current_value = 5
if not team_row.empty and position_to_modify in team_row.columns:
    current_value = team_row[position_to_modify].values[0]

new_value = st.sidebar.slider(
    f"Necesidad para {position_to_modify}", 
    min_value=1, 
    max_value=10, 
    value=int(current_value)
)

update_button = st.sidebar.button("Actualizar Necesidad")

if update_button:
    simulator.modify_team_needs(selected_team, position_to_modify, new_value)
    st.sidebar.success(f"Necesidad de {selected_team} para {position_to_modify} actualizada a {new_value}")

# Estado de sesión para almacenar resultados
if 'selections' not in st.session_state:
    st.session_state['selections'] = None
    st.session_state['has_simulated'] = False

# Ejecutar simulación cuando se presiona el botón
if simulate_button:
    st.write("### Simulando el draft...")
    
    # Limitar orden del draft según configuración
    limited_order = simulator.draft_order.head(simulation_limit).copy()
    
    # Guardar orden original
    original_order = simulator.draft_order.copy()
    
    # Asignar orden limitado
    simulator.draft_order = limited_order
    
    # Simular draft
    with st.spinner("Ejecutando simulación del draft..."):
        selections = simulator.simulate_draft()
        st.session_state['selections'] = selections
        st.session_state['has_simulated'] = True
    
    # Restaurar orden original
    simulator.draft_order = original_order
    
    st.success(f"Simulación completada. Se realizaron {len(selections)} selecciones.")

# Mostrar resultados si hay simulación
if st.session_state['has_simulated'] and st.session_state['selections']:
    # Convertir a DataFrame
    df_selections = pd.DataFrame(st.session_state['selections'])
    
    # Secciones de pestañas para mostrar diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs(["Resultados Generales", "Análisis por Equipo", "Análisis por Posición", "Visualizaciones"])
    
    with tab1:
        st.header("Resultados Generales del Draft")
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Picks", len(df_selections))
        with col2:
            avg_rank = df_selections['player_rank'].mean()
            st.metric("Ranking Promedio", f"{avg_rank:.2f}")
        with col3:
            avg_fit = df_selections['fit_score'].mean()
            st.metric("Fit Score Promedio", f"{avg_fit:.2f}")
        
        # Tabla de todos los picks
        st.subheader("Todos los Picks")
        st.dataframe(
            df_selections[['round', 'overall', 'team', 'player_name', 'player_position', 
                        'player_college', 'player_rank', 'fit_score']],
            hide_index=True,
            use_container_width=True
        )
        
        # Top picks por fit score
        st.subheader("Top 5 Picks con Mejor Fit Score")
        top_fits = df_selections.sort_values(by='fit_score', ascending=False).head(5)
        for idx, pick in top_fits.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.subheader(f"Pick #{pick['overall']}")
                    st.caption(f"Ronda {pick['round']}")
                with cols[1]:
                    st.subheader(f"{pick['player_name']} ({pick['player_position']})")
                    st.caption(f"Equipo: {pick['team']} | Universidad: {pick['player_college']}")
                    st.progress(min(pick['fit_score'], 100) / 100)
                    st.caption(f"Fit Score: {pick['fit_score']:.2f}/100 | Ranking: {pick['player_rank']}")
                    st.text(f"Razón: {pick['reasoning']}")
                st.divider()
    
    with tab2:
        st.header("Análisis por Equipo")
        
        # Selector de equipo
        team_options = sorted(df_selections['team'].unique())
        selected_team_analysis = st.selectbox("Seleccionar equipo para análisis", team_options)
        
        # Filtrar selecciones del equipo
        team_selections = df_selections[df_selections['team'] == selected_team_analysis]
        
        # Métricas del equipo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Picks", len(team_selections))
        with col2:
            team_avg_rank = team_selections['player_rank'].mean()
            st.metric("Ranking Promedio", f"{team_avg_rank:.2f}")
        with col3:
            team_avg_fit = team_selections['fit_score'].mean()
            st.metric("Fit Score Promedio", f"{team_avg_fit:.2f}")
        
        # Mostrar selecciones del equipo
        st.subheader(f"Selecciones de {selected_team_analysis}")
        for _, pick in team_selections.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.subheader(f"Pick #{pick['overall']}")
                    st.caption(f"Ronda {pick['round']}, Pick {pick['pick']}")
                with col2:
                    st.subheader(f"{pick['player_name']} ({pick['player_position']})")
                    st.caption(f"{pick['player_college']} - Ranking: {pick['player_rank']}")
                    st.progress(min(pick['fit_score'], 100) / 100)
                    st.caption(f"Fit Score: {pick['fit_score']:.2f}/100")
                    st.text(f"Razón: {pick['reasoning']}")
                st.divider()
        
        # Gráfico de posiciones seleccionadas por el equipo
        st.subheader("Distribución de Posiciones")
        pos_counts = team_selections['player_position'].value_counts()
        if len(pos_counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            pos_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            plt.title(f"Posiciones Seleccionadas por {selected_team_analysis}")
            plt.ylabel("")
            st.pyplot(fig)
    
    with tab3:
        st.header("Análisis por Posición")
        
        # Distribución global de posiciones
        st.subheader("Distribución de Posiciones en el Draft")
        pos_counts_all = df_selections['player_position'].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        pos_counts_all.plot(kind='bar', ax=ax)
        plt.title("Número de Jugadores Seleccionados por Posición")
        plt.xlabel("Posición")
        plt.ylabel("Número de Jugadores")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Selector de posición específica
        position_options = sorted(df_selections['player_position'].unique())
        selected_position = st.selectbox("Seleccionar posición para análisis detallado", position_options)
        
        # Filtrar por posición
        position_selections = df_selections[df_selections['player_position'] == selected_position]
        
        # Métricas de la posición
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Picks", len(position_selections))
        with col2:
            pos_avg_rank = position_selections['player_rank'].mean()
            st.metric("Ranking Promedio", f"{pos_avg_rank:.2f}")
        with col3:
            pos_avg_fit = position_selections['fit_score'].mean()
            st.metric("Fit Score Promedio", f"{pos_avg_fit:.2f}")
        
        # Distribución por ronda para esta posición
        st.subheader(f"Distribución de {selected_position} por Ronda")
        round_counts = position_selections['round'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        round_counts.plot(kind='bar', ax=ax)
        plt.title(f"Jugadores de {selected_position} Seleccionados por Ronda")
        plt.xlabel("Ronda")
        plt.ylabel("Número de Jugadores")
        st.pyplot(fig)
        
        # Tabla de jugadores en esta posición
        st.subheader(f"Jugadores de {selected_position} Seleccionados")
        st.dataframe(
            position_selections[['round', 'overall', 'team', 'player_name', 
                               'player_college', 'player_rank', 'fit_score']],
            hide_index=True,
            use_container_width=True
        )
    
    with tab4:
        st.header("Visualizaciones Avanzadas")
        
        # Distribución de posiciones por ronda
        st.subheader("Distribución de Posiciones por Ronda")
        position_round = pd.crosstab(df_selections['player_position'], df_selections['round'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        position_round.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Distribución de Posiciones por Ronda')
        plt.xlabel('Posición')
        plt.ylabel('Número de Jugadores')
        plt.legend(title='Ronda')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribución de Fit Scores
        st.subheader("Distribución de Fit Scores")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df_selections['fit_score'], bins=20, kde=True, ax=ax)
        plt.title('Distribución de Fit Scores')
        plt.xlabel('Fit Score')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Ranking promedio por equipo
        st.subheader("Ranking Promedio por Equipo")
        team_avg_rank = df_selections.groupby('team')['player_rank'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        team_avg_rank.plot(kind='bar', ax=ax)
        plt.title('Ranking Promedio de Selecciones por Equipo')
        plt.xlabel('Equipo')
        plt.ylabel('Ranking Promedio')
        # Invertir el eje Y para que los valores más bajos (mejores) estén arriba
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Mapa de calor de posiciones por equipo
        st.subheader("Mapa de Calor: Equipos vs Posiciones")
        team_pos_matrix = pd.crosstab(df_selections['team'], df_selections['player_position'])
        
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(team_pos_matrix, annot=True, cmap="YlGnBu", fmt='d', ax=ax)
        plt.title('Número de Selecciones por Equipo y Posición')
        plt.tight_layout()
        st.pyplot(fig)
else:
    # Mostrar contenido inicial si no hay simulación
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Simular Draft' para comenzar")
    
    # Mostrar información sobre el sistema
    st.header("Sobre este Simulador")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sistema de Lógica Difusa")
        st.write("""
        El simulador utiliza lógica difusa para calcular el "fit score" entre un prospecto y un equipo,
        considerando factores como:
        
        - Nivel de necesidad del equipo en esa posición
        - Ranking del jugador (calidad del prospecto)
        - Tendencias históricas del draft
        
        La lógica difusa simula el razonamiento humano con reglas como:
        "Si la necesidad es alta y el jugador es élite, entonces el ajuste es alto"
        """)
    
    with col2:
        st.subheader("Red Bayesiana")
        st.write("""
        El sistema implementa una red bayesiana que calcula probabilidades basadas en datos históricos:
        
        - ¿Qué tan probable es que un QB sea seleccionado en la primera ronda?
        - ¿Qué posiciones son más comunes en rondas tardías?
        - ¿Cuál es la distribución histórica de selecciones por posición?
        
        Estas probabilidades influyen en las decisiones del simulador, haciéndolas más realistas.
        """)
    
    # Mostrar ejemplos de análisis que se pueden realizar
    st.header("Análisis que puedes realizar")
    
    st.markdown("""
    Una vez que ejecutes la simulación, podrás:
    
    - Ver todos los picks del draft y sus puntuaciones de ajuste
    - Analizar las selecciones por equipo
    - Estudiar tendencias por posición y ronda
    - Visualizar distribuciones y patrones con gráficos interactivos
    - Comparar estrategias modificando las necesidades de los equipos
    
    Ajusta los parámetros en el panel lateral para personalizar la simulación según tus preferencias.
    """)
    
    # Mostrar requisitos
    st.header("Requisitos")
    
    st.markdown("""
    Para utilizar este simulador, debes tener los siguientes archivos CSV en el mismo directorio:
    
    - **prospects.csv**: Lista de prospectos con sus rankings y posiciones
    - **team_needs.csv**: Necesidades de cada equipo por posición
    - **trends.csv**: Tendencias históricas de selección por posición y ronda
    - **draft_order.csv**: Orden completo del draft
    
    Estos archivos contienen los datos necesarios para que el simulador funcione correctamente.
    """)

# Pie de página
st.markdown("---")
st.caption("Simulador de Draft NFL con Inteligencia Artificial | Desarrollado con Python, Redes Bayesianas y Lógica Difusa")