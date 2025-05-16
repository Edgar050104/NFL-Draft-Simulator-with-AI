import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

class NFLBayesianNetwork:
    """
    Implementa una red bayesiana para predecir probabilidades de selección
    en el draft de la NFL basado en tendencias históricas.
    """
    
    def __init__(self, trends_df):
        """
        Inicializa la red bayesiana con datos históricos.
        
        Args:
            trends_df (DataFrame): DataFrame con tendencias históricas por posición y ronda
        """
        self.trends_df = trends_df
        self.position_probs = {}  # Probabilidades generales por posición
        self.round_probs = {}     # Probabilidades por ronda
        self.position_round_probs = {}  # Probabilidades conjuntas
        
        # Construir las tablas de probabilidad
        self._build_probability_tables()
        
    def _build_probability_tables(self):
        """Construye todas las tablas de probabilidad necesarias para la red bayesiana"""
        # Total de jugadores drafteados en la historia (según datos)
        total_players = self.trends_df['TOTAL'].sum()
        
        # Probabilidades por posición P(Position)
        for pos in self.trends_df['POS'].unique():
            pos_data = self.trends_df[self.trends_df['POS'] == pos]
            pos_total = pos_data['TOTAL'].sum()
            self.position_probs[pos] = pos_total / total_players
        
        # Probabilidades por ronda P(Round)
        round_totals = {
            1: self.trends_df['R1'].sum(),
            2: self.trends_df['R2'].sum(),
            3: self.trends_df['R3'].sum(),
            4: self.trends_df['R4'].sum(),
            5: 0,  # Inicializar para R5-R7
            6: 0,
            7: 0
        }
        
        # Distribuir R5-R7 entre las rondas 5, 6 y 7 (proporcionalmente)
        r5_r7_total = self.trends_df['R5-R7'].sum()
        round_totals[5] = r5_r7_total / 3
        round_totals[6] = r5_r7_total / 3
        round_totals[7] = r5_r7_total / 3
        
        # Calcular probabilidades por ronda
        for round_num in range(1, 8):
            self.round_probs[round_num] = round_totals[round_num] / total_players
        
        # Probabilidades condicionales P(Round | Position)
        for pos in self.trends_df['POS'].unique():
            self.position_round_probs[pos] = {}
            pos_data = self.trends_df[self.trends_df['POS'] == pos]
            pos_total = pos_data['TOTAL'].sum()
            
            if pos_total > 0:
                # Rondas 1-4
                for round_num in range(1, 5):
                    round_col = f'R{round_num}'
                    round_count = pos_data[round_col].sum()
                    self.position_round_probs[pos][round_num] = round_count / pos_total
                
                # Rondas 5-7 (distribuir proporcionalmente)
                r5_r7_count = pos_data['R5-R7'].sum()
                for round_num in range(5, 8):
                    self.position_round_probs[pos][round_num] = (r5_r7_count / 3) / pos_total
    
    def get_position_probability(self, position):
        """
        Obtiene la probabilidad general de que un jugador de cierta posición sea drafteado.
        
        Args:
            position (str): Posición del jugador
        
        Returns:
            float: Probabilidad entre 0 y 1
        """
        if position in self.position_probs:
            return self.position_probs[position]
        return 0.01  # Valor por defecto para posiciones no encontradas
    
    def get_round_probability(self, round_num):
        """
        Obtiene la probabilidad general de que un jugador sea drafteado en cierta ronda.
        
        Args:
            round_num (int): Número de ronda (1-7)
        
        Returns:
            float: Probabilidad entre 0 y 1
        """
        if round_num in self.round_probs:
            return self.round_probs[round_num]
        return 0.01  # Valor por defecto
    
    def get_position_round_probability(self, position, round_num):
        """
        Obtiene la probabilidad de que un jugador de cierta posición sea elegido en una ronda específica.
        P(Round | Position)
        
        Args:
            position (str): Posición del jugador
            round_num (int): Número de ronda (1-7)
        
        Returns:
            float: Probabilidad entre 0 y 1
        """
        if position in self.position_round_probs and round_num in self.position_round_probs[position]:
            return self.position_round_probs[position][round_num]
        return 0.01  # Valor por defecto
    
    def visualize_network(self):
        """
        Crea una visualización de la red bayesiana.
        
        Returns:
            fig: Figura de matplotlib con el grafo de la red
        """
        G = nx.DiGraph()
        
        # Nodos
        G.add_node("Posición", pos=(0, 1))
        G.add_node("Ronda", pos=(1, 1))
        G.add_node("Selección", pos=(0.5, 0))
        
        # Aristas dirigidas
        G.add_edge("Posición", "Selección")
        G.add_edge("Ronda", "Selección")
        
        # Crear figura
        plt.figure(figsize=(8, 6))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Dibujar aristas
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        plt.title("Red Bayesiana para Predicción de Selecciones del Draft", fontsize=14)
        plt.axis('off')
        
        return plt.gcf()


class NFLFuzzySystem:
    """
    Implementa un sistema de lógica difusa para calcular el "fit score" 
    de un jugador con respecto a un equipo.
    """
    
    def __init__(self):
        """Inicializa el sistema de lógica difusa"""
        # Definir universos de variables
        self.need = ctrl.Antecedent(np.arange(0, 11, 1), 'need')
        self.ranking = ctrl.Antecedent(np.arange(0, 1501, 1), 'ranking')
        self.trend = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'trend')
        self.fit = ctrl.Consequent(np.arange(0, 101, 1), 'fit')
        
        # Definir funciones de membresía para necesidad
        self.need['baja'] = fuzz.trimf(self.need.universe, [0, 0, 4])
        self.need['media'] = fuzz.trimf(self.need.universe, [3, 5, 7])
        self.need['alta'] = fuzz.trimf(self.need.universe, [6, 10, 10])
        
        # Definir funciones de membresía para ranking
        self.ranking['elite'] = fuzz.trimf(self.ranking.universe, [0, 0, 15])
        self.ranking['top'] = fuzz.trimf(self.ranking.universe, [10, 50, 100])
        self.ranking['bueno'] = fuzz.trimf(self.ranking.universe, [80, 200, 350])
        self.ranking['promedio'] = fuzz.trimf(self.ranking.universe, [300, 600, 900])
        self.ranking['bajo'] = fuzz.trimf(self.ranking.universe, [800, 1500, 1500])
        
        # Definir funciones de membresía para tendencia
        self.trend['rara'] = fuzz.trimf(self.trend.universe, [0, 0, 0.05])
        self.trend['ocasional'] = fuzz.trimf(self.trend.universe, [0.03, 0.08, 0.13])
        self.trend['común'] = fuzz.trimf(self.trend.universe, [0.1, 0.2, 0.3])
        self.trend['frecuente'] = fuzz.trimf(self.trend.universe, [0.25, 1, 1])
        
        # Definir funciones de membresía para fit score
        self.fit['bajo'] = fuzz.trimf(self.fit.universe, [0, 0, 40])
        self.fit['medio'] = fuzz.trimf(self.fit.universe, [30, 50, 70])
        self.fit['alto'] = fuzz.trimf(self.fit.universe, [60, 100, 100])
        
        # Definir reglas
        self._create_rules()
        
        # Crear sistema de control
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def _create_rules(self):
        """Crea las reglas difusas para el sistema"""
        self.rules = [
            # Reglas para necesidad alta
            ctrl.Rule(self.need['alta'] & self.ranking['elite'], self.fit['alto']),
            ctrl.Rule(self.need['alta'] & self.ranking['top'], self.fit['alto']),
            ctrl.Rule(self.need['alta'] & self.ranking['bueno'], self.fit['medio']),
            ctrl.Rule(self.need['alta'] & self.ranking['promedio'], self.fit['medio']),
            ctrl.Rule(self.need['alta'] & self.ranking['bajo'], self.fit['bajo']),
            
            # Reglas para necesidad media
            ctrl.Rule(self.need['media'] & self.ranking['elite'], self.fit['alto']),
            ctrl.Rule(self.need['media'] & self.ranking['top'], self.fit['medio']),
            ctrl.Rule(self.need['media'] & self.ranking['bueno'], self.fit['medio']),
            ctrl.Rule(self.need['media'] & self.ranking['promedio'], self.fit['bajo']),
            ctrl.Rule(self.need['media'] & self.ranking['bajo'], self.fit['bajo']),
            
            # Reglas para necesidad baja
            ctrl.Rule(self.need['baja'] & self.ranking['elite'], self.fit['medio']),
            ctrl.Rule(self.need['baja'] & self.ranking['top'], self.fit['bajo']),
            ctrl.Rule(self.need['baja'] & self.ranking['bueno'], self.fit['bajo']),
            ctrl.Rule(self.need['baja'] & self.ranking['promedio'], self.fit['bajo']),
            ctrl.Rule(self.need['baja'] & self.ranking['bajo'], self.fit['bajo']),
            
            # Incorporar tendencia histórica
            ctrl.Rule(self.trend['frecuente'], self.fit['alto']),
            ctrl.Rule(self.trend['rara'], self.fit['bajo']),
            
            # Reglas combinadas
            ctrl.Rule(self.need['alta'] & self.ranking['elite'] & self.trend['frecuente'], self.fit['alto']),
            ctrl.Rule(self.need['alta'] & self.ranking['bueno'] & self.trend['frecuente'], self.fit['alto']),
            ctrl.Rule(self.need['media'] & self.ranking['elite'] & self.trend['rara'], self.fit['medio']),
        ]
    
    def calculate_fit(self, need_score, ranking, trend_prob):
        """
        Calcula el fit score usando el sistema difuso.
        
        Args:
            need_score (float): Puntuación de necesidad (0-10)
            ranking (int): Ranking del jugador (1-1500)
            trend_prob (float): Probabilidad histórica (0-1)
            
        Returns:
            float: Fit score (0-100)
        """
        # Establecer entradas
        self.simulation.input['need'] = need_score
        self.simulation.input['ranking'] = ranking
        self.simulation.input['trend'] = trend_prob
        
        try:
            # Computar resultado
            self.simulation.compute()
            return self.simulation.output['fit']
        except:
            # En caso de error, usar cálculo alternativo
            normalized_rank = max(0, min(100, 100 - (ranking / 15)))
            normalized_trend = trend_prob * 100
            
            # Fórmula ponderada simple
            return (need_score * 0.6 + normalized_rank * 0.3 + normalized_trend * 0.1) * 10
    
    def visualize_membership(self):
        """
        Visualiza las funciones de membresía del sistema difuso.
        
        Returns:
            fig: Figura de matplotlib con las funciones de membresía
        """
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(10, 12))
        
        # Graficar funciones de membresía para necesidad
        self.need.view(sim=self.simulation, ax=ax0)
        ax0.set_title('Necesidad del Equipo')
        ax0.legend()
        
        # Graficar funciones de membresía para ranking
        self.ranking.view(sim=self.simulation, ax=ax1)
        ax1.set_title('Ranking del Jugador')
        ax1.legend()
        
        # Graficar funciones de membresía para tendencia
        self.trend.view(sim=self.simulation, ax=ax2)
        ax2.set_title('Tendencia Histórica')
        ax2.legend()
        
        # Graficar funciones de membresía para fit
        self.fit.view(sim=self.simulation, ax=ax3)
        ax3.set_title('Fit Score')
        ax3.legend()
        
        plt.tight_layout()
        return fig


class NFLDraftSimulator:
    """
    Simulador completo de draft de la NFL que utiliza lógica difusa y redes bayesianas
    para tomar decisiones inteligentes.
    """
    
    def __init__(self, prospects_file, team_needs_file, trends_file, draft_order_file):
        """
        Inicializa el simulador de draft de la NFL cargando los archivos de datos.
        
        Args:
            prospects_file (str): Ruta al archivo CSV con los prospectos
            team_needs_file (str): Ruta al archivo CSV con las necesidades de los equipos
            trends_file (str): Ruta al archivo CSV con tendencias históricas del draft
            draft_order_file (str): Ruta al archivo CSV con el orden del draft
        """
        self.prospects = pd.read_csv(prospects_file)
        self.team_needs = pd.read_csv(team_needs_file)
        self.trends = pd.read_csv(trends_file)
        self.draft_order = pd.read_csv(draft_order_file)
        
        # Convertir el ranking a numérico si no lo es
        if not pd.api.types.is_numeric_dtype(self.prospects['RANK']):
            # Manejar casos especiales como X, XX, XXX (representan posiciones fuera del top 1000)
            self.prospects['RANK'] = self.prospects['RANK'].apply(self._convert_rank)
        
        # Lista para almacenar las selecciones realizadas
        self.selections = []
        
        # Inicializar los sistemas de lógica difusa y redes bayesianas
        self.fuzzy_system = NFLFuzzySystem()
        self.bayesian_network = NFLBayesianNetwork(self.trends)
    
    def _convert_rank(self, rank):
        """Convierte rankings con formato especial (X, XX, XXX) a numéricos"""
        if isinstance(rank, str):
            if rank == 'X':
                return 1001
            elif rank == 'XX':
                return 1201
            elif rank == 'XXX':
                return 1401
        return float(rank)
    
    def get_team_needs(self, team, position):
        """
        Obtiene la puntuación de necesidad de un equipo para una posición específica.
        
        Args:
            team (str): Nombre del equipo
            position (str): Posición del jugador
            
        Returns:
            float: Puntuación de necesidad (0-10)
        """
        team_row = self.team_needs[self.team_needs['TEAM'] == team]
        
        if team_row.empty:
            return 5  # Valor por defecto
        
        # Buscar la columna que corresponde a esta posición
        if position in team_row.columns:
            return team_row[position].values[0]
        
        # Si no encuentra la posición exacta, usar un enfoque alternativo
        # Por ejemplo, buscar en MUST-HAVES o PARTIAL NEEDS
        must_haves = str(team_row['MUST-HAVES'].values[0])
        partial_needs = str(team_row['PARTIAL NEEDS'].values[0])
        
        if position in must_haves:
            return 9
        elif position in partial_needs:
            return 6
        else:
            return 3
    
    def calculate_fit_score(self, team, position, ranking, round_num):
        """
        Calcula el "fit score" para un jugador con respecto a un equipo usando lógica difusa y redes bayesianas.
        
        Args:
            team (str): Nombre del equipo
            position (str): Posición del jugador
            ranking (int): Ranking general del jugador
            round_num (int): Número de ronda
            
        Returns:
            float: Puntuación de ajuste entre 0 y 100
        """
        # Obtener la necesidad del equipo para esta posición
        need_score = self.get_team_needs(team, position)
        
        # Obtener la probabilidad histórica de esta posición en esta ronda
        trend_prob = self.bayesian_network.get_position_round_probability(position, round_num)
        
        # Usar el sistema difuso para calcular el fit score
        fit_score = self.fuzzy_system.calculate_fit(need_score, ranking, trend_prob)
        
        # Añadir un poco de aleatoriedad para simular incertidumbre en decisiones reales
        randomness = random.uniform(0.9, 1.1)
        final_fit = fit_score * randomness
        
        return min(100, max(0, final_fit))
    
    def get_best_player(self, team, round_num, available_players):
        """
        Determina el mejor jugador disponible para un equipo en función de sus necesidades.
        
        Args:
            team (str): Nombre del equipo
            round_num (int): Número de ronda actual
            available_players (DataFrame): DataFrame con los jugadores disponibles
            
        Returns:
            dict: Información del jugador seleccionado y su puntuación
        """
        best_score = -1
        best_player = None
        best_reasoning = ""
        
        # Evaluar cada jugador disponible
        for idx, player in available_players.iterrows():
            position = player['POS']
            ranking = player['RANK']
            name = player['NAME']
            college = player['COLLEGE']
            
            # Calcular el fit score
            fit_score = self.calculate_fit_score(team, position, ranking, round_num)
            
            # Crear razonamiento para esta selección
            need_score = self.get_team_needs(team, position)
            need_text = "baja"
            if need_score >= 8:
                need_text = "muy alta"
            elif need_score >= 6:
                need_text = "alta"
            elif need_score >= 4:
                need_text = "media"
            
            # Determinar categoría de ranking
            rank_text = "bajo ranking"
            if ranking <= 15:
                rank_text = "elite (top 15)"
            elif ranking <= 50:
                rank_text = "alto potencial (top 50)"
            elif ranking <= 100:
                rank_text = "talento de primera (top 100)"
            elif ranking <= 300:
                rank_text = "buen prospecto"
            
            # Comprobar probabilidad histórica
            prob = self.bayesian_network.get_position_round_probability(position, round_num)
            prob_text = "inusual"
            if prob >= 0.15:
                prob_text = "muy común"
            elif prob >= 0.1:
                prob_text = "común"
            elif prob >= 0.05:
                prob_text = "ocasional"
            
            reasoning = f"Necesidad {need_text} en {position}, jugador de {rank_text}, históricamente {prob_text} en ronda {round_num}"
            
            # Actualizar si es el mejor hasta ahora
            if fit_score > best_score:
                best_score = fit_score
                best_player = player
                best_reasoning = reasoning
        
        if best_player is not None:
            return {
                "player": best_player,
                "score": best_score,
                "reasoning": best_reasoning
            }
        return None
    
    def simulate_draft(self):
        """
        Simula el draft completo siguiendo el orden establecido.
        
        Returns:
            list: Lista de selecciones realizadas
        """
        # Reiniciar selecciones
        self.selections = []
        
        # Copiar prospectos para trabajar con jugadores disponibles
        available_players = self.prospects.copy()
        
        # Recorrer cada pick en el orden del draft
        for idx, pick in self.draft_order.iterrows():
            team = pick['TEAM']
            round_num = pick['ROUND']
            overall = pick['OVERALL']
            
            # Obtener el mejor jugador para este equipo
            selection = self.get_best_player(team, round_num, available_players)
            
            if selection:
                player = selection["player"]
                score = selection["score"]
                reasoning = selection["reasoning"]
                
                # Registrar la selección
                draft_pick = {
                    "round": round_num,
                    "pick": pick['PICK'],
                    "overall": overall,
                    "team": team,
                    "player_name": player['NAME'],
                    "player_position": player['POS'],
                    "player_college": player['COLLEGE'],
                    "player_rank": player['RANK'],
                    "fit_score": score,
                    "reasoning": reasoning
                }
                
                self.selections.append(draft_pick)
                
                # Eliminar al jugador de los disponibles
                available_players = available_players[available_players['NAME'] != player['NAME']]
                
                print(f"Pick #{overall}: {team} selecciona a {player['NAME']} ({player['POS']}, {player['COLLEGE']})")
                print(f"Razón: {reasoning}")
                print(f"Fit Score: {score:.2f}/100")
                print("-" * 80)
        
        return self.selections
    
    def get_team_draft(self, team_name):
        """
        Obtiene todas las selecciones de un equipo específico.
        
        Args:
            team_name (str): Nombre del equipo
            
        Returns:
            list: Lista de selecciones del equipo
        """
        return [pick for pick in self.selections if pick["team"] == team_name]
    
    def visualize_draft(self):
        """
        Visualiza los resultados del draft con gráficos.
        
        Returns:
            dict: Diccionario con las figuras generadas
        """
        if not self.selections:
            print("No hay draft simulado para visualizar. Ejecute simulate_draft() primero.")
            return {}
        
        # Convertir selecciones a DataFrame para facilitar análisis
        df_selections = pd.DataFrame(self.selections)
        
        figures = {}
        
        # Gráfico 1: Distribución de posiciones por ronda
        plt.figure(figsize=(12, 8))
        position_round = pd.crosstab(df_selections['player_position'], df_selections['round'])
        position_round.plot(kind='bar', stacked=True)
        plt.title('Distribución de Posiciones por Ronda')
        plt.xlabel('Posición')
        plt.ylabel('Número de Jugadores')
        plt.legend(title='Ronda')
        plt.tight_layout()
        figures['posiciones_por_ronda'] = plt.gcf()
        
        # Gráfico 2: Distribución de Fit Scores
        plt.figure(figsize=(10, 6))
        sns.histplot(df_selections['fit_score'], bins=20, kde=True)
        plt.title('Distribución de Fit Scores')
        plt.xlabel('Fit Score')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        figures['distribucion_fit_scores'] = plt.gcf()
        
        # Gráfico 3: Ranking promedio por equipo (Top 10 equipos)
        plt.figure(figsize=(12, 8))
        team_avg_rank = df_selections.groupby('team')['player_rank'].mean().sort_values().head(10)
        team_avg_rank.plot(kind='bar')
        plt.title('Top 10 Equipos por Ranking Promedio de Selecciones')
        plt.xlabel('Equipo')
        plt.ylabel('Ranking Promedio')
        plt.tight_layout()
        figures['ranking_promedio_equipos'] = plt.gcf()
        
        return figures
    
    def modify_team_needs(self, team, position, new_value):
        """
        Modifica las necesidades de un equipo para una posición específica.
        
        Args:
            team (str): Nombre del equipo
            position (str): Posición a modificar
            new_value (int): Nuevo valor de necesidad (1-10)
        """
        if position in self.team_needs.columns:
            team_idx = self.team_needs[self.team_needs['TEAM'] == team].index
            if len(team_idx) > 0:
                self.team_needs.loc[team_idx[0], position] = new_value
                print(f"Necesidad de {team} para {position} actualizada a {new_value}")
            else:
                print(f"Equipo {team} no encontrado")
        else:
            print(f"Posición {position} no encontrada en el dataset")
    
    def export_results(self, filename="nfl_draft_results.csv"):
        """
        Exporta los resultados del draft a un archivo CSV.
        
        Args:
            filename (str): Nombre del archivo de salida
            
        Returns:
            bool: True si la exportación fue exitosa, False en caso contrario
        """
        if not self.selections:
            print("No hay draft simulado para exportar. Ejecute simulate_draft() primero.")
            return False
        
        try:
            df_selections = pd.DataFrame(self.selections)
            df_selections.to_csv(filename, index=False)
            print(f"Resultados del draft exportados exitosamente a {filename}")
            return True
        except Exception as e:
            print(f"Error al exportar resultados: {str(e)}")
            return False