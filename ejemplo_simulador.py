import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import random
import os
from nfl_bayesian_fuzzy import NFLDraftSimulator, NFLBayesianNetwork, NFLFuzzySystem

def main():
    """
    Ejemplo de uso del simulador de draft de la NFL.
    Este script muestra las principales funcionalidades implementadas.
    """
    print("="*80)
    print("SIMULADOR INTELIGENTE DE DRAFT NFL CON MACHINE LEARNING, REDES BAYESIANAS Y LÓGICA DIFUSA")
    print("="*80)
    
    # Verificar la existencia de los archivos de datos
    files_to_check = [
        "prospects.csv",
        "team_needs.csv",
        "trends.csv",
        "draft_order.csv"
    ]
    
    missing_files = []
    for file in files_to_check:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\nERROR: No se encontraron los siguientes archivos necesarios:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPor favor, asegúrate de que estos archivos estén en el directorio actual.")
        return
    
    # Inicializar simulador
    print("\nInicializando simulador de draft de la NFL...")
    simulator = NFLDraftSimulator(
        prospects_file="prospects.csv",
        team_needs_file="team_needs.csv",
        trends_file="trends.csv",
        draft_order_file="draft_order.csv"
    )
    print("Simulador inicializado correctamente.")
    
    # Mostrar información sobre los sistemas de IA usados
    print("\n" + "="*50)
    print("COMPONENTES DE INTELIGENCIA ARTIFICIAL")
    print("="*50)
    
    print("\n1. RED BAYESIANA")
    print("La red bayesiana estima probabilidades basadas en datos históricos del draft.")
    print("Algunos ejemplos de probabilidades calculadas:")
    
    positions_to_show = ["QB", "WR", "OT", "CB", "EDGE"]
    for position in positions_to_show:
        for round_num in range(1, 4):
            prob = simulator.bayesian_network.get_position_round_probability(position, round_num)
            print(f"  - Probabilidad de que un {position} sea elegido en la ronda {round_num}: {prob:.2%}")
    
    print("\n2. SISTEMA DE LÓGICA DIFUSA")
    print("El sistema de lógica difusa calcula el 'fit score' de un jugador para un equipo.")
    print("Este score considera la necesidad del equipo, el ranking del jugador y las tendencias históricas.")
    print("A continuación, algunos ejemplos de cálculos de fit score:")
    
    examples = [
        {"team": "DAL", "position": "WR", "ranking": 10, "round": 1},
        {"team": "DAL", "position": "QB", "ranking": 5, "round": 1},
        {"team": "KC", "position": "CB", "ranking": 25, "round": 1},
        {"team": "GB", "position": "LB", "ranking": 150, "round": 3}
    ]
    
    for example in examples:
        team = example["team"]
        position = example["position"]
        ranking = example["ranking"]
        round_num = example["round"]
        
        need = simulator.get_team_needs(team, position)
        fit = simulator.calculate_fit_score(team, position, ranking, round_num)
        
        print(f"  - Equipo: {team}, Posición: {position}, Ranking: {ranking}, Ronda: {round_num}")
        print(f"    Necesidad: {need}/10, Fit Score: {fit:.2f}/100")
    
    # Demostrar la simulación del draft
    print("\n" + "="*50)
    print("SIMULACIÓN DEL DRAFT")
    print("="*50)
    
    # Preguntar si quiere simular todo el draft o solo una parte
    full_sim = input("\n¿Deseas simular el draft completo? (s/n, por defecto: n): ").lower() == 's'
    
    if not full_sim:
        num_picks = int(input("¿Cuántos picks deseas simular? (por defecto: 10): ") or 10)
        simulator.draft_order = simulator.draft_order.head(num_picks)
    
    print("\nSimulando el draft de la NFL...")
    selections = simulator.simulate_draft()
    print(f"\nSimulación completada. Se realizaron {len(selections)} selecciones.")
    
    # Mostrar análisis de las selecciones
    print("\n" + "="*50)
    print("ANÁLISIS DE RESULTADOS")
    print("="*50)
    
    # Convertir a DataFrame para facilitar el análisis
    df_selections = pd.DataFrame(selections)
    
    # 1. Estadísticas generales
    print("\n1. ESTADÍSTICAS GENERALES")
    print(f"Total de picks: {len(df_selections)}")
    print(f"Ranking promedio: {df_selections['player_rank'].mean():.2f}")
    print(f"Fit Score promedio: {df_selections['fit_score'].mean():.2f}")
    
    # 2. Top 5 picks con mejor fit score
    print("\n2. TOP 5 PICKS CON MEJOR FIT SCORE")
    top_fits = df_selections.sort_values(by='fit_score', ascending=False).head(5)
    for idx, pick in top_fits.iterrows():
        print(f"Pick #{pick['overall']}: {pick['team']} selecciona a {pick['player_name']} ({pick['player_position']})")
        print(f"  Ranking: {pick['player_rank']}, Fit Score: {pick['fit_score']:.2f}")
        print(f"  Razón: {pick['reasoning']}")
    
    # 3. Distribución de posiciones
    print("\n3. DISTRIBUCIÓN DE POSICIONES")
    pos_counts = df_selections['player_position'].value_counts()
    for pos, count in pos_counts.items():
        print(f"{pos}: {count} jugadores ({count/len(df_selections):.1%})")
    
    # Consultar selecciones de un equipo específico
    print("\n" + "="*50)
    print("CONSULTA DE EQUIPO ESPECÍFICO")
    print("="*50)
    
    team_options = df_selections['team'].unique()
    print(f"\nEquipos con selecciones en esta simulación: {', '.join(team_options)}")
    
    team_to_check = input("Ingresa el código del equipo a consultar (por ejemplo, DAL): ").upper()
    
    if team_to_check in team_options:
        team_picks = simulator.get_team_draft(team_to_check)
        print(f"\nSelecciones de {team_to_check}:")
        
        for pick in team_picks:
            print(f"Ronda {pick['round']}, Pick #{pick['overall']}: {pick['player_name']} ({pick['player_position']}, {pick['player_college']})")
            print(f"  Fit Score: {pick['fit_score']:.2f}")
            print(f"  Razón: {pick['reasoning']}")
            print()
    else:
        print(f"El equipo {team_to_check} no tiene selecciones en esta simulación.")
    
    # Modificar necesidades y simular de nuevo
    print("\n" + "="*50)
    print("MODIFICACIÓN DE NECESIDADES Y NUEVA SIMULACIÓN")
    print("="*50)
    
    modify = input("\n¿Deseas modificar las necesidades de un equipo y simular de nuevo? (s/n, por defecto: n): ").lower() == 's'
    
    if modify:
        # Elegir un equipo para modificar
        all_teams = simulator.team_needs['TEAM'].unique()
        print(f"\nEquipos disponibles: {', '.join(all_teams)}")
        
        team_to_modify = input("Ingresa el código del equipo a modificar: ").upper()
        
        if team_to_modify in all_teams:
            # Mostrar necesidades actuales
            team_row = simulator.team_needs[simulator.team_needs['TEAM'] == team_to_modify]
            positions = [col for col in simulator.team_needs.columns 
                        if col not in ['TEAM', 'MUST-HAVES', 'PARTIAL NEEDS', 'TOTAL PICKS', 'T-100 PICKS']]
            
            print(f"\nNecesidades actuales de {team_to_modify}:")
            for pos in positions:
                if pos in team_row.columns:
                    value = team_row[pos].values[0]
                    print(f"{pos}: {value}")
            
            # Elegir posición a modificar
            pos_to_modify = input("\nIngresa la posición a modificar (por ejemplo, QB): ").upper()
            
            if pos_to_modify in positions:
                current_value = team_row[pos_to_modify].values[0] if pos_to_modify in team_row.columns else 0
                new_value = int(input(f"Ingresa el nuevo valor de necesidad para {pos_to_modify} (1-10, actual: {current_value}): "))
                
                if 1 <= new_value <= 10:
                    # Modificar la necesidad
                    simulator.modify_team_needs(team_to_modify, pos_to_modify, new_value)
                    
                    # Volver a simular el draft
                    print("\nSimulando el draft con las necesidades modificadas...")
                    
                    # Reiniciar el orden del draft si se limitó anteriormente
                    if not full_sim:
                        simulator.draft_order = pd.read_csv("draft_order.csv").head(num_picks)
                    
                    new_selections = simulator.simulate_draft()
                    
                    # Comparar resultados
                    old_picks = [pick for pick in selections if pick['team'] == team_to_modify]
                    new_picks = [pick for pick in new_selections if pick['team'] == team_to_modify]
                    
                    print(f"\nComparación de selecciones para {team_to_modify}:")
                    
                    print("\nSelecciones ANTERIORES:")
                    for pick in old_picks:
                        print(f"Pick #{pick['overall']}: {pick['player_name']} ({pick['player_position']})")
                    
                    print("\nSelecciones NUEVAS:")
                    for pick in new_picks:
                        print(f"Pick #{pick['overall']}: {pick['player_name']} ({pick['player_position']})")
                        
                    # Verificar si cambió la selección de la posición modificada
                    old_pos_picks = [p for p in old_picks if p['player_position'] == pos_to_modify]
                    new_pos_picks = [p for p in new_picks if p['player_position'] == pos_to_modify]
                    
                    if len(old_pos_picks) != len(new_pos_picks):
                        print(f"\nEl cambio en la necesidad de {pos_to_modify} causó una diferencia en el número de selecciones en esa posición.")
                    
                    # Verificar cambios generales
                    if [p['player_name'] for p in old_picks] != [p['player_name'] for p in new_picks]:
                        print("\nLa modificación de necesidades resultó en cambios en las selecciones del equipo.")
                    else:
                        print("\nA pesar de la modificación, las selecciones del equipo no cambiaron.")
                else:
                    print("Valor no válido. Debe estar entre 1 y 10.")
            else:
                print(f"La posición {pos_to_modify} no existe en el dataset.")
        else:
            print(f"El equipo {team_to_modify} no existe en el dataset.")
    
    # Exportar resultados
    print("\n" + "="*50)
    print("EXPORTACIÓN DE RESULTADOS")
    print("="*50)
    
    export = input("\n¿Deseas exportar los resultados del draft a un archivo CSV? (s/n, por defecto: n): ").lower() == 's'
    
    if export:
        filename = input("Ingresa el nombre del archivo (por defecto: nfl_draft_results.csv): ") or "nfl_draft_results.csv"
        simulator.export_results(filename)
    
    print("\n" + "="*50)
    print("VISUALIZACIÓN DE RESULTADOS")
    print("="*50)
    
    visualize = input("\n¿Deseas visualizar los resultados con gráficos? (s/n, por defecto: n): ").lower() == 's'
    
    if visualize:
        figures = simulator.visualize_draft()
        print(f"Se generaron {len(figures)} visualizaciones.")
        print("Para mostrar los gráficos, ejecuta este script en un entorno que soporte matplotlib (Jupyter Notebook, Spyder, etc.)")
    
    print("\n" + "="*50)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*50)
    
    print("\nEjecuta 'streamlit run dashboard.py' para iniciar el dashboard interactivo.")
    print("Ejecuta 'python ejemplo_simulador.py' para repetir esta demostración.")


if __name__ == "__main__":
    main()