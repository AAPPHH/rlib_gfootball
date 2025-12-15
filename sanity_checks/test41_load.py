"""
Google Football Dump Analyzer
Lädt und analysiert die Dump-Dateien von Google Football
"""

import pickle
import json
from pathlib import Path
import numpy as np
from pprint import pprint
import sys

def analyze_dump(dump_file_path):
    """Analysiert eine einzelne Dump-Datei"""
    
    print(f"\n{'='*80}")
    print(f"Analysiere Dump: {dump_file_path.name}")
    print(f"{'='*80}")
    
    try:
        # Versuche die Datei zu laden
        with open(dump_file_path, 'rb') as f:
            dump_data = pickle.load(f)
    except Exception as e:
        print(f"Fehler beim Laden der Dump-Datei: {e}")
        return
    
    # Zeige die Hauptstruktur
    print("\n1. HAUPTSTRUKTUR:")
    print(f"   Typ: {type(dump_data)}")
    
    if isinstance(dump_data, dict):
        print(f"   Schlüssel: {list(dump_data.keys())}")
        print()
        
        # Analysiere jeden Schlüssel
        for key, value in dump_data.items():
            print(f"\n2. ANALYSE VON '{key}':")
            print(f"   Typ: {type(value)}")
            
            if isinstance(value, list):
                print(f"   Anzahl Einträge: {len(value)}")
                if len(value) > 0:
                    print(f"   Typ des ersten Eintrags: {type(value[0])}")
                    
                    # Wenn es sich um Steps/Frames handelt
                    if key in ['steps', 'frames']:
                        print(f"\n   Beispiel-Frame (erstes Element):")
                        analyze_frame(value[0])
                        
                        if len(value) > 1:
                            print(f"\n   Letztes Frame:")
                            analyze_frame(value[-1])
                    
            elif isinstance(value, dict):
                print(f"   Schlüssel: {list(value.keys())[:10]}")  # Erste 10 Schlüssel
                if len(value) > 10:
                    print(f"   ... und {len(value) - 10} weitere")
                    
            elif isinstance(value, (str, int, float, bool)):
                print(f"   Wert: {value}")
            
            elif isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Min/Max: {value.min():.4f} / {value.max():.4f}")
    
    elif isinstance(dump_data, list):
        print(f"   Anzahl Frames: {len(dump_data)}")
        if len(dump_data) > 0:
            print("\n   Erstes Frame:")
            analyze_frame(dump_data[0])
            
            print("\n   Letztes Frame:")
            analyze_frame(dump_data[-1])

def analyze_frame(frame):
    """Analysiert ein einzelnes Frame/Step"""
    
    if isinstance(frame, dict):
        print("   Frame-Struktur (Dictionary):")
        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                print(f"     - {key}: ndarray mit Shape {value.shape}, dtype={value.dtype}")
                # Zeige ein paar Beispielwerte
                if value.size < 20:
                    print(f"       Werte: {value.flatten()[:20]}")
                else:
                    print(f"       Erste 5 Werte: {value.flatten()[:5]}")
            elif isinstance(value, list):
                print(f"     - {key}: Liste mit {len(value)} Elementen")
                if len(value) > 0 and len(value) < 5:
                    print(f"       Werte: {value}")
                elif len(value) >= 5:
                    print(f"       Erste 3 Werte: {value[:3]}")
            elif isinstance(value, dict):
                print(f"     - {key}: Dictionary mit Schlüsseln: {list(value.keys())[:5]}")
            else:
                print(f"     - {key}: {type(value).__name__} = {str(value)[:100]}")
    
    elif isinstance(frame, tuple):
        print(f"   Frame-Struktur (Tuple mit {len(frame)} Elementen):")
        for i, element in enumerate(frame):
            if isinstance(element, np.ndarray):
                print(f"     Element {i}: ndarray Shape {element.shape}")
            else:
                print(f"     Element {i}: {type(element).__name__}")
    
    elif isinstance(frame, np.ndarray):
        print(f"   Frame ist ein numpy Array:")
        print(f"     Shape: {frame.shape}")
        print(f"     Dtype: {frame.dtype}")
        print(f"     Min/Max: {frame.min():.4f} / {frame.max():.4f}")
        print(f"     Erste 10 Werte: {frame.flatten()[:10]}")
    
    else:
        print(f"   Frame-Typ: {type(frame)}")

def find_dump_files(directory):
    """Findet alle Dump-Dateien im angegebenen Verzeichnis"""
    
    dump_dir = Path(directory)
    if not dump_dir.exists():
        print(f"Verzeichnis {dump_dir} existiert nicht!")
        return []
    
    # Suche nach .dump Dateien
    dump_files = list(dump_dir.glob("*.dump"))
    
    # Falls keine .dump Dateien, suche nach .pkl Dateien
    if not dump_files:
        dump_files = list(dump_dir.glob("*.pkl"))
    
    # Suche auch in Unterverzeichnissen
    if not dump_files:
        dump_files = list(dump_dir.rglob("*.dump"))
        if not dump_files:
            dump_files = list(dump_dir.rglob("*.pkl"))
    
    return sorted(dump_files)

def analyze_observation_structure(obs):
    """Analysiert die Struktur einer Observation (115-dimensional vector)"""
    
    print("\n3. OBSERVATION STRUKTUR (simple115v2):")
    print(f"   Gesamtlänge: {len(obs) if isinstance(obs, (list, np.ndarray)) else 'N/A'}")
    
    if isinstance(obs, (list, np.ndarray)) and len(obs) == 115:
        # Die bekannte Struktur von simple115v2
        indices = {
            "Ball position": (0, 3),
            "Ball direction": (3, 6),
            "Ball rotation": (6, 9),
            "Ball ownership": (9, 12),
            "Active player pos": (12, 14),
            "Game mode": (14, 21),
            "Own team positions": (21, 43),  # 11 Spieler * 2 Koordinaten
            "Own team directions": (43, 65),  # 11 Spieler * 2 Richtungen
            "Own team tired factors": (65, 76),  # 11 Spieler
            "Own team yellow cards": (76, 87),  # 11 Spieler
            "Own team active": (87, 88),
            "Own team roles": (88, 99),  # 11 Spieler
            "Opponent positions": (99, 115),  # 8 Spieler * 2 Koordinaten (sichtbar)
        }
        
        print("\n   Detaillierte Aufteilung:")
        for name, (start, end) in indices.items():
            values = obs[start:end] if isinstance(obs, list) else obs[start:end]
            print(f"   [{start:3d}-{end:3d}] {name:25s}: {values[:5]}..." if len(values) > 5 else f"   [{start:3d}-{end:3d}] {name:25s}: {values}")

def main():
    # Standard-Pfad oder über Kommandozeile
    if len(sys.argv) > 1:
        dump_dir = sys.argv[1]
    else:
        dump_dir = "gfootball_dumps"
    
    print(f"Suche Dump-Dateien in: {dump_dir}")
    
    dump_files = find_dump_files(dump_dir)
    
    if not dump_files:
        print(f"\nKeine Dump-Dateien gefunden in {dump_dir}")
        print("Stelle sicher, dass das Collect-Script erfolgreich ausgeführt wurde.")
        return
    
    print(f"\nGefundene Dump-Dateien: {len(dump_files)}")
    for i, file in enumerate(dump_files, 1):
        print(f"  {i}. {file.name}")
    
    # Analysiere jede Dump-Datei
    for dump_file in dump_files:
        analyze_dump(dump_file)
        
        # Optionale detaillierte Analyse
        print("\n" + "="*40)
        response = input("Möchtest du eine detailliertere Analyse dieser Datei? (j/n): ")
        if response.lower() == 'j':
            analyze_detailed(dump_file)

def analyze_detailed(dump_file_path):
    """Führt eine detaillierte Analyse durch"""
    
    with open(dump_file_path, 'rb') as f:
        dump_data = pickle.load(f)
    
    # Spezielle Behandlung für episode_done dumps
    if isinstance(dump_data, dict) and 'observation' in dump_data and isinstance(dump_data['observation'], dict):
        print("\n=== EPISODE-END DUMP ANALYSE ===")
        obs_data = dump_data['observation']
        
        print("\n1. BALL INFORMATION:")
        if 'ball' in obs_data:
            print(f"   Position: {obs_data['ball']}")
        if 'ball_direction' in obs_data:
            print(f"   Richtung: {obs_data['ball_direction']}")
        if 'ball_rotation' in obs_data:
            print(f"   Rotation: {obs_data['ball_rotation']}")
        if 'ball_owned_team' in obs_data:
            print(f"   Ballbesitz Team: {obs_data['ball_owned_team']} (-1=keiner, 0=links, 1=rechts)")
        if 'ball_owned_player' in obs_data:
            print(f"   Ballbesitz Spieler: {obs_data['ball_owned_player']}")
        
        print("\n2. LINKES TEAM (Spieler-kontrolliert):")
        if 'left_team' in obs_data:
            positions = obs_data['left_team']
            print(f"   Anzahl Spieler: {len(positions)}")
            for i, pos in enumerate(positions[:3]):  # Erste 3 Spieler
                print(f"   Spieler {i}: Position {pos}")
        if 'left_team_direction' in obs_data:
            print(f"   Bewegungsrichtungen: {obs_data['left_team_direction'][:3]}...")
        if 'left_team_tired_factor' in obs_data:
            print(f"   Müdigkeit: {obs_data['left_team_tired_factor'][:3]}...")
        if 'left_team_yellow_card' in obs_data:
            print(f"   Gelbe Karten: {obs_data['left_team_yellow_card']}")
        if 'left_team_roles' in obs_data:
            role_names = ["GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF", "LW", "RW"]
            roles = obs_data['left_team_roles']
            print(f"   Rollen: ", end="")
            for i, role in enumerate(roles[:5]):
                if 0 <= role < len(role_names):
                    print(f"{role_names[role]}", end=" ")
                else:
                    print(f"{role}", end=" ")
            print("...")
        
        print("\n3. RECHTES TEAM (Gegner):")
        if 'right_team' in obs_data:
            positions = obs_data['right_team']
            print(f"   Anzahl Spieler: {len(positions)}")
            for i, pos in enumerate(positions[:3]):
                print(f"   Spieler {i}: Position {pos}")
        if 'right_team_direction' in obs_data:
            print(f"   Bewegungsrichtungen: {obs_data['right_team_direction'][:3]}...")
        
        print("\n4. SPIEL-STATUS:")
        if 'score' in obs_data:
            print(f"   Spielstand: {obs_data['score']}")
        if 'steps_left' in obs_data:
            print(f"   Verbleibende Steps: {obs_data['steps_left']}")
        if 'game_mode' in obs_data:
            game_modes = ["Normal", "KickOff", "GoalKick", "FreeKick", "Corner", "ThrowIn", "Penalty"]
            mode = obs_data['game_mode']
            if 0 <= mode < len(game_modes):
                print(f"   Spielmodus: {game_modes[mode]} ({mode})")
            else:
                print(f"   Spielmodus: {mode}")
        
        print("\n5. DEBUG INFORMATION:")
        if 'debug' in dump_data:
            debug = dump_data['debug']
            if 'action' in debug:
                print(f"   Letzte Aktionen: {debug['action']}")
            if 'frame_cnt' in debug:
                print(f"   Frame Count: {debug['frame_cnt']}")
            if 'time' in debug:
                print(f"   Zeit: {debug['time']}")
            if 'config' in debug:
                config = debug['config']
                print(f"   Konfiguration:")
                for key in ['action_set', 'players', 'offsides', 'game_difficulty']:
                    if key in config:
                        print(f"     - {key}: {config[key]}")
        
        print("\n6. REWARDS:")
        print(f"   Reward diese Runde: {dump_data.get('reward', 'N/A')}")
        print(f"   Kumulative Reward: {dump_data.get('cumulative_reward', 'N/A')}")
        
        # Konvertiere zu simple115v2 wenn möglich
        if all(key in obs_data for key in ['ball', 'left_team', 'right_team']):
            print("\n7. SIMPLE115V2 REPRESENTATION:")
            try:
                # Erstelle eine vereinfachte Darstellung
                simple_obs = []
                # Ball (3+3+3=9 values)
                simple_obs.extend(obs_data['ball'])
                if 'ball_direction' in obs_data:
                    simple_obs.extend(obs_data['ball_direction'])
                if 'ball_rotation' in obs_data:
                    simple_obs.extend(obs_data['ball_rotation'])
                # ... weitere Konvertierung
                print(f"   Konvertierte Länge: {len(simple_obs)} (von 115 erwartet)")
            except:
                print("   Konvertierung nicht möglich")
    
    # Original Code für andere Dump-Typen
    elif isinstance(dump_data, dict):
        if 'steps' in dump_data and len(dump_data['steps']) > 0:
            # Analysiere die Observations
            first_step = dump_data['steps'][0]
            if 'observation' in first_step:
                obs = first_step['observation']
                if isinstance(obs, list) and len(obs) > 0:
                    # Für Multi-Agent (2 Spieler)
                    print("\n=== MULTI-AGENT OBSERVATIONS ===")
                    for i, agent_obs in enumerate(obs):
                        print(f"\nAgent {i} Observation:")
                        if isinstance(agent_obs, (list, np.ndarray)):
                            analyze_observation_structure(agent_obs)
                elif isinstance(obs, (list, np.ndarray)):
                    analyze_observation_structure(obs)
            
            # Zeige Aktionen
            if 'action' in first_step:
                print("\n=== AKTIONEN ===")
                actions = first_step['action']
                print(f"Aktionen im ersten Step: {actions}")
                print("Mögliche Aktionen in GFootball:")
                action_names = [
                    "idle", "left", "top_left", "top", "top_right", "right",
                    "bottom_right", "bottom", "bottom_left", "long_pass",
                    "high_pass", "short_pass", "shot", "sprint", "release_direction",
                    "release_sprint", "sliding", "dribble", "release_dribble"
                ]
                if isinstance(actions, list):
                    for i, action in enumerate(actions):
                        if isinstance(action, (int, np.integer)) and 0 <= action < len(action_names):
                            print(f"  Agent {i}: {action} = {action_names[action]}")
                elif isinstance(actions, (int, np.integer)) and 0 <= actions < len(action_names):
                    print(f"  Aktion: {actions} = {action_names[actions]}")
            
            # Statistiken über das gesamte Spiel
            if 'steps' in dump_data:
                print(f"\n=== SPIELSTATISTIKEN ===")
                print(f"Gesamtanzahl Steps: {len(dump_data['steps'])}")
                
                # Sammle Rewards
                rewards = []
                for step in dump_data['steps']:
                    if 'reward' in step:
                        rewards.append(step['reward'])
                
                if rewards:
                    rewards_array = np.array(rewards)
                    print(f"Rewards - Min: {rewards_array.min()}, Max: {rewards_array.max()}, Summe: {rewards_array.sum()}")
                    
                    # Zeige wann Tore gefallen sind
                    goal_steps = np.where(rewards_array != 0)[0]
                    if len(goal_steps) > 0:
                        print(f"Tore/Punkte bei Steps: {goal_steps.tolist()}")

if __name__ == "__main__":
    main()