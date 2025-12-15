#!/usr/bin/env python

import sys

def print_version(module_name, import_name=None):
    """
    Versucht, ein Modul zu importieren und seine Version auszugeben.
    """
    if import_name is None:
        import_name = module_name
    
    try:
        module = __import__(import_name)
        
        # Gängige Wege, die Version zu finden
        if hasattr(module, "__version__"):
            version = module.__version__
        elif hasattr(module, "VERSION"):
            version = module.VERSION
        else:
            # Spezieller Fall für ray.rllib (Teil von ray)
            if module_name.startswith("ray."):
                version = " (Teil von ray)"
            else:
                version = "Konnte __version__ nicht finden"
                
        print(f"{module_name:<15} {version}")
        
    except ImportError:
        print(f"{module_name:<15} NICHT INSTALLIERT")
    except Exception as e:
        print(f"{module_name:<15} FEHLER: {e}")

print("--- System ---")
print(f"{'Python':<15} {sys.version.split()[0]}")
print("\n--- Python Standard-Bibliothek (Built-in) ---")
print_version("os")
print_version("time")
print_version("pathlib")
print_version("random")
print_version("typing")

print("\n--- Externe Pakete ---")
print_version("numpy")
print_version("gymnasium")
print_version("gfootball")
print_version("ray")

# Diese sind Teil von Ray und haben keine eigene Version
print("\n--- Ray-Komponenten (Teil von 'ray') ---")
print_version("ray.rllib", "ray.rllib")
print_version("ray.tune", "ray.tune")