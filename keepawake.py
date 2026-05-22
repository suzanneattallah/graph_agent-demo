"""
keepawake.py - Empêche Windows de mettre l'ordi en veille.
Appelle SetThreadExecutionState toutes les 30s tant qu'il tourne.
"""
import ctypes
import time
import sys

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED

result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
print(f"[keepawake] Anti-veille activé (result={result})", flush=True)

try:
    while True:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        time.sleep(30)
except KeyboardInterrupt:
    # Restaure l'état normal à l'arrêt
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("[keepawake] Arrêté, veille restaurée.", flush=True)
    sys.exit(0)
