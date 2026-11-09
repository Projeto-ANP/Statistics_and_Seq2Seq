"""Gera as MCMs do CARTE usando a media das 10 execucoes (sementes 11-20)."""
import importlib.util
import os

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcm_carte.py")
spec = importlib.util.spec_from_file_location("mcm_carte_base", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ROOT = "/home/lucas/Documents/mestrado/Statistics_and_Seq2Seq"
m.CARTE_FOLDER = "orchestrator_react_consist_seed11_20_mean"
m.OUT_DIR = os.path.join(ROOT, "outputs", "resultados", "mcm_carte_mean10_png")
m.AUX_DIR = os.path.join(ROOT, "outputs", "resultados", "mcm_carte")

os.makedirs(m.OUT_DIR, exist_ok=True)
os.makedirs(m.AUX_DIR, exist_ok=True)
m.main()
