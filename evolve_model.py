from openevolve import OpenEvolve
import asyncio

async def run_open_evolve():
    evolve = OpenEvolve(
        initial_program_path="evaluator.py",        # Usa build_model base
        evaluation_file="evaluator.py",             # Valutazione AMR
        config_path="config.yaml"                   # Configurazione
    )
    best = await evolve.run(iterations=5)
    return best.metrics
