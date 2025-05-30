from openevolve import OpenEvolve
import asyncio
import os

async def run_open_evolve():
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # O lo setti altrove
    evolve = OpenEvolve(
        initial_program_path="model.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml"
    )
    best = await evolve.run(iterations=5)
    return best.metrics
