import asyncio
from openevolve import OpenEvolve
import os
from streamlit import secrets

async def run_open_evolve():
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

    evolve = OpenEvolve(
        initial_program_path="evaluator.py",
        evaluation_file="evaluator.py",
        llm_config={"api_key": secrets["OPENAI_API_KEY"]},  # ✅ niente config.yaml
        evolution_config={
            "iterations": 5,
            "population_size": 3,
            "temperature": 0.7,
            "top_k": 1,
            "mutate_top_k": 2,
            "crossover_top_k": 1
        },
        evaluation_config={
            "maximize": "auc"
        }
    )

    best = await evolve.run()
    return best.metrics
