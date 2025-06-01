import os
import asyncio

from mini_alphaevolve import (
    MiniAlphaEvolve,
    EvolutionaryDatabase,
    LLMEnsemble,
    PromptSampler,
    DiffGenerator,
    AsyncEvaluator,
    Candidate
)
from evaluator import evaluate

async def run_mini_alpha_evolve():
    with open("model.py", "r") as f:
        initial_code = f.read()

    metrics = ["auc"]

    db = EvolutionaryDatabase(
        metrics=metrics,
        population_size=3,
        num_islands=1
    )

    llm_ensemble = LLMEnsemble(models={
        "gpt-4": {"quality": 10, "latency": 3, "weight": 1.0}
    })

    base_prompt = """
## Codice padre
```python
{parent_code}
```

## Ispirazioni
{inspirations}

## Valutazione precedente
{evaluation}

{external_context}

{meta_instruction}

Apporta solo modifiche migliorative nel seguente formato diff:

<<<<<<< SEARCH
# codice da sostituire
=======
# nuovo codice migliorato
>>>>>>> REPLACE
"""

    prompt_sampler = PromptSampler(
        base_template=base_prompt,
        variants={},
        max_examples=3,
        max_context_length=8000
    )

    diff_generator = DiffGenerator()

    async_evaluator = AsyncEvaluator(
        evaluation_function=evaluate,
        max_workers=2
    )

    evolve = MiniAlphaEvolve(
        evaluation_function=evaluate,
        llm_ensemble=llm_ensemble,
        db=db,
        prompt_sampler=prompt_sampler,
        diff_generator=diff_generator,
        async_evaluator=async_evaluator,
        max_iterations=5,
        parallel_evaluations=2,
        log_level="INFO"
    )

    result = await evolve.run(
        initial_code=initial_code,
        metrics=metrics
    )

    # Salva il miglior codice trovato (opzionale)
    best_code_path = "best_model_evolved.py"
    with open(best_code_path, "w") as out_f:
        out_f.write(result["code"])

    return result["scores"]
