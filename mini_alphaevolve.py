import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

@dataclass
class Candidate:
    code: str
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    generation: int
    parent_id: str = None
    id: str = None

class EvolutionaryDatabase:
    def __init__(self, metrics: List[str], population_size: int = 100, num_islands: int = 3, diversity_metric: Callable = None):
        self.metrics = metrics
        self.population_size = population_size
        self.num_islands = num_islands
        self.diversity_metric = diversity_metric
        self.islands = [{} for _ in range(num_islands)]
        self.all_candidates = {}
        self.generation = 0

    def add_candidate(self, candidate: Candidate) -> str:
        import uuid
        candidate.id = str(uuid.uuid4())
        candidate.generation = self.generation
        self.all_candidates[candidate.id] = candidate
        island_idx = hash(candidate.id) % self.num_islands
        self.islands[island_idx][candidate.id] = candidate
        self._prune_island(island_idx)
        return candidate.id

    def get_best_candidates(self, metric: str, n: int = 1) -> List[Candidate]:
        sorted_candidates = sorted(
            self.all_candidates.values(),
            key=lambda c: c.scores.get(metric, float('-inf')),
            reverse=True
        )
        return sorted_candidates[:n]

    def sample_diverse_parents(self, n: int = 3) -> List[Candidate]:
        parents = []
        for island in self.islands:
            if island:
                candidates = list(island.values())
                scores = [sum(c.scores.values()) for c in candidates]
                softmax_scores = np.exp(scores) / np.sum(np.exp(scores))
                chosen_idx = np.random.choice(len(candidates), p=softmax_scores)
                parents.append(candidates[chosen_idx])
        if len(parents) > n:
            parents = np.random.choice(parents, size=n, replace=False).tolist()
        return parents

    def _prune_island(self, island_idx: int):
        island = self.islands[island_idx]
        if len(island) <= self.population_size:
            return
        candidates = list(island.values())
        aggregate_scores = []
        for candidate in candidates:
            normalized_scores = []
            for metric in self.metrics:
                if metric in candidate.scores:
                    metric_values = [c.scores.get(metric, 0) for c in candidates]
                    min_val = min(metric_values)
                    max_val = max(metric_values)
                    range_val = max_val - min_val
                    if range_val > 0:
                        normalized = (candidate.scores[metric] - min_val) / range_val
                    else:
                        normalized = 0.5
                    normalized_scores.append(normalized)
            if normalized_scores:
                aggregate_scores.append(sum(normalized_scores) / len(normalized_scores))
            else:
                aggregate_scores.append(0)
        sorted_indices = np.argsort(aggregate_scores)[::-1]
        keep_indices = sorted_indices[:self.population_size]
        new_island = {}
        for idx in keep_indices:
            candidate = candidates[idx]
            new_island[candidate.id] = candidate
        self.islands[island_idx] = new_island

    def increment_generation(self):
        self.generation += 1

import random

class LLMEnsemble:
    def __init__(self, models: Dict[str, Dict[str, Any]], sampling_strategy: str = "mixed"):
        self.models = models
        self.strategy = sampling_strategy
        total_weight = sum(model_info.get("weight", 1.0) for model_info in self.models.values())
        for name, info in self.models.items():
            info["normalized_weight"] = info.get("weight", 1.0) / total_weight

    def sample_model(self) -> str:
        if self.strategy == "quality":
            return max(self.models.items(), key=lambda x: x[1].get("quality", 0))[0]
        elif self.strategy == "speed":
            return min(self.models.items(), key=lambda x: x[1].get("latency", float('inf')))[0]
        else:
            rand = random.random()
            cumulative = 0
            for name, info in self.models.items():
                cumulative += info["normalized_weight"]
                if rand <= cumulative:
                    return name
            return list(self.models.keys())[0]

    async def generate_code(self, prompt: str, temperature: float = 0.7):
        import os
        import openai

        model_name = self.sample_model()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY non impostata nelle variabili d'ambiente.")

        openai.api_key = api_key

        if model_name in ["gpt-3.5-turbo", "gpt-4"]:
            # Nuova sintassi OpenAI >=1.0.0 (SOLO SINCRONA)
            response = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
                stream=False,
            )
            content = response.choices[0].message.content
            return content, {"model_used": model_name, "model_info": self.models[model_name]}
        else:
            raise NotImplementedError(f"Modello {model_name} non supportato per la chiamata API.")

class PromptSampler:
    def __init__(self, base_template: str, variants: Dict[str, List[str]] = None, max_examples: int = 3, max_context_length: int = 8000):
        self.base_template = base_template
        self.variants = variants or {}
        self.max_examples = max_examples
        self.max_context_length = max_context_length
        self.meta_prompts = []

    def build_prompt(self, parent_code: str, inspirations: List[Dict[str, Any]], eval_results: Dict[str, Any] = None, external_context: str = None) -> str:
        prompt_components = {}
        import random
        for key, variants in self.variants.items():
            prompt_components[key] = random.choice(variants)
        meta_instruction = ""
        if self.meta_prompts:
            meta_instruction = random.choice(self.meta_prompts)
        inspiration_text = ""
        for insp in inspirations[:self.max_examples]:
            code = insp.get("code", "")
            scores = insp.get("scores", {})
            output = insp.get("output", "")
            if len(code) > 1000:
                code = code[:500] + "\n...\n" + code[-500:]
            scores_text = ", ".join(f"{k}: {v:.3f}" for k, v in scores.items())
            inspiration_text += f"\n## Punteggio: {scores_text}\n```python\n{code}\n```\n"
            if output:
                inspiration_text += f"\n## Output:\n{output}\n"
        eval_text = ""
        if eval_results:
            eval_text = "## Risultati della valutazione:\n"
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    eval_text += f"{key}: {value:.4f}\n"
                else:
                    eval_text += f"{key}: {value}\n"
        prompt = self.base_template.format(
            parent_code=parent_code,
            inspirations=inspiration_text,
            evaluation=eval_text,
            external_context=external_context or "",
            meta_instruction=meta_instruction,
            **prompt_components
        )
        if len(prompt) > self.max_context_length:
            excess = len(prompt) - self.max_context_length
            inspiration_text = inspiration_text[:-excess]
            prompt = self.base_template.format(
                parent_code=parent_code,
                inspirations=inspiration_text,
                evaluation=eval_text,
                external_context=external_context or "",
                meta_instruction=meta_instruction,
                **prompt_components
            )
        return prompt

    def add_meta_prompt(self, prompt: str):
        self.meta_prompts.append(prompt)

    def evolve_meta_prompts(self, llm_fn, top_candidates: List[Dict[str, Any]]):
        examples = "\n\n".join([
            f"Candidato con punteggio {c.get('scores', {})}:\n{c.get('code', '')[:200]}..."
            for c in top_candidates[:3]
        ])
        meta_prompt_request = f"""
        Analizzando i seguenti candidati di successo:
        
        {examples}
        
        Genera 3 diverse istruzioni di meta-prompt che aiuterebbero un modello linguistico
        a creare candidati ancora migliori. Le istruzioni dovrebbero essere specifiche,
        chiare e focalizzate sul miglioramento delle prestazioni.
        
        Fornisci ogni meta-prompt come testo semplice, uno per riga.
        """
        response = llm_fn(meta_prompt_request)
        new_prompts = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                new_prompts.append(line)
        self.meta_prompts.extend(new_prompts[:3])
        if len(self.meta_prompts) > 10:
            self.meta_prompts = self.meta_prompts[-10:]
        return new_prompts

class DiffGenerator:
    @staticmethod
    def generate_diff_prompt(code: str) -> str:
        return f"""
        Analizza questo codice e suggerisci miglioramenti specifici:
        
        ```python
        {code}
        ```
        
        Fornisci le tue modifiche nel formato seguente:
        
        <<<<<<< SEARCH
        # Codice originale da sostituire
        =======
        # Nuovo codice che lo sostituisce
        >>>>>>> REPLACE
        
        Puoi fornire più blocchi di modifiche. Assicurati che il codice di ricerca (SEARCH) sia esattamente una parte del codice originale.
        """

    @staticmethod
    def apply_diff(original_code: str, diff: str) -> str:
        import re
        diff_pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
        diff_blocks = re.findall(diff_pattern, diff, re.DOTALL)
        result = original_code
        for search, replace in diff_blocks:
            search = search.strip()
            replace = replace.strip()
            result = result.replace(search, replace)
        return result

import concurrent.futures

class AsyncEvaluator:
    def __init__(self, evaluation_function: Callable, max_workers: int = 4):
        self.evaluate = evaluation_function
        self.max_workers = max_workers
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    async def evaluate_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        tasks = []
        for candidate in candidates:
            tasks.append(loop.run_in_executor(
                self.executor,
                self.evaluate,
                candidate
            ))
        results = await asyncio.gather(*tasks)
        return results

    async def evaluate_with_cascade(self, candidates: List[str], cascade_thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        initial_results = await self.evaluate_candidates(candidates)
        passing_candidates = []
        passing_indices = []
        for i, result in enumerate(initial_results):
            if result.get('quick_score', 0) >= cascade_thresholds.get('quick_score', 0):
                passing_candidates.append(candidates[i])
                passing_indices.append(i)
        if not passing_candidates:
            return initial_results
        final_results = await self.evaluate_candidates(passing_candidates)
        for i, result in zip(passing_indices, final_results):
            initial_results[i] = result
        return initial_results

class MiniAlphaEvolve:
    def __init__(
        self,
        evaluation_function: Callable,
        llm_ensemble,
        db,
        prompt_sampler,
        diff_generator,
        async_evaluator,
        max_iterations: int = 20,
        parallel_evaluations: int = 4,
        log_level: str = "INFO"
    ):
        self.evaluate = evaluation_function
        self.llm_ensemble = llm_ensemble
        self.db = db
        self.prompt_sampler = prompt_sampler
        self.diff_generator = diff_generator
        self.async_evaluator = async_evaluator
        self.max_iterations = max_iterations
        self.parallel_evaluations = parallel_evaluations

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MiniAlphaEvolve")

    async def run(self, initial_code: str, metrics: List[str]) -> Dict[str, Any]:
        self.logger.info("Valutazione del codice iniziale")
        initial_result = await self.async_evaluator.evaluate_candidates([initial_code])
        initial_scores = initial_result[0]

        initial_candidate = Candidate(
            code=initial_code,
            scores=initial_scores,
            metadata={"source": "initial"},
            generation=0
        )
        self.db.add_candidate(initial_candidate)

        iteration = 0
        best_scores = {m: initial_scores.get(m, 0) for m in metrics}
        best_candidate = initial_candidate

        self.logger.info(f"Punteggi iniziali: {best_scores}")

        while iteration < self.max_iterations:
            self.logger.info(f"Iterazione {iteration+1}/{self.max_iterations}")
            self.db.increment_generation()
            new_candidates = []
            tasks = []
            for _ in range(self.parallel_evaluations):
                tasks.append(self.generate_candidate())
            batch_candidates = await asyncio.gather(*tasks)
            new_candidates.extend([c for c in batch_candidates if c is not None])

            if not new_candidates:
                self.logger.warning("Nessun nuovo candidato generato in questa iterazione")
                iteration += 1
                continue

            self.logger.info(f"Valutazione di {len(new_candidates)} candidati")
            candidate_codes = [c.code for c in new_candidates]
            results = await self.async_evaluator.evaluate_candidates(candidate_codes)

            for candidate, scores in zip(new_candidates, results):
                candidate.scores = scores
                self.db.add_candidate(candidate)
                for metric in metrics:
                    if metric in scores and scores[metric] > best_scores.get(metric, float('-inf')):
                        best_scores[metric] = scores[metric]
                        best_candidate = candidate
                        self.logger.info(f"Nuovo miglior punteggio per {metric}: {scores[metric]}")

            iteration += 1

        return {
            "code": best_candidate.code,
            "scores": best_candidate.scores,
            "generation": best_candidate.generation,
            "id": best_candidate.id
        }

    async def generate_candidate(self):
        # 1. Campiona genitori dal database
        parents = self.db.sample_diverse_parents(n=1)
        if not parents:
            self.logger.warning("Nessun genitore disponibile")
            return None
        parent = parents[0]

        # 2. Campiona ispirazioni (altri candidati top diversi dal genitore)
        best_candidates = []
        for metric in self.db.metrics:
            best_candidates.extend(self.db.get_best_candidates(metric, n=2))
        inspirations = []
        for candidate in best_candidates:
            if candidate.id != parent.id:
                inspirations.append({
                    "code": candidate.code,
                    "scores": candidate.scores,
                    "output": candidate.metadata.get("output", "")
                })

        # 3. Costruisci il prompt
        prompt = self.prompt_sampler.build_prompt(
            parent_code=parent.code,
            inspirations=inspirations,
            eval_results=parent.scores
        )

        # 4. Genera il diff usando l'ensemble LLM
        diff_response, model_info = await self.llm_ensemble.generate_code(prompt)

        # 5. Applica il diff al codice del genitore
        try:
            new_code = self.diff_generator.apply_diff(parent.code, diff_response)
        except Exception as e:
            self.logger.warning(f"Errore nell'applicazione del diff: {e}")
            return None

        # 6. Crea il nuovo candidato
        new_candidate = Candidate(
            code=new_code,
            scores={},
            metadata={"parent_id": parent.id, "model_used": model_info.get("model_used")},
            generation=self.db.generation
        )
        return new_candidate
