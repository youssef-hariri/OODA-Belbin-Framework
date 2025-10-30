# src/agents/specialist_agent.py

import re
import json
from typing import Dict, Any, Optional, Tuple
from src.llm_client import LLMClient
from src.prompts.personas import PERSONAS_DATA

class SpecialistAgent:
    """
    Represents a single LLM agent conditioned with a specific cultural persona.
    --- REFINED ---
    Prompts have been enhanced to combat cognitive inertia and the "whack-a-mole" problem.
    """

    def __init__(self, persona_id: str, llm_client: LLMClient):
        if persona_id not in PERSONAS_DATA:
            raise ValueError(f"Persona ID '{persona_id}' not found in PERSONAS_DATA.")
        
        self.persona_id = persona_id
        self.persona_data = PERSONAS_DATA[persona_id]
        self.llm_client = llm_client

    def generate_solution_attempt(self,
                                  arc_problem: Dict[str, Any],
                                  previous_code: Optional[str] = None,
                                  fitness_vector: Optional[Dict[str, float]] = None,
                                  gate_status: Optional[str] = None,
                                  is_stuck: bool = False) -> Tuple[str, str]:
        """
        Generates a solution attempt for a given ARC problem.
        Uses refined prompts for correction and meta-cognition.
        """
        if previous_code and fitness_vector and gate_status:
            if is_stuck:
                # Use the more forceful meta-cognitive prompt
                user_prompt = self._create_meta_cognitive_prompt(
                    arc_problem, previous_code, gate_status
                )
            else:
                # Use the chain-of-thought correction prompt
                user_prompt = self._create_correction_prompt(
                    arc_problem, previous_code, fitness_vector, gate_status
                )
        else:
            # This is the first attempt
            user_prompt = self._create_initial_prompt(arc_problem)

        persona_prompt = self.persona_data["description"]
        
        raw_response = self.llm_client.generate_response(
            user_prompt=user_prompt,
            persona_prompt=persona_prompt
        )
        
        solution_code = self._extract_python_code(raw_response)
        
        return solution_code, user_prompt

    def _extract_python_code(self, response_text: str) -> str:
        """Extracts Python code from a markdown-formatted string."""
        code_block_pattern = re.compile(r"```(?:python\n)?(.*?)```", re.DOTALL)
        match = code_block_pattern.search(response_text)
        
        if match:
            return match.group(1).strip()
        else:
            # Fallback for cases where the LLM might not use markdown
            return response_text.strip()

    def _create_initial_prompt(self, arc_problem: Dict[str, Any]) -> str:
        """Creates the initial, detailed prompt for the LLM to generate the first solution."""
        prompt = (
            "You are an expert AI programmer tasked with solving an Abstraction and Reasoning Corpus (ARC) problem.\n"
            "Your goal is to infer a general transformation rule from a few 'train' examples and apply it to a 'test' input.\n"
            "The grids are represented as 2D lists of integers from 0-9.\n"
            "You must write a single Python function named 'solve' that takes the test input grid (a list of lists) and returns the transformed output grid.\n"
            "Do not write any code outside of the 'solve' function. Do not include any example usage, print statements, or conversational text. Your entire response must be only the Python code inside a markdown block."
        )

        for i, train_pair in enumerate(arc_problem['train']):
            prompt += f"\n\n--- Train Example {i+1} ---\n"
            prompt += f"Input Grid:\n{json.dumps(train_pair['input'])}\n"
            prompt += f"Output Grid:\n{json.dumps(train_pair['output'])}\n"

        test_input = arc_problem['test'][0]['input']
        prompt += f"\n--- Test Case ---\n"
        prompt += f"Based on the training examples, apply the inferred rule to this test input grid:\n"
        prompt += f"Test Input Grid:\n{json.dumps(test_input)}\n\n"
        prompt += "Your task is to write the Python function `solve(grid)` that will produce the correct output grid for this test case."

        return prompt

    def _create_correction_prompt(self, arc_problem: Dict[str, Any], previous_code: str, fitness_vector: Dict[str, float], gate_status: str) -> str:
        """
        --- REFINED: Added Chain-of-Thought Step ---
        Creates a prompt that instructs the LLM to plan its debug strategy before coding.
        """
        diag_vector = {k: round(v, 4) for k, v in fitness_vector.items()}

        prompt = (
            "You are an expert AI programmer debugging a solution for an Abstraction and Reasoning Corpus (ARC) problem.\n"
            "Your previous attempt failed. You must analyze the provided error feedback and your previous code to generate a corrected solution.\n\n"
            f"--- YOUR PREVIOUS FAILED CODE ---\n"
            f"```python\n{previous_code}\n```\n\n"
            f"--- FAILURE ANALYSIS ---\n"
            f"Your code was executed and failed the evaluation. Here is the diagnostic feedback:\n"
            f"- Failure Status: {gate_status}\n"
            f"- Detailed Fitness Scores: {json.dumps(diag_vector)}\n\n"
            "A 'GATE_1_FAILED' status means your code produced the wrong number of objects.\n"
            "A 'GATE_2_FAILED' status means the objects had the wrong shape or structure.\n\n"
            "--- YOUR TASK ---\n"
            "1.  **Analyze the Failure**: In a few sentences, explain why your previous code failed based on the feedback.\n"
            "2.  **Formulate a Plan**: Outline your step-by-step plan to fix the code.\n"
            "3.  **Write the Corrected Code**: Provide the new, corrected Python function `solve(grid)`.\n\n"
            "Your entire response must follow this structure and place the final Python code inside a markdown block."
        )
        return prompt

    def _create_meta_cognitive_prompt(self, arc_problem: Dict[str, Any], previous_code: str, gate_status: str) -> str:
        """
        --- REFINED: More Forceful Instructions ---
        Creates a special prompt to force the agent to abandon a flawed strategy.
        """
        prompt = (
            "You are an expert AI programmer debugging a solution for an Abstraction and Reasoning Corpus (ARC) problem.\n"
            "**CRITICAL FEEDBACK:** Your previous attempts have failed with the same logical error. Your current approach is fundamentally flawed and must be abandoned.\n\n"
            f"--- FLAWED STRATEGY ---\n"
            f"Your previous code, shown below, failed with the error: {gate_status}.\n"
            f"```python\n{previous_code}\n```\n\n"
            "--- NEW INSTRUCTIONS ---\n"
            "**DO NOT** try to fix or modify your previous code. It is incorrect.\n"
            "You **MUST** abandon your current strategy and devise a completely new and different method for solving the problem.\n"
            "Re-examine the original problem description below and come up with a fresh, alternative approach.\n\n"
            f"--- ORIGINAL PROBLEM DESCRIPTION ---\n"
            f"{self._create_initial_prompt(arc_problem)}\n\n"
            "Your task is to write a completely new Python function `solve(grid)` based on a different line of reasoning.\n"
            "Your entire response must be only the Python code inside a markdown block."
        )
        return prompt

    def get_scores(self) -> Dict[str, float]:
        """Returns the creativity scores for this agent's persona."""
        return self.persona_data["scores"]

