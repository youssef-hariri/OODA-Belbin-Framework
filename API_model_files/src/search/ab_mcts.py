# src/search/ab_mcts.py

import math
import csv
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.agents.specialist_agent import SpecialistAgent
from src.solution_evaluator import SolutionEvaluator
from src.utils.logger import ExperimentLogger
from src.utils.code_executor import execute_solution

correction_log_lock = threading.Lock()

class MCTSNode:
    """Represents a node in the MCTS tree."""
    def __init__(self, solution_code: str, parent: Optional['MCTSNode'] = None, generating_agent_id: Optional[str] = None):
        self.solution_code = solution_code
        self.parent = parent
        self.generating_agent_id = generating_agent_id
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.official_score: int = 0
        self.gate_status: str = "UNKNOWN"

    def is_fully_expanded(self, num_agents: int) -> bool:
        return len(self.children) >= num_agents

    def select_best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        best_child, best_score = None, -1
        for child in self.children:
            if child.visits == 0:
                return child
            exploit_term = child.value / child.visits
            explore_term = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            ucb1_score = exploit_term + explore_term
            if ucb1_score > best_score:
                best_score, best_child = ucb1_score, child
        return best_child

class ABMCTS:
    """
    Implements a sequential Agent-Based Monte Carlo Tree Search.
    --- MODIFIED ---
    Now triggers the agent's meta-cognitive prompt after consecutive failures.
    """
    def __init__(self, team: List[SpecialistAgent], evaluator: SolutionEvaluator,
                 problem_id: str, arc_problem: Dict[str, Any],
                 dynamic_weights: Dict[str, float], condition_name: str,
                 budget: int, start_iteration: int, logger: ExperimentLogger,
                 early_stopping_enabled: bool = True,
                 max_correction_attempts: int = 2):
        self.team = team
        self.evaluator = evaluator
        self.problem_id = problem_id
        self.arc_problem = arc_problem
        self.total_budget = budget
        self.dynamic_weights = dynamic_weights
        self.start_iteration = start_iteration
        self.logger = logger
        self.early_stopping_enabled = early_stopping_enabled
        self.max_correction_attempts = max_correction_attempts
        self.root = MCTSNode(solution_code="", generating_agent_id="root")
        self.best_solution_found: Optional[str] = None
        self.best_score_found: float = 0.0
        self.best_official_score: int = 0
        self.best_gate_status: str = "UNKNOWN"
        self.api_calls_made = 0
        
        self.correction_log_path = self.logger.log_path.parent.parent / "correction_log.csv"
        self._setup_correction_log()

    def _setup_correction_log(self):
        with correction_log_lock:
            if not self.correction_log_path.is_file() or self.correction_log_path.stat().st_size == 0:
                with open(self.correction_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "problem_id", "mcts_iteration", "agent_id", "attempt_number",
                        "previous_gate_status", "new_gate_status", "previous_score",
                        "new_score", "score_improvement", "prompt"
                    ])

    def _log_correction_attempt(self, mcts_iteration: int, agent_id: str, attempt_num: int, prev_status: str, new_status: str, prev_score: float, new_score: float, prompt: str):
        with correction_log_lock:
            with open(self.correction_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.problem_id, mcts_iteration, agent_id, attempt_num,
                    prev_status, new_status, prev_score, new_score, new_score - prev_score,
                    prompt
                ])

    def search(self) -> Tuple[Optional[str], int, int, str]:
        self.logger.log(f"Starting MCTS search with a total budget of {self.total_budget} API calls.")
        
        mcts_iterations = 0
        while self.api_calls_made < self.total_budget:
            mcts_iterations += 1
            self.logger.log(f"--- MCTS Iteration {mcts_iterations} (API Calls: {self.api_calls_made}/{self.total_budget}) ---")
            
            leaf_node = self._select(self.root)
            
            if not leaf_node.is_fully_expanded(len(self.team)):
                new_node = self._expand_and_correct(leaf_node, mcts_iterations)
            else:
                new_node = leaf_node.select_best_child()

            if new_node:
                score_to_propagate = new_node.value
                self._backpropagate(new_node, score_to_propagate)

                if new_node.value > self.best_score_found:
                    self.best_score_found = new_node.value
                    self.best_solution_found = new_node.solution_code
                    self.best_official_score = new_node.official_score
                    self.best_gate_status = new_node.gate_status
                    self.logger.log(f"New best score found: {self.best_score_found:.4f} (Official: {self.best_official_score}, Gate: {self.best_gate_status})")

                if self.best_official_score == 1:
                    self.logger.log("SUCCESS! Official solution found. Ending search.")
                    break
        
        self.logger.log(f"Search finished. Total API calls made: {self.api_calls_made}")
        return self.best_solution_found, self.api_calls_made, self.best_official_score, self.best_gate_status
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        current_node = node
        while current_node.children:
            if not current_node.is_fully_expanded(len(self.team)):
                return current_node
            current_node = current_node.select_best_child()
        return current_node

    def _expand_and_correct(self, parent_node: MCTSNode, mcts_iteration: int) -> Optional[MCTSNode]:
        """
        --- MODIFIED ---
        Detects cognitive inertia and sets the 'is_stuck' flag for the final attempt.
        """
        agent_index = len(parent_node.children)
        agent_to_use = self.team[agent_index]
        self.logger.log(f"Expanding node with agent: {agent_to_use.persona_id}")

        current_code = None
        fitness_vector = None
        previous_gate_status = "INIT"
        previous_score = 0.0
        gate_status_history = []

        for i in range(self.max_correction_attempts + 1):
            if self.api_calls_made >= self.total_budget:
                self.logger.log("API budget exhausted during correction cycle. Halting expansion.")
                return None

            is_stuck = False
            # --- NEW: Meta-cognitive trigger logic ---
            # Check if this is the final attempt and if the last two statuses were the same failed gate
            if i == self.max_correction_attempts and len(gate_status_history) >= 2:
                if "FAILED" in gate_status_history[-1] and gate_status_history[-1] == gate_status_history[-2]:
                    is_stuck = True
                    self.logger.log(f"[Meta-Cognitive Trigger] Agent is stuck on '{gate_status_history[-1]}'. Engaging meta-cognitive prompt.")

            log_prefix = f"Correction {i}" if i > 0 else "Initial Attempt"
            self.logger.log(f"--- [{log_prefix}] API Call {self.api_calls_made + 1}/{self.total_budget} ---")

            self.api_calls_made += 1
            
            current_code, current_prompt = agent_to_use.generate_solution_attempt(
                self.arc_problem,
                previous_code=current_code,
                fitness_vector=fitness_vector,
                gate_status=previous_gate_status,
                is_stuck=is_stuck # Pass the flag to the agent
            )

            weighted_score, gate_status, fitness_vector, official_score = self._simulate(current_code)
            gate_status_history.append(gate_status)

            self._log_correction_attempt(
                mcts_iteration=mcts_iteration,
                agent_id=agent_to_use.persona_id,
                attempt_num=i + 1,
                prev_status=previous_gate_status,
                new_status=gate_status,
                prev_score=previous_score,
                new_score=weighted_score,
                prompt=current_prompt
            )

            previous_gate_status = gate_status
            previous_score = weighted_score

            if official_score == 1 or i == self.max_correction_attempts:
                new_node = MCTSNode(solution_code=current_code, parent=parent_node, generating_agent_id=agent_to_use.persona_id)
                new_node.value = weighted_score
                new_node.visits = 1
                new_node.gate_status = gate_status
                new_node.official_score = official_score
                parent_node.children.append(new_node)
                return new_node
        
        return None

    def _simulate(self, solution_code: str) -> Tuple[float, str, Dict, int]:
        if not solution_code:
            return 0.0, "NO_CODE", {}, 0
        
        test_case = self.arc_problem['test'][0]
        test_input = test_case['input']
        correct_output = test_case.get('output', [])
        
        if not correct_output:
            self.logger.log("Warning: No correct output found. Cannot evaluate.")
            return 0.0, "NO_SOLUTION_FILE", {}, 0
        
        fitness_vector = self.evaluator.get_fitness_vector(solution_code, test_input, correct_output)
        
        condition_dir = self.logger.log_path.parent
        
        weighted_score, gate_status = self.evaluator.evaluate_solution(
            solution_code=solution_code, test_input_grid=test_input,
            correct_output_grid=correct_output, problem_id=self.problem_id,
            agent_id="correction_cycle",
            attempt_num=self.api_calls_made,
            dynamic_weights=self.dynamic_weights, condition_dir=condition_dir, logger=self.logger
        )
        
        generated_output = execute_solution(solution_code, test_input)
        official_score = 1 if generated_output == correct_output else 0
        
        return weighted_score, gate_status, fitness_vector, official_score

    def _backpropagate(self, node: MCTSNode, score: float):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += score
            current_node = current_node.parent

