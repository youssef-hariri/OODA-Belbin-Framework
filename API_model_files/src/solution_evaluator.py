# src/solution_evaluator.py

import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import time
import os

from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, euler_number
from src.utils.logger import ExperimentLogger
from src.utils.code_executor import execute_solution

class SolutionEvaluator:
    """
    Evaluates solution quality using a multi-objective fitness function.
    --- MODIFIED ---
    Now includes a Gated Scoring Framework and returns a status string
    for detailed data collection.
    """
    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def get_fitness_vector(self, solution_code: str, test_input_grid: List[List[int]], correct_output_grid: List[List[int]]) -> Dict[str, float]:
        """Executes code and returns the full diagnostic fitness vector."""
        generated_output = self._execute_sandboxed_code(solution_code, test_input_grid)
        
        failure_vector = { "pixel_accuracy": 0.0, "ssim_discrete": 0.0, "hist_bc": 0.0, "obj_count": 0.0, "jaccard_objects": 0.0, "symmetry": 0.0, "euler": 0.0, "graph_spectral": 0.0, "prop_obj_count": 0.0, "truth_obj_count": 0.0 }
        
        if generated_output is None: return failure_vector
        try:
            prop_grid = np.array(generated_output, dtype=int)
            truth_grid = np.array(correct_output_grid, dtype=int)
        except (TypeError, ValueError): return failure_vector

        if prop_grid.ndim != 2:
            return failure_vector

        return self._calculate_fitness_vector(prop_grid, truth_grid)

    def evaluate_solution(self, solution_code: str, test_input_grid: List[List[int]],
                          correct_output_grid: List[List[int]], problem_id: str,
                          agent_id: str, attempt_num: int, dynamic_weights: Dict[str, float],
                          condition_dir: Path, logger: Optional[ExperimentLogger] = None) -> Tuple[float, str]:
        """
        Evaluates a solution and returns a tuple containing:
        1. The final weighted score (float).
        2. A status string indicating if a gate was failed (str).
        """
        log = logger.log if logger else print
        
        self._log_solution_code(condition_dir, problem_id, agent_id, attempt_num, solution_code)
        
        fitness_vector = self.get_fitness_vector(solution_code, test_input_grid, correct_output_grid)
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +                 NEW: GATED SCORING FRAMEWORK                      +
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # This preemptive check aggressively penalizes logically flawed solutions
        # to prevent the MCTS algorithm from getting stuck in local optima.

        # Gate 1: Object Count Integrity
        if fitness_vector.get('obj_count', 0) < 0.5:
            log(f"[Gated Scoring] FAILED Gate 1 (Object Count: {fitness_vector.get('obj_count', 0):.2f} < 0.5). Applying penalty.")
            penalized_score = 0.1 * fitness_vector.get('pixel_accuracy', 0)
            return penalized_score, "GATE_1_FAILED"

        # Gate 2: Object Structure Integrity
        if fitness_vector.get('jaccard_objects', 0) < 0.6:
            log(f"[Gated Scoring] FAILED Gate 2 (Jaccard Objects: {fitness_vector.get('jaccard_objects', 0):.2f} < 0.6). Applying penalty.")
            penalized_score = 0.1 * fitness_vector.get('pixel_accuracy', 0)
            return penalized_score, "GATE_2_FAILED"
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +                  ORIGINAL WEIGHTED SUMMATION                      +
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # This part only runs if the solution passes both critical gates.
        
        score_metric_keys = [
            "pixel_accuracy", "ssim_discrete", "hist_bc", "obj_count",
            "jaccard_objects", "symmetry", "euler", "graph_spectral"
        ]
        
        numerator = sum(dynamic_weights.get(key, 1.0) * fitness_vector.get(key, 0.0) for key in score_metric_keys)
        denominator = sum(dynamic_weights.get(key, 1.0) for key in score_metric_keys)
        
        final_score = numerator / denominator if denominator > 0 else 0.0
        
        log(f"Evaluation complete. Final weighted score: {final_score:.4f}")
        return final_score, "PASSED"

    def _calculate_fitness_vector(self, prop_grid: np.ndarray, truth_grid: np.ndarray) -> Dict[str, float]:
        if prop_grid.shape != truth_grid.shape:
            h_t, w_t = truth_grid.shape; h_p, w_p = prop_grid.shape; resized_prop = np.zeros_like(truth_grid)
            h_min, w_min = min(h_t, h_p), min(w_t, w_p); resized_prop[:h_min, :w_min] = prop_grid[:h_min, :w_min]; prop_grid = resized_prop
        
        f_pixel_accuracy = np.mean(prop_grid == truth_grid)
        min_dim = min(prop_grid.shape); f_ssim_discrete = f_pixel_accuracy if min_dim < 7 else ssim(prop_grid, truth_grid, data_range=9, win_size=min_dim if min_dim % 2 != 0 else min_dim - 1)
        f_hist_bc = self._calculate_histogram_similarity(prop_grid, truth_grid)
        
        try:
            prop_objects, prop_obj_count = label(prop_grid > 0, connectivity=1, return_num=True)
            truth_objects, truth_obj_count = label(truth_grid > 0, connectivity=1, return_num=True)
            f_obj_count = max(0, 1 - abs(prop_obj_count - truth_obj_count) / max(1, truth_obj_count))
            f_jaccard_objects = self._calculate_object_jaccard(prop_objects > 0, truth_objects > 0)
        except Exception:
            prop_obj_count, truth_obj_count = 0, 0
            f_obj_count, f_jaccard_objects = 0.0, 0.0

        f_symmetry = self._calculate_symmetry_congruence(prop_grid, truth_grid); f_euler = self._calculate_euler_fitness(prop_grid, truth_grid); f_graph_spectral = 0.5
        
        return {
            "pixel_accuracy": f_pixel_accuracy, "ssim_discrete": f_ssim_discrete, "hist_bc": f_hist_bc,
            "obj_count": f_obj_count, "jaccard_objects": f_jaccard_objects, "symmetry": f_symmetry,
            "euler": f_euler, "graph_spectral": f_graph_spectral,
            "prop_obj_count": float(prop_obj_count), "truth_obj_count": float(truth_obj_count)
        }

    def _calculate_histogram_similarity(self, g1, g2): return np.sum(np.sqrt((np.histogram(g1, bins=10, range=(0, 9))[0]/max(1,g1.size)) * (np.histogram(g2, bins=10, range=(0, 9))[0]/max(1,g2.size))))
    def _calculate_object_jaccard(self, m1, m2): return np.sum(m1 & m2) / max(1, np.sum(m1 | m2))
    def _calculate_symmetry_congruence(self, g1, g2): s1 = {np.array_equal(g1, np.fliplr(g1)), np.array_equal(g1, np.flipud(g1))}; s2 = {np.array_equal(g2, np.fliplr(g2)), np.array_equal(g2, np.flipud(g2))}; return 1.0 if s1 == s2 else 0.0
    def _calculate_euler_fitness(self, g1, g2): colors = [c for c in np.unique(np.concatenate((g1.flatten(), g2.flatten()))) if c > 0]; return 1.0 if not colors else sum(1.0 for c in colors if euler_number(g1 == c) == euler_number(g2 == c)) / len(colors)
    
    def _log_solution_code(self, condition_dir: Path, problem_id: str, agent_id: str, attempt_num: int, code: str):
        """
        Logs the code for each attempt with a retry mechanism to handle
        "Too many open files" errors gracefully.
        """
        max_retries = 20
        base_delay = 2

        for attempt in range(max_retries):
            try:
                code_log_dir = condition_dir / "code_attempts"
                code_log_dir.mkdir(parents=True, exist_ok=True)
                filename = f"attempt_{attempt_num:03d}_agent_{agent_id.replace(' ', '_')}.py"
                with open(code_log_dir / filename, 'w') as f:
                    f.write(f"# Problem: {problem_id}\n# Agent: {agent_id}\n# Attempt: {attempt_num}\n\n{code or '# No code generated'}")
                return
            except OSError as e:
                if e.errno == 24:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Warning: [Errno 24] Too many open files. Pausing for {delay}s before retrying to log code...")
                        time.sleep(delay)
                    else:
                        print(f"CRITICAL: Failed to log code for attempt {attempt_num} after {max_retries} retries. Skipping this log.")
                else:
                    print(f"Error logging solution code (non-retryable OSError): {e}")
                    return
            except Exception as e:
                print(f"An unexpected error occurred while logging solution code: {e}")
                return

    def _execute_sandboxed_code(self, solution_code: str, input_grid: List[List[int]]) -> Any:
        """
        Executes code in a separate thread with a timeout.
        """
        if not solution_code:
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_solution, solution_code, input_grid)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return None
            except Exception:
                return None

