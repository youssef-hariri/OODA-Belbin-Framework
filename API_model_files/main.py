# main.py

import csv
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
import shutil
import numpy as np
import concurrent.futures
import time
import os
import threading

# Import project components
from src.llm_client import LLMClient
from src.problem_loader import ARCProblemLoader
from src.solution_evaluator import SolutionEvaluator
from src.agents.specialist_agent import SpecialistAgent
from src.agents.team_manager import TeamManagerAgent, StrategicSelectorAgent
from src.search.ab_mcts import ABMCTS
from src.utils.logger import ExperimentLogger
from src.utils.code_executor import execute_solution

# --- CONFIGURATION ---
DATASET_PATH = "./data"
CHALLENGES_FILE = "missing_challenges_from 20_above_0.9.json"
SOLUTIONS_FILE = "missing_solutions_from_20_above_0.9.json"
OUTPUTS_DIR = Path("outputs_missing_from_20_above_0.9")
MAX_ADAPTATION_CYCLES = 5
BUDGET_PER_CYCLE = 50
MAX_CONCURRENT_PROBLEMS = 4
BATCH_SIZE = MAX_CONCURRENT_PROBLEMS

# --- Global lock for writing to the main results file ---
csv_writer_lock = threading.Lock()

def get_completed_problems(results_file_path: Path) -> Set[str]:
    """Reads the results CSV to find which problems are already completed."""
    if not results_file_path.is_file():
        return set()
    
    completed_problems = set()
    try:
        with open(results_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or 'problem_id' not in header: return set()
            
            problem_id_index = header.index('problem_id')
            for row in reader:
                if row and len(row) > problem_id_index:
                    problem_id = row[problem_id_index]
                    completed_problems.add(problem_id)
    except (FileNotFoundError, StopIteration, ValueError):
        return set()
    return completed_problems

def setup_experiment_results_file(results_file_path: Path):
    """Creates the results CSV file with headers if it doesn't exist."""
    if not results_file_path.parent.exists():
        results_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results_file_path.is_file() or os.path.getsize(results_file_path) == 0:
        with open(results_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "problem_id", "selected_strategy", "cycle", "team_composition", "final_weighted_score", "official_score", "best_solution_code", "fitness_vector", "gate_status"])

def log_result(results_file_path: Path, problem_id: str, condition: str, cycle: int, team: List[SpecialistAgent], score: float, official_score: int, solution: str, fitness_vector: Dict, gate_status: str):
    """Logs a single row of results to the main CSV file in a thread-safe manner."""
    timestamp = datetime.datetime.now().isoformat()
    team_ids = sorted([agent.persona_id for agent in team])
    team_str = ", ".join(team_ids)
    
    with csv_writer_lock:
        with open(results_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, problem_id, condition, cycle, team_str, score, official_score, solution, json.dumps(fitness_vector), gate_status])

# ======================================================================
# COMPLETE IMPLEMENTATIONS OF THE RUN CONDITIONS
# ======================================================================

def run_adaptive_cp_ats(problem_id, problem_data, llm_client, evaluator, team_manager, results_file_path, logger):
    condition_name = "Adaptive_CP-ATS_Fixed_Cycle"
    logger.log(f"\n{'='*20} Running {condition_name} for Problem: {problem_id} {'='*20}")
    
    team_manager.reset_state()
    current_team = team_manager.select_initial_team(problem_data, logger)
    total_attempts = 0
    for cycle in range(1, MAX_ADAPTATION_CYCLES + 1):
        logger.log(f"\n--- Adaptation Cycle {cycle}/{MAX_ADAPTATION_CYCLES} ---")
        dynamic_weights = team_manager.get_dynamic_weights(problem_data)
        mcts = ABMCTS(team=current_team, evaluator=evaluator, problem_id=problem_id, arc_problem=problem_data, dynamic_weights=dynamic_weights, condition_name=condition_name, budget=BUDGET_PER_CYCLE, start_iteration=total_attempts, logger=logger, early_stopping_enabled=True)
        
        best_solution_code, iterations_this_cycle, official_score, gate_status = mcts.search()
        total_attempts += iterations_this_cycle
        
        test_input = problem_data['test'][0]['input']
        correct_output = problem_data['test'][0]['output']
        fitness_vector = evaluator.get_fitness_vector(best_solution_code, test_input, correct_output)
        final_weighted_score = mcts.best_score_found if mcts.best_score_found > 0 else 0.0
        
        logger.log(f"[Evaluation] Cycle {cycle} Official Score: {official_score}, Weighted Score: {final_weighted_score:.4f} (used {iterations_this_cycle} iterations)")
        log_result(results_file_path, problem_id, condition_name, cycle, current_team, final_weighted_score, official_score, best_solution_code, fitness_vector, gate_status)
        
        if official_score == 1:
            logger.log(f"\n[SUCCESS] Problem {problem_id} solved in cycle {cycle}!")
            return
        if cycle < MAX_ADAPTATION_CYCLES:
            deficit = team_manager.diagnose_failure(fitness_vector, final_weighted_score, logger)
            current_team = team_manager.recompose_team(current_team, deficit, logger)
            
    logger.log(f"\n[FAILURE] Problem {problem_id} not solved after {MAX_ADAPTATION_CYCLES} cycles.")

def run_adaptive_budget_pool(problem_id: str, problem_data: Dict, llm_client: LLMClient, evaluator: SolutionEvaluator, team_manager: TeamManagerAgent, results_file_path: Path, logger: ExperimentLogger):
    condition_name = "Adaptive_CP-ATS_Budget_Pool"
    logger.log(f"\n{'='*20} Running {condition_name} for Problem: {problem_id} {'='*20}")
    
    team_manager.reset_state()
    current_team = team_manager.select_initial_team(problem_data, logger)
    total_budget = MAX_ADAPTATION_CYCLES * BUDGET_PER_CYCLE
    total_attempts = 0
    cycle = 1
    while total_attempts < total_budget:
        logger.log(f"\n--- Adaptation Cycle {cycle} (Total Attempts: {total_attempts}/{total_budget}) ---")
        remaining_budget = total_budget - total_attempts
        budget_for_this_cycle = min(BUDGET_PER_CYCLE, remaining_budget)
        dynamic_weights = team_manager.get_dynamic_weights(problem_data)
        mcts = ABMCTS(team=current_team, evaluator=evaluator, problem_id=problem_id, arc_problem=problem_data, dynamic_weights=dynamic_weights, condition_name=condition_name, budget=budget_for_this_cycle, start_iteration=total_attempts, logger=logger, early_stopping_enabled=True)
        
        best_solution_code, iterations_this_cycle, official_score, gate_status = mcts.search()
        total_attempts += iterations_this_cycle
        
        test_input = problem_data['test'][0]['input']
        correct_output = problem_data['test'][0]['output']
        fitness_vector = evaluator.get_fitness_vector(best_solution_code, test_input, correct_output)
        final_weighted_score = mcts.best_score_found if mcts.best_score_found > 0 else 0.0
        
        logger.log(f"[Evaluation] Cycle {cycle} Official Score: {official_score}, Weighted Score: {final_weighted_score:.4f} (used {iterations_this_cycle} iterations)")
        log_result(results_file_path, problem_id, condition_name, cycle, current_team, final_weighted_score, official_score, best_solution_code, fitness_vector, gate_status)
        
        if official_score == 1:
            logger.log(f"\n[SUCCESS] Problem {problem_id} solved in cycle {cycle} after {total_attempts} total iterations!")
            return
            
        deficit = team_manager.diagnose_failure(fitness_vector, final_weighted_score, logger)
        current_team = team_manager.recompose_team(current_team, deficit, logger)
        cycle += 1
        
    logger.log(f"\n[FAILURE] Problem {problem_id} not solved within the total budget of {total_budget} iterations.")

def run_static_condition(condition_name, problem_id, problem_data, llm_client, evaluator, team_manager, results_file_path, logger):
    logger.log(f"\n--- Running Static Condition: {condition_name} for Problem: {problem_id} ---")
    
    if condition_name == "CP-ATS_Static":
        team = team_manager.select_initial_team(problem_data, logger)
    elif condition_name == "Baseline_Static_Heterogeneous":
        team = [SpecialistAgent(pid, llm_client) for pid in ["Culture_5", "Culture_13", "Culture_Expert"]]
    elif condition_name == "Baseline_Static_Homogeneous":
        team = [SpecialistAgent(pid, llm_client) for pid in ["Culture_Expert", "Culture_Expert", "Culture_Expert"]]
    else: # Baseline_Single_Agent
        team = [SpecialistAgent("Culture_Neutral", llm_client)]
    
    dynamic_weights = team_manager.get_dynamic_weights(problem_data)
    total_budget = MAX_ADAPTATION_CYCLES * BUDGET_PER_CYCLE
    
    mcts = ABMCTS(team=team, evaluator=evaluator, problem_id=problem_id, arc_problem=problem_data, dynamic_weights=dynamic_weights, condition_name=condition_name, budget=total_budget, start_iteration=0, logger=logger, early_stopping_enabled=False)
    
    best_solution_code, _, official_score, gate_status = mcts.search()
    
    test_input = problem_data['test'][0]['input']
    correct_output = problem_data['test'][0]['output']
    final_weighted_score = mcts.best_score_found if mcts.best_score_found > 0 else 0.0
    fitness_vector = evaluator.get_fitness_vector(best_solution_code, test_input, correct_output)
    
    logger.log(f"--- Finished Problem {problem_id}. Final Weighted Score: {final_weighted_score:.4f}, Official Score: {official_score} ---")
    log_result(results_file_path, problem_id, condition_name, 0, team, final_weighted_score, official_score, best_solution_code, fitness_vector, gate_status)

# ======================================================================
# CONCURRENTLY-EFFICIENT WORKER FUNCTIONS
# ======================================================================

def process_single_problem(problem_info, results_file_path, llm_client, evaluator, team_manager, selector_agent):
    """
    Encapsulates all work for a single problem, using shared components.
    """
    problem_id, problem_data, solutions = problem_info
    
    print(f"STARTING PROCESSING FOR PROBLEM: {problem_id}")
    if problem_id not in solutions:
        print(f"Warning: No solution for {problem_id}. Skipping.")
        return problem_id

    problem_data['test'][0]['output'] = solutions[problem_id][0]
    
    problem_dir = OUTPUTS_DIR / problem_id
    logger = ExperimentLogger(log_path=problem_dir / "run_log.txt")
    
    chosen_strategy = "None"
    try:
        chosen_strategy = selector_agent.select_strategy(problem_data, logger)
        
        if chosen_strategy == "Adaptive_CP-ATS_Fixed_Cycle":
            run_adaptive_cp_ats(problem_id, problem_data, llm_client, evaluator, team_manager, results_file_path, logger)
        elif chosen_strategy == "Adaptive_CP-ATS_Budget_Pool":
            run_adaptive_budget_pool(problem_id, problem_data, llm_client, evaluator, team_manager, results_file_path, logger)
        else:
            run_static_condition(chosen_strategy, problem_id, problem_data, llm_client, evaluator, team_manager, results_file_path, logger)
            
    except Exception as e:
        error_msg = f"CRITICAL ERROR on {chosen_strategy} for {problem_id}: {e}"
        print(error_msg)
        logger.log(error_msg)
        log_result(results_file_path, problem_id, "STRATEGY_SELECTOR_ERROR", -1, [], -1.0, 0, f"ERROR: {e}", {}, "ERROR")
    finally:
        logger.close()
    
    print(f"FINISHED PROCESSING FOR PROBLEM: {problem_id}")
    return problem_id

def process_problem_batch(problem_batch, results_file, llm_client, evaluator, team_manager, selector_agent):
    """
    Creates a temporary ThreadPoolExecutor to process one batch of problems.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS) as executor:
        future_to_problem = {
            executor.submit(process_single_problem, p_info, results_file, llm_client, evaluator, team_manager, selector_agent): p_info[0]
            for p_info in problem_batch
        }
        for future in concurrent.futures.as_completed(future_to_problem):
            problem_id = future_to_problem[future]
            try:
                result = future.result()
                print(f"Successfully completed processing for problem: {result}")
            except Exception as exc:
                print(f"Problem {problem_id} generated an exception: {exc}")

# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    print(f"Results will be saved in '{OUTPUTS_DIR}'")
    
    results_file = OUTPUTS_DIR / "consolidated_results.csv"
    setup_experiment_results_file(results_file)

    try:
        print("Initializing components...")
        problem_loader = ARCProblemLoader(dataset_path=DATASET_PATH)
        print(f"Loading dataset from: '{DATASET_PATH}'")
        challenges = problem_loader.get_challenges(CHALLENGES_FILE)
        solutions = problem_loader.get_solutions(SOLUTIONS_FILE)
        print("Dataset loaded successfully.")

        # Initialize all shared components ONCE
        llm_client = LLMClient()
        evaluator = SolutionEvaluator()
        team_manager = TeamManagerAgent(llm_client)
        selector_agent = StrategicSelectorAgent()
        print("Shared components loaded and ready.")

    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return
    
    all_problem_list = [(pid, data, solutions) for pid, data in challenges.items()]
    completed_problems = get_completed_problems(results_file)
    
    if completed_problems:
        print(f"Found {len(completed_problems)} completed problems in results file. Skipping them.")
        problems_to_run = [p for p in all_problem_list if p[0] not in completed_problems]
    else:
        problems_to_run = all_problem_list
        
    if not problems_to_run:
        print("All problems have already been processed. Experiment complete.")
        return
        
    total_problems_to_run = len(problems_to_run)
    print(f"\nStarting/Resuming Strategic Selector experiment for {total_problems_to_run} remaining problems...")
    print(f"Processing in batches of {BATCH_SIZE} with up to {MAX_CONCURRENT_PROBLEMS} concurrent workers per batch.")

    for i in range(0, total_problems_to_run, BATCH_SIZE):
        batch = problems_to_run[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_problems_to_run + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n{'='*20} PROCESSING BATCH {batch_num}/{total_batches} {'='*20}")
        # Pass the initialized components to the batch processor
        process_problem_batch(batch, results_file, llm_client, evaluator, team_manager, selector_agent)
        print(f"--- COMPLETED BATCH {batch_num}/{total_batches} ---")
        time.sleep(5)

    print(f"\n\n{'='*20}\nExperiment Complete!\n{'='*20}\nAll results saved to '{results_file}'.")

if __name__ == "__main__":
    main()
