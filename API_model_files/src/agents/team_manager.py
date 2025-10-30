# src/agents/team_manager.py

import itertools
import random
import numpy as np
from skimage.measure import label
from typing import List, Dict, Any, Tuple, Optional

# Import project components
from src.prompts.personas import PERSONAS_DATA, ARCHETYPE_INNOVATOR, ARCHETYPE_ADAPTOR
from src.agents.specialist_agent import SpecialistAgent
from src.llm_client import LLMClient
from src.utils.logger import ExperimentLogger

# Expanded mapping of Belbin roles to specific agent personas
BELBIN_ROLE_MAP = {
    'Plant': 'Culture_5',           # Most creative, highest novelty
    'Implementer': 'Culture_Expert',# Most practical, highest elaboration
    'Monitor_Evaluator': 'Culture_10',# Analytical and strategic
    'Completer_Finisher': 'Culture_Neutral', # Detail-oriented
    'Teamworker': 'Culture_4',      # Collaborative, builds consensus
    'Shaper': 'Culture_8'           # Assertive, task-focused, provides drive
}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + NEW: SHARED ANALYSIS UTILITIES                                            +
# + This logic was duplicated and has been refactored into shared functions   +
# + to ensure consistency and maintainability (DRY principle).                +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def count_objects(grid: np.ndarray) -> int:
    """Counts the number of distinct objects in a grid."""
    return label(grid > 0, connectivity=1, background=0, return_num=True)[1]

def extract_problem_features(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts key features from an ARC problem's first training pair."""
    features = {}
    
    # Ensure there is at least one training pair
    if not problem.get('train'):
        return {} # Return empty features if no training data
        
    first_pair = problem['train'][0]
    input_grid = np.array(first_pair['input'])
    output_grid = np.array(first_pair['output'])
    
    features['single_training_example'] = len(problem['train']) == 1
    
    unique_input_colors = set(np.unique(input_grid))
    unique_output_colors = set(np.unique(output_grid))
    features['new_colors_introduced'] = not unique_output_colors.issubset(unique_input_colors)
    
    features['symmetry_based'] = np.array_equal(np.fliplr(input_grid), output_grid)
    features['grid_growth'] = output_grid.size / max(1, input_grid.size)
    
    input_objects = count_objects(input_grid)
    output_objects = count_objects(output_grid)
    features['object_disassembly'] = output_objects > input_objects
    features['stable_object_count'] = output_objects == input_objects
    
    return features

def calculate_innovation_score(features: Dict[str, Any]) -> float:
    """Calculates a score based on problem features to gauge novelty vs. adaptation."""
    score = 0.0
    
    # Factors increasing innovation need (more creative, novel solutions)
    if features.get('single_training_example'): score += 0.5
    if features.get('new_colors_introduced'):   score += 0.4
    if features.get('grid_growth', 0) > 2.0:    score += 0.2
    if features.get('object_disassembly'):      score += 0.3

    # Factors decreasing innovation need (more adaptive, procedural solutions)
    if features.get('symmetry_based'):          score -= 0.6
    if features.get('stable_object_count'):     score -= 0.3
    
    # Clamp the score to a [-1.0, 1.0] range
    return max(-1.0, min(1.0, score))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + NEW: STRATEGIC SELECTOR AGENT                                             +
# + This agent selects the entire framework based on a priori analysis.       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class StrategicSelectorAgent:
    """
    A meta-agent that analyzes a problem and selects the most appropriate 
    problem-solving framework (strategy) to deploy.
    """
    def select_strategy(self, arc_problem: Dict[str, Any], logger: Optional[ExperimentLogger] = None) -> str:
        """Analyzes the problem and returns the name of the optimal strategy to use."""
        log = logger.log if logger else print
        log("--- [Strategic Selector] Analyzing problem to select strategy... ---")
        
        # CORRECTED: Uses the shared utility functions
        features = extract_problem_features(arc_problem)
        innovation_score = calculate_innovation_score(features)

        # Rule 1: Monolithic Task Heuristic
        if features.get('single_training_example') and features.get('stable_object_count') and not features.get('new_colors_introduced'):
            log("[Strategic Selector] Heuristic Triggered: Monolithic Task. Deploying Baseline_Single_Agent.")
            return 'Baseline_Single_Agent'

        # Rule 2: Expert Anomaly Heuristic
        if innovation_score <= -0.5:
            log(f"[Strategic Selector] Heuristic Triggered: Expert Anomaly (Innovation Score: {innovation_score:.2f}). Deploying Baseline_Static_Homogeneous.")
            return 'Baseline_Static_Homogeneous'

        # Rule 3: Cognitive Labyrinth Heuristic
        if innovation_score >= 0.5:
            log(f"[Strategic Selector] Heuristic Triggered: Cognitive Labyrinth (Innovation Score: {innovation_score:.2f}). Deploying Adaptive_CP-ATS_Budget_Pool.")
            return 'Adaptive_CP-ATS_Budget_Pool'

        # Default Rule: General Diversity
        log("[Strategic Selector] Default Rule: General Diversity. Deploying Baseline_Static_Heterogeneous.")
        return 'Baseline_Static_Heterogeneous'


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + ORIGINAL TEAM MANAGER AGENT (Now uses shared utilities)                   +
# + This agent is still required by the adaptive frameworks when they are     +
# + selected by the StrategicSelectorAgent.                                   +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TeamManagerAgent:
    """
    Implements the strategic core of the CP-ATS framework.
    Handles initial team selection and dynamic, real-time adaptation.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.persona_ids = list(PERSONAS_DATA.keys())
        self.team_size = 3
        self.all_possible_teams = list(itertools.combinations(self.persona_ids, self.team_size))
        self.reset_state()

    def reset_state(self):
        """Resets the agent's state for each new problem."""
        self.last_best_score = -1.0
        self.consecutive_failures_no_improvement = 0
        self.last_error_type = None
        self.last_cycle_score = -1.0

    def select_initial_team(self, arc_problem: Dict[str, Any], logger: Optional[ExperimentLogger] = None) -> List[SpecialistAgent]:
        """Selects the initial team using a priori analysis."""
        log = logger.log if logger else print
        log("--- [TeamManager] Selecting initial team for problem... ---")
        
        # CORRECTED: Uses the shared utility functions
        features = extract_problem_features(arc_problem)
        innovation_score = calculate_innovation_score(features)
        
        if innovation_score <= -0.5:
            log("[TeamManager] Expert Anomaly heuristic triggered. Deploying homogeneous 'Implementer' team.")
            best_team_ids = ('Culture_Expert', 'Culture_Expert', 'Culture_Expert')
        else:
            target_profile = self._generate_target_profile(innovation_score)
            log(f"Target Profile: {target_profile}")
            best_team_ids = self._find_best_team_composition(target_profile)

        log(f"Initial Team Selected: {best_team_ids}")
        return [SpecialistAgent(pid, self.llm_client) for pid in best_team_ids]

    def diagnose_failure(self, fitness_vector: Dict[str, float], current_score: float, logger: Optional[ExperimentLogger] = None) -> str:
        """Implements the full, academically-grounded diagnostic framework."""
        log = logger.log if logger else print
        log(f"[TeamManager] Diagnosing failure from fitness vector: {fitness_vector}")

        # State Tracking
        if current_score > self.last_best_score:
            self.last_best_score = current_score
            self.consecutive_failures_no_improvement = 0
        else:
            self.consecutive_failures_no_improvement += 1
        
        current_diagnosis = "Undefined Failure"
        deficit_role = 'Plant'

        # --- Diagnostic Rules (in order of precedence) ---
        if self.last_cycle_score > 0 and current_score < self.last_cycle_score * 0.8:
            current_diagnosis, deficit_role = "Team Disintegration", 'Teamworker'
        elif self.consecutive_failures_no_improvement >= 2:
            current_diagnosis, deficit_role = "Stalled Progress", 'Plant'
        elif fitness_vector.get('prop_obj_count', 0) > (fitness_vector.get('truth_obj_count', 0) + 3):
            current_diagnosis, deficit_role = "Over-Engineering", 'Shaper'
        elif 0.9 < fitness_vector.get('pixel_accuracy', 0) < 1.0:
            if self.last_error_type == 'Precision Error':
                current_diagnosis, deficit_role = "Persistent Precision Error", 'Implementer'
            else:
                current_diagnosis, deficit_role = "Precision Error", 'Completer_Finisher'
        elif fitness_vector.get('obj_count', 0) < 0.8 or fitness_vector.get('jaccard_objects', 0) < 0.8:
            current_diagnosis, deficit_role = "Compositional Error", 'Implementer'
        elif fitness_vector.get('jaccard_objects', 0) > 0.8 and (fitness_vector.get('symmetry', 1.0) < 0.9 or fitness_vector.get('ssim_discrete', 0) < 0.8):
            current_diagnosis, deficit_role = "Arrangement/Shape Error", 'Monitor_Evaluator'
        elif current_score < 0.5:
            current_diagnosis, deficit_role = "Ideation Failure", 'Plant'

        log(f"[TeamManager] Diagnosis: {current_diagnosis} -> '{deficit_role}' deficit.")
        self.last_error_type = current_diagnosis
        self.last_cycle_score = current_score
        return deficit_role

    def recompose_team(self, current_team: List[SpecialistAgent], deficit: str, logger: Optional[ExperimentLogger] = None) -> List[SpecialistAgent]:
        """Swaps an agent to address the diagnosed deficit."""
        log = logger.log if logger else print
        log(f"[TeamManager] Recomposing team to address '{deficit}' deficit.")
        current_team_ids = [agent.persona_id for agent in current_team]
        
        required_agent_id = BELBIN_ROLE_MAP.get(deficit)
        if not required_agent_id:
            log(f"[TeamManager] Warning: No agent for role {deficit}. No change.")
            return current_team

        # Prioritize removing non-essential roles
        non_essential = [pid for pid in current_team_ids if pid not in [BELBIN_ROLE_MAP['Plant'], BELBIN_ROLE_MAP['Implementer']]]
        agent_to_remove = random.choice(non_essential) if non_essential else random.choice(current_team_ids)
            
        new_team_ids = list(current_team_ids)
        new_team_ids.remove(agent_to_remove)
        new_team_ids.append(required_agent_id)
        
        log(f"[TeamManager] New team: {sorted(new_team_ids)}")
        return [SpecialistAgent(pid, self.llm_client) for pid in new_team_ids]

    # --- Helper methods specific to TeamManagerAgent ---
    def _generate_target_profile(self, score: float) -> Dict[str, float]:
        """Generates a target persona profile based on the innovation score."""
        weight_innovator = (score + 1) / 2
        weight_adaptor = 1 - weight_innovator
        target_profile = {
            k: round((weight_innovator * ARCHETYPE_INNOVATOR[k]) + (weight_adaptor * ARCHETYPE_ADAPTOR[k]), 4)
            for k in ARCHETYPE_INNOVATOR
        }
        return target_profile

    def _find_best_team_composition(self, target: Dict[str, float]) -> Tuple[str, ...]:
        """Finds the team whose average profile is closest to the target profile."""
        best_team, min_dist = None, float('inf')
        target_vector = np.array(list(target.values()))
        
        for team_ids in self.all_possible_teams:
            team_profiles = [np.array(list(PERSONAS_DATA[pid]['scores'].values())) for pid in team_ids]
            avg_profile = np.mean(team_profiles, axis=0)
            distance = np.linalg.norm(target_vector - avg_profile)
            
            if distance < min_dist:
                min_dist = distance
                best_team = team_ids
                
        return best_team

    def get_dynamic_weights(self, arc_problem: Dict[str, Any]) -> Dict[str, float]:
        """Generates dynamic weights for the fitness function based on problem features."""
        features = extract_problem_features(arc_problem)
        weights = {
            "pixel_accuracy": 1.0, "ssim_discrete": 1.0, "hist_bc": 1.0,
            "obj_count": 1.0, "jaccard_objects": 1.0, "symmetry": 1.0,
            "euler": 1.0, "graph_spectral": 0.5
        }
        if features.get('symmetry_based'):
            weights['symmetry'] *= 5.0
        if features.get('object_disassembly') or features.get('stable_object_count'):
            weights['obj_count'] *= 3.0
            weights['jaccard_objects'] *= 3.0
        if features.get('new_colors_introduced'):
            weights['hist_bc'] *= 4.0
            
        return weights
