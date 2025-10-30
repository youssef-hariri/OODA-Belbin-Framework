# The LLM Team Composition Paradox: Why the Right Team Isn't Always Diverse: Developing a Contingency Model on the ARC-AGI Benchmark with an OODA-Belbin Framework Inspired by Multicultural Human Teams

This repository contains the official code for the research paper: **"The LLM Team Composition Paradox: Why the Right Team Isn't Always Diverse: Developing a Contingency Model on the ARC-AGI Benchmark with an OODA-Belbin Framework Inspired by Multicultural Human Teams"**.

The experiment investigates the optimal team architecture for Large Language Model (LLM) agents tasked with solving complex "wicked problems" from the Abstraction and Reasoning Corpus (ARC).

[Link to Full Paper (Coming Soon)]

## 📖 Abstract & Key Findings

This research challenges the prevailing assumption that architecturally complex, diverse, or adaptive teams are universally superior for problem-solving. The central finding is a **data-driven contingency model**, which demonstrates that the optimal agentic architecture is contingent on the *type* of problem being solved.

Our findings reveal a series of paradoxes:

  * **The Simplicity Hypothesis:** A significant portion of tasks requiring holistic, abstract reasoning are most effectively solved by a **single agent** (`Baseline_Single_Agent`), where the overhead of teamwork is a detriment.
  * **The "Expert Anomaly":** A non-diverse, **homogeneous team** of 'Implementer' agents (`Baseline_Static_Homogeneous`) uniquely excels at high-precision procedural tasks ("Local Geometric Analysis").
  * **The "Cognitive Labyrinth":** The most algorithmically complex problems are mastered not by the most complex adaptive system, but by a cognitively diverse, **static heterogeneous team** (`Baseline_Static_Heterogeneous`).
  * **The "Cost of Complexity":** The most complex, real-time adaptive framework (`Adaptive_CP-ATS_Budget_Pool`) was the poorest-performing condition, revealing a clear "cost of complexity" in agentic reasoning.

## 🏛️ The Hierarchical OODA-Belbin Framework

The code in this repository implements the final **Hierarchical OODA-Belbin Framework**, a two-layer system designed to operationalize our contingency model.

1.  **Strategic Layer (The Selector):**
    A top-level `StrategicSelectorAgent` first analyzes (or "Orients" to) the problem's features. Instead of picking agents, it picks the *entire problem-solving architecture* (e.g., "Single Agent" or "Homogeneous Team") best suited for that problem type.

2.  **Tactical Layer (The Solver):**
    If an adaptive strategy is chosen, a `TeamManagerAgent` manages the team. If the agents get stuck (a state we call "Generative Exhaustion"), a **Self-Correction Loop** engages, using "Chain-of-Thought" and "Meta-cognitive" prompts to force the agent to debug its own flawed logic.

### Code-to-Concept Map

Here is how the core concepts from the paper map directly to the code in this repository:

| Concept (from Paper) | Code Implementation | File |
| :--- | :--- | :--- |
| **Strategic Selector Agent** | `StrategicSelectorAgent` class | `src/agents/team_manager.py` |
| "OODA Reconnaissance" | `extract_problem_features()`<br>`calculate_innovation_score()` | `src/agents/team_manager.py` |
| Strategic Deployment Rules | `select_strategy()` method | `src/agents/team_manager.py` |
| **Agentic Team Manager (CP-ATS)** | `TeamManagerAgent` class | `src/agents/team_manager.py` |
| Belbin Role Mapping | `BELBIN_ROLE_MAP` dictionary | `src/agents/team_manager.py` |
| Cultural Personas | `PERSONAS_DATA` dictionary | `src/prompts/personas.py` |
| Dynamic Team Recomposition | `diagnose_failure()` & `recompose_team()` | `src/agents/team_manager.py` |
| **Tactical Self-Correction Loop** | `_expand_and_correct()` method | `src/search/ab_mcts.py` |
| "Chain-of-Thought" Prompt | `_create_correction_prompt()` | `src/agents/specialist_agent.py` |
| "Meta-cognitive" Prompt | `_create_meta_cognitive_prompt()` | `src/agents/specialist_agent.py` |
| **Gated Scoring Framework** | `evaluate_solution()` (Gate 1 & 2 checks) | `src/solution_evaluator.py` |
| Multi-Objective Fitness Function | `get_fitness_vector()` method | `src/solution_evaluator.py` |
| **Search Algorithm** | `ABMCTS` class | `src/search/ab_mcts.py` |
| **Main Experiment Orchestrator** | `main.py` | `main.py` |

-----

## 📂 Repository Structure

```bash
.
├── Data tables/
│   ├── Comprehensive successful solutions data and analysis/
│   │   └── Successful solutions data and analysis.csv
│   ├── chi square tables.pdf
│   ├── Summary-Data contamination per condition.csv
│   ├── Summary-Ghost problem contamination (V1 but no V2 version).csv
│   ├── Summary-Percentage of high quality successful solves.csv
│   ├── Summary-Percentage of successful solves per framework-1.csv
│   ├── Summary-Percentage of successful solves per framework.csv
│   ├── Summary-Real success data for the 120 problems experiment.csv
│   ├── Summary-Real successes, Percentage of high quality solutions and data integrity.csv
│   └── Raw successful solutions data and Gemini Pro 2.5 analysis.csv
├── Outputs/
│   ├── output_100_problems_total_failure.zip
│   ├── outputs_120_problems.zip
│   ├── outputs_5_final_runs_20_problems.zip
│   ├── outputs_initial_5_problems.zip
│   └── outputs_repeated_5_problems.zip
├── data - copy ARC dataset inside/
│   └── (ARC-AGI dataset JSON files go here)
├── src/
│   ├── agents/
│   │   ├── specialist_agent.py
│   │   └── team_manager.py
│   ├── prompts/
│   │   └── personas.py
│   ├── search/
│   │   └── ab_mcts.py
│   ├── utils/
│   │   ├── code_executor.py
│   │   └── logger.py
│   ├── llm_client.py
│   ├── problem_loader.py
│   └── solution_evaluator.py
├── main.py
└── requirements.txt
```

-----

## 🚀 How to Run the Experiment

Follow these steps to set up and run the full experiment.

### 1\. Installation

First, clone the repository and install the required Python packages.

```bash
git clone https://github.com/your-username/llm-team-paradox.git
cd llm-team-paradox
```

The `requirements.txt` file lists all necessary libraries. Install them using pip:

```bash
pip install -r requirements.txt
```

### 2\. API Key Configuration

The system uses the DeepSeek Coder model via its API. You must set your API key as an environment variable, which `src/llm_client.py` will read.

```bash
# On macOS/Linux
export DEEPSEEK_API_KEY="your-api-key-here"

# On Windows (in Command Prompt)
set DEEPSEEK_API_KEY="your-api-key-here"
```

### 3\. Dataset Setup

The code expects the ARC-AGI dataset to be in the folder named `data` at the root of the repository.

Your directory structure must look like this:

```bash
llm-team-paradox/
│
├── data/
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   └── ... (any other challenge/solution files)
│
├── src/
│   └── ... (all source code)
│
├── main.py
└── requirements.txt
```

### 4\. Experiment Configuration

You can configure the experiment by editing the global variables at the top of `main.py`:

```python
# main.py

# --- CONFIGURATION ---
DATASET_PATH = "./data" # Make sure this matches your folder name
CHALLENGES_FILE = "json challenges filename"
SOLUTIONS_FILE = "json solutions filename"
OUTPUTS_DIR = Path("outputs")
MAX_ADAPTATION_CYCLES = 5
BUDGET_PER_CYCLE = 50
MAX_CONCURRENT_PROBLEMS = 4 # Warning: Setting this too high may exceed API rate limits
BATCH_SIZE = MAX_CONCURRENT_PROBLEMS
```

### 5\. Run the Experiment

Once configured, you can start the experiment by running `main.py`:

```bash
python main.py
```

The script will:

1.  Initialize all components (LLM client, evaluator, etc.).
2.  Check the `OUTPUTS_DIR` for a `consolidated_results.csv` file to find and skip any problems that have already been processed. The experiment is fully resumable.
3.  Load the remaining problems and process them in batches.
4.  For each problem, the `StrategicSelectorAgent` will choose the optimal strategy.
5.  The chosen framework will run, and all results will be logged to `consolidated_results.csv`.
6.  Detailed logs and all generated code for every attempt will be saved in the `OUTPUTS_DIR` subfolders.

-----

## 📊 Data and Results

This repository includes the complete data used to generate the paper's findings, separated into two categories:

### 1\. Data tables

This folder contains the **final, processed analysis tables** and statistical tests. This is the best place to see the evidence for the paper's conclusions.

  * **`Comprehensive successful solutions data and analysis/`**: Contains the master CSV file detailing every successful solve and, most importantly, the **qualitative code audit** that classified it as a "real success" or "false positive".
  * **`Summary-*.csv`**: Various summary tables detailing data contamination, high-quality success rates, real success percentages, etc..
  * **`chi square tables.pdf`**: The statistical test results, such as the one showing no significant difference in raw success rates.

### 2\. Outputs

This folder contains the **complete raw experimental runs** as `.zip` files. Each file corresponds to a specific Design Science Research (DSR) cycle from the paper. Inside each zip, you will find the raw `.txt` log files, all generated Python code for every attempt, and the CSV results for that specific run.

  * **`outputs_initial_5_problems.zip`**: DSR Cycle 1. This run produced **Anomaly \#1**, the "a priori team selection problem".
  * **`outputs_repeated_5_problems.zip`**: DSR Cycle 2. This run produced **Anomaly \#2**, the "Universal Failure" against the official metric, which proved the need for real-time adaptation.
  * **`outputs_120_problems.zip`**: The main experiment (120 problems) that generated the core contingency model and the four problem archetypes.
  * **`output_100_problems_total_failure.zip`**: DSR Cycle 3. This run against the "Frontier Challenge" (100 eval problems) produced **Anomaly \#3**, "The Local Optima Trap".
  * **`outputs_5_final_runs_20_problems.zip`**: The final validation run (5 runs on 20 "trap" problems) testing the `Self-Correction Loop`, which achieved the first-ever robust solve on a "trap" problem.

-----

## Citation

If you use this research or code in your work, please cite the original paper.

```bibtex
@article{Hariri2025LLMParadox,
  title   = {The LLM Team Composition Paradox: Why the Right Team Isn't Always Diverse},
  author  = {Youssef Hariri},
  journal = {Rennes School of Business and Upgrad},
  year    = {2025}
}
```
