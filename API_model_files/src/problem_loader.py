# src/problem_loader.py

import json
from pathlib import Path
from typing import Dict, Any

class ARCProblemLoader:
    """
    A utility class to load ARC-AGI dataset files from the Kaggle competition format.
    
    This loader handles the parsing of the JSON files containing the challenges
    and their corresponding solutions.
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the ARCProblemLoader.

        Args:
            dataset_path: The root path to the directory containing the ARC dataset JSON files.
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")

    def _load_json_file(self, file_name: str) -> Dict[str, Any]:
        """
        A helper function to load and parse a JSON file.

        Args:
            file_name: The name of the JSON file to load.

        Returns:
            A dictionary containing the parsed JSON data.
        
        Raises:
            FileNotFoundError: If the specified file does not exist in the dataset path.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        file_path = self.dataset_path / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"The file '{file_name}' was not found in '{self.dataset_path}'")
        
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_challenges(self, challenge_file: str) -> Dict[str, Any]:
        """
        Loads the challenge tasks from a specified file.
        
        Each key in the returned dictionary is a task ID, and the value contains
        the 'train' and 'test' pairs for that task.

        Args:
            challenge_file: The filename of the challenge set (e.g., 'arc-agi_training_challenges.json').

        Returns:
            A dictionary of challenge tasks.
        """
        return self._load_json_file(challenge_file)

    def get_solutions(self, solution_file: str) -> Dict[str, Any]:
        """
        Loads the solution data from a specified file.

        Each key in the returned dictionary is a task ID, and the value contains
        the correct output grid(s) for the corresponding test inputs.

        Args:
            solution_file: The filename of the solution set (e.g., 'arc-agi_training_solutions.json').

        Returns:
            A dictionary of solutions.
        """
        return self._load_json_file(solution_file)

if __name__ == '__main__':
    # Example usage.
    # To run this, create a dummy 'data' directory with a sample json file.
    # For example, create 'data/test_challenges.json' with content:
    # { "task1": { "train": [], "test": [] } }
    
    try:
        # Create a dummy directory and file for testing purposes
        dummy_dir = Path("data")
        dummy_dir.mkdir(exist_ok=True)
        dummy_file = dummy_dir / "test_challenges.json"
        with open(dummy_file, 'w') as f:
            json.dump({"task_id_001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}}, f)

        print("--- Testing ARCProblemLoader ---")
        loader = ARCProblemLoader(dataset_path="data")
        
        challenges = loader.get_challenges("test_challenges.json")
        print(f"Successfully loaded {len(challenges)} challenge(s).")
        
        first_task_id = list(challenges.keys())[0]
        print(f"First task ID: {first_task_id}")
        print(f"Data for first task: {challenges[first_task_id]}")

        # Clean up the dummy file and directory
        dummy_file.unlink()
        dummy_dir.rmdir()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

