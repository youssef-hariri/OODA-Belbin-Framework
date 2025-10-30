# src/utils/code_executor.py

import numpy as np
from typing import List, Any

def execute_solution(solution_code: str, input_grid: List[List[int]]) -> Any:
    """
    A simplified, non-sandboxed executor for running solution code.
    This function is centralized here to prevent circular imports.

    Args:
        solution_code: The string containing the Python code for the solution.
        input_grid: The input grid for the ARC problem.

    Returns:
        The output from the executed 'solve' function, or None if an error occurs.
    """
    if not solution_code:
        return None
    try:
        local_scope = {}
        # Define a safe global environment for the execution
        safe_globals = {
            "__builtins__": {
                'range': range, 'list': list, 'dict': dict, 'int': int, 'len': len,
                'max': max, 'min': min, 'sum': sum, 'zip': zip, 'abs': abs, 'sorted': sorted
            },
            "np": np
        }
        exec(solution_code, safe_globals, local_scope)
        solve_function = local_scope.get('solve')
        
        if callable(solve_function):
            return solve_function(input_grid)
        else:
            return None
    except Exception:
        # Silently fail on execution error, returning None
        return None

