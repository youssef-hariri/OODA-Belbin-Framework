# src/utils/logger.py

import datetime
from pathlib import Path

class ExperimentLogger:
    """
    A simple logger to write to both a file and the console.
    
    --- MODIFIED ---
    Now opens the file handle once upon initialization and closes it explicitly.
    This is much more efficient and prevents "Too many open files" errors in
    highly concurrent environments.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # --- MODIFIED: Open file once and keep the handle ---
        self._log_file = open(self.log_path, 'w')
        self.log(f"Log started at {datetime.datetime.now().isoformat()}\n{'='*40}")

    def log(self, message: str):
        """Prints a message to the console and writes it to the log file."""
        print(message)
        # --- MODIFIED: Write to the open file handle ---
        self._log_file.write(f"{message}\n")
        self._log_file.flush() # Ensure message is written immediately

    def close(self):
        """Closes the log file handle."""
        self.log(f"\n{'='*40}\nLog finished at {datetime.datetime.now().isoformat()}")
        self._log_file.close()


