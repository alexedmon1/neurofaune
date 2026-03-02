"""Lightweight progress tracking for long-running analysis scripts."""

import json
import os
from datetime import datetime
from pathlib import Path

_FILENAME = "_progress.json"


class AnalysisProgress:
    """Writes a progress file that persists across interruptions.

    The file is kept on completion as a record (status changes to "completed").
    """

    def __init__(self, output_dir: Path, script: str, total_tasks: int):
        self.path = Path(output_dir) / _FILENAME
        self._state = {
            "script": script,
            "pid": os.getpid(),
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "total_tasks": total_tasks,
            "completed_tasks": 0,
            "failed_tasks": 0,
        }
        self._write()

    def update(self, *, task: str, phase: str, completed: int, failed: int = 0):
        """Write current state to _progress.json (atomic overwrite)."""
        self._state["current_task"] = task
        self._state["current_phase"] = phase
        self._state["completed_tasks"] = completed
        self._state["failed_tasks"] = failed
        self._state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write()

    def finish(self):
        """Mark analysis as complete."""
        self._state["status"] = "completed"
        self._state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._state.pop("current_task", None)
        self._state.pop("current_phase", None)
        self._write()

    def _write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.rename(self.path)  # atomic on POSIX
